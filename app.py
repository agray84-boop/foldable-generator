import streamlit as st
import os
import re
import tempfile
from typing import List, Tuple, Optional, Dict

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Parsing / normalization (Word copy/paste friendly)
# ============================================================

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # 2x -> 2*x, (x+1)(x-2) -> *
    convert_xor,                          # x^2 supported
)

def _normalize_word_text(s: str) -> str:
    """
    Normalize what teachers paste from Word.
    - unicode minus and multiplication symbols
    - supports unicode radicals √63, √(x+1)
    - interprets classroom shorthand 1/2x as (1/2)*x
    """
    s = (s or "").strip()
    s = s.replace("−", "-").replace("·", "*").replace("×", "*")
    s = re.sub(r"\s+", " ", s).strip()

    # Convert unicode superscripts x⁶ -> x^6 (harmless if not used)
    sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    s = re.sub(
        r"([A-Za-z])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda m: m.group(1) + "^" + m.group(2).translate(sup_map),
        s
    )

    # Interpret classroom shorthand: 1/2x means (1/2)*x
    s = re.sub(r'(\d+)\s*/\s*(\d+)\s*([A-Za-z])', r'(\1/\2)*\3', s)

    # --- RADICAL SUPPORT ---
    # √(something) -> sqrt(something)
    s = re.sub(r"√\s*\(([^)]+)\)", r"sqrt(\1)", s)
    # √63, √x -> sqrt(63), sqrt(x)
    s = re.sub(r"√\s*([A-Za-z0-9]+)", r"sqrt(\1)", s)

    # Ensure explicit multiplication: 5sqrt(63) -> 5*sqrt(63)
    s = re.sub(r"(\d)\s*sqrt\(", r"\1*sqrt(", s)
    s = re.sub(r"\)\s*sqrt\(", r")*sqrt(", s)

    return s

def _parse(s: str, *, evaluate: bool = True) -> sp.Expr:
    s = _normalize_word_text(s)
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=evaluate)


# ============================================================
# Smart prompt detection / parsing
# ============================================================

def _looks_like_composition_prompt(t: str) -> bool:
    t_low = t.lower()
    return ("f(g(x))" in t_low or "fog" in t_low or "compose" in t_low or "composition" in t_low) and ("f(x" in t_low and "g(x" in t_low)

def _parse_composition_prompt(t: str) -> Optional[Dict[str, str]]:
    """
    Accepts something like:
    "find f(g(x)), if g(x) = x+3 and f(x)=x^2+3"
    Returns dict with keys: f, g, var (default x)
    """
    s = _normalize_word_text(t)

    # Try to find f(x)=... and g(x)=...
    m_f = re.search(r"f\s*\(\s*x\s*\)\s*=\s*([^,;]+)", s, flags=re.IGNORECASE)
    m_g = re.search(r"g\s*\(\s*x\s*\)\s*=\s*([^,;]+)", s, flags=re.IGNORECASE)

    if not (m_f and m_g):
        return None

    f_expr = m_f.group(1).strip()
    g_expr = m_g.group(1).strip()
    return {"f": f_expr, "g": g_expr, "var": "x"}


# ============================================================
# Worked-solution line builders (math-only lines, final uses ∴)
# ============================================================

def _linear_solution_lines(left: sp.Expr, right: sp.Expr, x: sp.Symbol) -> List[str]:
    expr = sp.simplify(left - right)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x)
    b = poly.coeff_monomial(1)

    lines: List[str] = []
    lines.append(sp.latex(sp.Eq(left, right)))
    lines.append(sp.latex(sp.Eq(expr, 0)))
    lines.append(sp.latex(sp.Eq(a * x, -b)))

    if a == 0:
        lines.append(r"\therefore\ \mathrm{no\ unique\ solution}")
        return lines[:5]

    sol = sp.simplify(-b / a)
    lines.append(sp.latex(sp.Eq(x, sol)))
    lines.append(r"\therefore\ x=" + sp.latex(sol))
    return lines[:5]

def _quadratic_solution_lines(left: sp.Expr, right: sp.Expr, x: sp.Symbol) -> List[str]:
    expr = sp.simplify(left - right)
    lines: List[str] = []
    lines.append(sp.latex(sp.Eq(left, right)))
    lines.append(sp.latex(sp.Eq(expr, 0)))

    factored = sp.factor(expr)
    if factored != expr:
        lines.append(sp.latex(sp.Eq(factored, 0)))
        sols = sp.solve(sp.Eq(expr, 0), x)
        sols = [sp.simplify(s) for s in sols]
        if len(sols) == 1:
            lines.append(r"\therefore\ x=" + sp.latex(sols[0]))
        elif len(sols) >= 2:
            lines.append(r"\therefore\ x=" + sp.latex(sols[0]) + r",\ x=" + sp.latex(sols[1]))
        else:
            lines.append(r"\therefore\ \mathrm{no\ real\ solutions}")
        return lines[:5]

    # Quadratic formula (compact)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x**2)
    b = poly.coeff_monomial(x)
    c = poly.coeff_monomial(1)
    disc = sp.simplify(b**2 - 4*a*c)

    lines.append(r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}")
    lines.append(r"x=\frac{" + sp.latex(-b) + r"\pm\sqrt{" + sp.latex(disc) + r"}}{" + sp.latex(2*a) + r"}")

    sols = sp.solve(sp.Eq(expr, 0), x)
    sols = [sp.simplify(s) for s in sols]
    if len(sols) == 1:
        lines.append(r"\therefore\ x=" + sp.latex(sols[0]))
    elif len(sols) >= 2:
        lines.append(r"\therefore\ x=" + sp.latex(sols[0]) + r",\ x=" + sp.latex(sols[1]))
    else:
        lines.append(r"\therefore\ \mathrm{no\ real\ solutions}")

    return lines[:5]

def _simplify_expression_lines(expr: sp.Expr) -> List[str]:
    simp = sp.simplify(expr)
    return [sp.latex(expr), sp.latex(simp), r"\therefore\ " + sp.latex(simp)]

def _simplify_rational_lines(expr: sp.Expr) -> List[str]:
    # show original, factored, canceled, therefore
    num, den = sp.fraction(sp.together(expr))
    num_f = sp.factor(num)
    den_f = sp.factor(den)
    canceled = sp.cancel(expr)

    lines = [
        sp.latex(expr),
        r"\frac{" + sp.latex(num_f) + r"}{" + sp.latex(den_f) + r"}",
        sp.latex(canceled),
        r"\therefore\ " + sp.latex(canceled),
    ]
    return lines[:5]

def _simplify_radical_lines(expr: sp.Expr) -> List[str]:
    # For things like 5*sqrt(63), sympy.simplify handles it well
    simp = sp.simplify(expr)
    # Provide an intermediate where possible by pulling square factors (optional)
    # We'll show original, simplified, therefore
    return [sp.latex(expr), sp.latex(simp), r"\therefore\ " + sp.latex(simp)]

def _compose_functions_lines(f_expr: sp.Expr, g_expr: sp.Expr, x: sp.Symbol) -> List[str]:
    # f(g(x)) = substitute g into f
    composed = sp.simplify(f_expr.subs(x, g_expr))
    lines = [
        r"g(x)=" + sp.latex(g_expr),
        r"f(x)=" + sp.latex(f_expr),
        r"f(g(x))=" + sp.latex(f_expr.subs(x, g_expr)),
        r"\therefore\ f(g(x))=" + sp.latex(composed),
    ]
    return lines[:5]

def _solve_system_2x2_lines(eq1: sp.Eq, eq2: sp.Eq) -> List[str]:
    x, y = sp.symbols("x y")
    sol = sp.solve((eq1, eq2), (x, y), dict=True)
    lines = [sp.latex(eq1), sp.latex(eq2)]
    if sol:
        s0 = sol[0]
        lines.append(r"x=" + sp.latex(s0.get(x)))
        lines.append(r"y=" + sp.latex(s0.get(y)))
        lines.append(r"\therefore\ (x,y)=(" + sp.latex(s0.get(x)) + r"," + sp.latex(s0.get(y)) + r")")
    else:
        lines.append(r"\therefore\ \mathrm{no\ solution}")
    return lines[:5]


# ============================================================
# Decide problem type + build problem latex + solution lines
# ============================================================

def compute_problem_and_solution_lines(raw: str, var: str = "x") -> Tuple[str, Optional[List[str]]]:
    """
    Returns:
      problem_latex: render EXACT structure (no pre-simplify) when possible
      solution_lines: worked steps (math-only) or None if parsing fails
    """
    raw_norm = _normalize_word_text(raw)
    if not raw_norm:
        return r"\mathrm{(blank)}", None

    # --- Smart composition prompt ---
    if _looks_like_composition_prompt(raw_norm):
        parsed = _parse_composition_prompt(raw_norm)
        if parsed:
            x = sp.Symbol(parsed["var"])
            # Display (keep structure)
            try:
                f_disp = _parse(parsed["f"], evaluate=False)
                g_disp = _parse(parsed["g"], evaluate=False)
                problem_ltx = r"\mathrm{Find}\ f(g(x))\ \mathrm{if}\ g(x)=" + sp.latex(g_disp) + r"\ \mathrm{and}\ f(x)=" + sp.latex(f_disp)
            except Exception:
                problem_ltx = raw_norm

            # Solve
            try:
                f = _parse(parsed["f"], evaluate=True)
                g = _parse(parsed["g"], evaluate=True)
                lines = _compose_functions_lines(f, g, x)
                return problem_ltx, lines
            except Exception:
                return problem_ltx, None

    # --- 2x2 systems: separated by ';' ---
    if ";" in raw_norm and raw_norm.count("=") >= 2:
        parts = [p.strip() for p in raw_norm.split(";") if p.strip()]
        if len(parts) >= 2:
            try:
                x, y = sp.symbols("x y")
                def parse_eq(s: str) -> sp.Eq:
                    L, R = s.split("=", 1)
                    return sp.Eq(_parse(L, evaluate=True), _parse(R, evaluate=True))
                eq1 = parse_eq(parts[0])
                eq2 = parse_eq(parts[1])

                # Display (no simplify)
                try:
                    L1, R1 = parts[0].split("=", 1)
                    L2, R2 = parts[1].split("=", 1)
                    eq1_disp = sp.Eq(_parse(L1, evaluate=False), _parse(R1, evaluate=False), evaluate=False)
                    eq2_disp = sp.Eq(_parse(L2, evaluate=False), _parse(R2, evaluate=False), evaluate=False)
                    problem_ltx = sp.latex(eq1_disp) + r",\ " + sp.latex(eq2_disp)
                except Exception:
                    problem_ltx = raw_norm

                return problem_ltx, _solve_system_2x2_lines(eq1, eq2)
            except Exception:
                return raw_norm, None

    # --- Equation ---
    if "=" in raw_norm:
        left_s, right_s = raw_norm.split("=", 1)
        x = sp.Symbol(var)

        # Display parse (no simplify)
        try:
            left_disp = _parse(left_s, evaluate=False)
            right_disp = _parse(right_s, evaluate=False)
            problem_ltx = sp.latex(sp.Eq(left_disp, right_disp, evaluate=False))
        except Exception:
            problem_ltx = raw_norm

        # Solve parse
        try:
            left = _parse(left_s, evaluate=True)
            right = _parse(right_s, evaluate=True)
        except Exception:
            return problem_ltx, None

        expr = sp.simplify(left - right)
        try:
            deg = sp.Poly(expr, x).degree()
        except Exception:
            deg = 1

        if deg == 2:
            return problem_ltx, _quadratic_solution_lines(left, right, x)
        return problem_ltx, _linear_solution_lines(left, right, x)

    # --- Expression ---
    # Display (no simplify)
    try:
        expr_disp = _parse(raw_norm, evaluate=False)
        problem_ltx = sp.latex(expr_disp)
    except Exception:
        problem_ltx = raw_norm

    # Solve (try several “types”)
    try:
        expr = _parse(raw_norm, evaluate=True)
    except Exception:
        return problem_ltx, None

    # If contains sqrt, treat as radical simplify
    if "sqrt(" in raw_norm:
        return problem_ltx, _simplify_radical_lines(expr)

    # If looks like rational (division), show rational steps
    if "/" in raw_norm:
        return problem_ltx, _simplify_rational_lines(expr)

    # Default expression simplify
    return problem_ltx, _simplify_expression_lines(expr)


# ============================================================
# Render single-line LaTeX to PNG with AUTO-FIT
# ============================================================

def _clean_for_mathtext(latex: str) -> str:
    s = (latex or "").strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.replace(r"\displaystyle", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)  # just in case
    return s.strip()

def _render_math_png(math_latex: str, font_size: int = 16, dpi: int = 300) -> str:
    math_latex = _clean_for_mathtext(math_latex)
    math_latex = f"${math_latex}$"

    # Measure
    fig = plt.figure()
    fig.patch.set_alpha(0)
    t = fig.text(0, 0, math_latex, fontsize=font_size)
    fig.canvas.draw()
    bbox = t.get_window_extent()
    w, h = bbox.width / dpi, bbox.height / dpi
    plt.close(fig)

    # Render
    fig = plt.figure(figsize=(w + 0.35, h + 0.35), dpi=dpi)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, math_latex, fontsize=font_size, ha="center", va="center")

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path

def _draw_math_centered_autofit(c: Canvas, math_latex: str, cx: float, cy: float, w: float, h: float,
                               base_font: int = 16, min_font: int = 10):
    """
    Auto-fit by lowering font size until the rendered image fits nicely.
    """
    for fs in range(base_font, min_font - 1, -1):
        path = _render_math_png(math_latex, font_size=fs)
        try:
            img = ImageReader(path)
            iw, ih = img.getSize()
            scale = min(w / iw, h / ih)
            # If it fits without needing extreme downscaling, accept
            if scale >= 0.98 or fs == min_font:
                dw, dh = iw * min(scale, 1.0), ih * min(scale, 1.0)
                c.drawImage(img, cx - dw / 2, cy - dh / 2, dw, dh, mask="auto")
                return
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


# ============================================================
# PDF layout: HORIZONTAL bands like your sample
# ============================================================

def build_foldable_pdf(out_path: str,
                       eq1_ltx: str, eq1_lines: Optional[List[str]], instr1: str,
                       eq2_ltx: str, eq2_lines: Optional[List[str]], instr2: str,
                       eq3_ltx: str, eq3_lines: Optional[List[str]], instr3: str):
    page_w, page_h = letter
    c = Canvas(out_path, pagesize=letter)

    # Band boundaries from TOP: 0, 2.75, 5.5, 8.25, 11 inches
    y_top = page_h
    y_marks_from_top = [0.0, 2.75 * inch, 5.5 * inch, 8.25 * inch, 11.0 * inch]
    ys = [y_top - m for m in y_marks_from_top]  # [top, line1, line2, line3, bottom]

    def fold_lines_horizontal():
        c.saveState()
        c.setDash(6, 6)
        c.setLineWidth(1)
        for y in (ys[1], ys[2], ys[3]):
            c.line(0.5 * inch, y, page_w - 0.5 * inch, y)
        c.restoreState()

    def draw_boxed_label(band_top_y: float, text: str):
        box_w, box_h = 0.55 * inch, 0.35 * inch
        pad_x = 0.35 * inch
        pad_y = 0.35 * inch
        c.saveState()
        c.setLineWidth(1.2)
        c.rect(pad_x, band_top_y - pad_y - box_h, box_w, box_h, stroke=1, fill=0)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(pad_x + box_w / 2, band_top_y - pad_y - box_h + 0.10 * inch, text)
        c.restoreState()

    def band_rect(band_index: int) -> Tuple[float, float]:
        return ys[band_index], ys[band_index + 1]

    def band_content_box(band_index: int) -> Tuple[float, float, float, float]:
        band_top,_







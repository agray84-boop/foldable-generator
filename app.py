import streamlit as st
import os
import re
import tempfile
from typing import List, Tuple, Optional

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
# 1) Parsing utilities (Word-friendly + better radicals/fractions)
# ============================================================

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # 2x -> 2*x, (x+1)(x-2) -> *
    convert_xor,                          # x^2 supported
)

# Unicode fraction map (Word sometimes pastes these)
_UFRAC = {
    "½": "(1/2)",
    "⅓": "(1/3)",
    "⅔": "(2/3)",
    "¼": "(1/4)",
    "¾": "(3/4)",
    "⅕": "(1/5)",
    "⅖": "(2/5)",
    "⅗": "(3/5)",
    "⅘": "(4/5)",
    "⅙": "(1/6)",
    "⅚": "(5/6)",
    "⅛": "(1/8)",
    "⅜": "(3/8)",
    "⅝": "(5/8)",
    "⅞": "(7/8)",
}

def _normalize_word_text(s: str) -> str:
    """
    Normalize what teachers paste from Word.
    Supports:
      - unicode minus and multiplication symbols
      - unicode fractions (½ etc)
      - mixed numbers: "1 1/2" -> "(1+1/2)"
      - classroom shorthand: 1/2x -> (1/2)*x
      - radicals: √63, √(x+1) -> sqrt(63), sqrt(x+1)
      - unicode superscripts: x⁶ -> x^6
    """
    s = (s or "").strip()
    if not s:
        return s

    # Normalize symbols
    s = s.replace("−", "-").replace("·", "*").replace("×", "*")
    s = re.sub(r"\s+", " ", s).strip()

    # Unicode fractions like ½
    for k, v in _UFRAC.items():
        s = s.replace(k, v)

    # Mixed numbers like "3 1/2" -> "(3+1/2)"
    # (Only when there's a space between whole and fraction)
    s = re.sub(r"\b(\d+)\s+(\d+)\s*/\s*(\d+)\b", r"(\1+(\2/\3))", s)

    # Unicode superscripts: x⁶ -> x^6
    sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    s = re.sub(
        r"([A-Za-z])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda m: m.group(1) + "^" + m.group(2).translate(sup_map),
        s
    )

    # Classroom shorthand: 1/2x means (1/2)*x (NOT 1/(2x))
    s = re.sub(r'(\d+)\s*/\s*(\d+)\s*([A-Za-z])', r'(\1/\2)*\3', s)

    # RADICAL SUPPORT:
    # √(something) -> sqrt(something)
    s = re.sub(r"√\s*\(([^)]+)\)", r"sqrt(\1)", s)
    # √63, √x -> sqrt(63), sqrt(x)
    s = re.sub(r"√\s*([A-Za-z0-9]+)", r"sqrt(\1)", s)
    # Ensure explicit multiplication: 5sqrt(63) -> 5*sqrt(63), )sqrt( -> )*sqrt(
    s = re.sub(r"(\d)\s*sqrt\(", r"\1*sqrt(", s)
    s = re.sub(r"\)\s*sqrt\(", r")*sqrt(", s)

    return s


def _parse_expr_friendly(s: str, *, evaluate: bool = True) -> sp.Expr:
    """
    Parse expression with Word-normalization.
    Use evaluate=False for DISPLAY (preserve structure),
    and evaluate=True for COMPUTATION (simplify/solve).
    """
    s = _normalize_word_text(s)
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=evaluate)


# ============================================================
# 2) Composition-of-functions prompt support (new)
# ============================================================

def _try_parse_composition_prompt(raw: str) -> Optional[Tuple[str, List[str]]]:
    """
    Recognize prompts like:
      "find f(g(x)), if g(x)=x+3 and f(x)=x^2+3"
      "Find f∘g if g(x)=... and f(x)=..."
      "Compute f(g(2)) if ..."
    Returns (problem_latex, solution_lines) or None if no match.
    """
    txt = (raw or "").strip()
    if not txt:
        return None

    # Normalize some symbols
    txt_norm = txt.replace("∘", "o")
    txt_norm = re.sub(r"\s+", " ", txt_norm)

    # Must mention f and g definitions
    # Extract g(x)=... and f(x)=...
    mg = re.search(r"g\s*\(\s*x\s*\)\s*=\s*([^,;]+)", txt_norm, flags=re.IGNORECASE)
    mf = re.search(r"f\s*\(\s*x\s*\)\s*=\s*([^,;]+)", txt_norm, flags=re.IGNORECASE)
    if not (mg and mf):
        return None

    g_rhs_raw = mg.group(1).strip()
    f_rhs_raw = mf.group(1).strip()

    # Determine target: f(g(x)) or g(f(x)) or numeric like f(g(2))
    target = None
    m_target = re.search(r"(f\s*\(\s*g\s*\(\s*x\s*\)\s*\)|g\s*\(\s*f\s*\(\s*x\s*\)\s*\))", txt_norm, flags=re.IGNORECASE)
    if m_target:
        target = re.sub(r"\s+", "", m_target.group(1)).lower()

    # Numeric target: f(g(2)) etc
    m_num = re.search(r"(f|g)\s*\(\s*(f|g)\s*\(\s*([\-]?\d+)\s*\)\s*\)", txt_norm, flags=re.IGNORECASE)
    num_outer = num_inner = None
    num_value = None
    if m_num:
        num_outer = m_num.group(1).lower()
        num_inner = m_num.group(2).lower()
        num_value = int(m_num.group(3))
        target = f"{num_outer}({num_inner}({num_value}))"

    # If not specified, default to f(g(x))
    if target is None:
        target = "f(g(x))"

    # Parse functions for computation
    x = sp.Symbol("x")
    try:
        g_expr = _parse_expr_friendly(g_rhs_raw, evaluate=True)
        f_expr = _parse_expr_friendly(f_rhs_raw, evaluate=True)
    except Exception:
        return None

    # Build solution
    lines: List[str] = []

    # Display versions (do not evaluate) for problem statement
    try:
        g_disp = _parse_expr_friendly(g_rhs_raw, evaluate=False)
        f_disp = _parse_expr_friendly(f_rhs_raw, evaluate=False)
    except Exception:
        g_disp, f_disp = g_expr, f_expr

    # Compose
    if target.startswith("f(g(") and num_value is None:
        composed = sp.simplify(f_expr.subs(x, g_expr))
        problem_ltx = r"\mathrm{Find}\; f(g(x)) \;\mathrm{if}\; " + sp.latex(sp.Eq(sp.Function("g")(x), g_disp, evaluate=False)) \
                      + r"\;\mathrm{and}\; " + sp.latex(sp.Eq(sp.Function("f")(x), f_disp, evaluate=False))
        lines.append(r"f(g(x))")
        lines.append(r"= " + sp.latex(f_expr.subs(x, sp.Symbol(r"g(x)"))))  # formal step
        lines.append(r"= " + sp.latex(f_expr.subs(x, g_expr)))
        lines.append(r"= " + sp.latex(composed))
        lines.append(r"\therefore\ f(g(x))=" + sp.latex(composed))
        return problem_ltx, lines[:5]

    if target.startswith("g(f(") and num_value is None:
        composed = sp.simplify(g_expr.subs(x, f_expr))
        problem_ltx = r"\mathrm{Find}\; g(f(x)) \;\mathrm{if}\; " + sp.latex(sp.Eq(sp.Function("g")(x), g_disp, evaluate=False)) \
                      + r"\;\mathrm{and}\; " + sp.latex(sp.Eq(sp.Function("f")(x), f_disp, evaluate=False))
        lines.append(r"g(f(x))")
        lines.append(r"= " + sp.latex(g_expr.subs(x, sp.Symbol(r"f(x)"))))
        lines.append(r"= " + sp.latex(g_expr.subs(x, f_expr)))
        lines.append(r"= " + sp.latex(composed))
        lines.append(r"\therefore\ g(f(x))=" + sp.latex(composed))
        return problem_ltx, lines[:5]

    # Numeric: f(g(n)) or g(f(n))
    if num_value is not None and num_outer and num_inner:
        inner_expr = g_expr if num_inner == "g" else f_expr
        outer_expr = f_expr if num_outer == "f" else g_expr

        inner_val = sp.simplify(inner_expr.subs(x, num_value))
        outer_val = sp.simplify(outer_expr.subs(x, inner_val))

        problem_ltx = r"\mathrm{Find}\; " + re.sub(r"\s+", "", target) + r"\;\mathrm{if}\; " \
                      + sp.latex(sp.Eq(sp.Function("g")(x), g_disp, evaluate=False)) \
                      + r"\;\mathrm{and}\; " + sp.latex(sp.Eq(sp.Function("f")(x), f_disp, evaluate=False))

        lines.append(re.sub(r"\s+", "", target))
        lines.append(r"= " + (r"f(" if num_outer == "f" else r"g(") + sp.latex(inner_val) + r")")
        lines.append(r"= " + sp.latex(outer_expr.subs(x, inner_val)))
        lines.append(r"= " + sp.latex(outer_val))
        lines.append(r"\therefore\ " + re.sub(r"\s+", "", target) + "=" + sp.latex(outer_val))
        return problem_ltx, lines[:5]

    return None


# ============================================================
# 3) Step generation (math-only lines, no labels, no boxes)
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
    # Try stronger simplification for radicals/rationals
    simp2 = sp.simplify(sp.together(sp.radsimp(simp)))
    if simp2 != simp:
        return [sp.latex(expr), sp.latex(simp), sp.latex(simp2), r"\therefore\ " + sp.latex(simp2)][:5]
    return [sp.latex(expr), sp.latex(simp), r"\therefore\ " + sp.latex(simp)][:5]


def compute_problem_and_solution_lines(raw: str, var: str = "x") -> Tuple[str, Optional[List[str]]]:
    """
    Returns:
      problem_latex: LaTeX for the ORIGINAL problem (not pre-simplified)
      solution_lines: list of LaTeX lines (math-only), or None if parsing fails
    """
    raw = (raw or "").strip()
    if not raw:
        return r"\mathrm{(blank)}", None

    # 1) Try advanced composition prompt
    comp = _try_parse_composition_prompt(raw)
    if comp:
        return comp

    x = sp.Symbol(var)
    raw_norm = _normalize_word_text(raw)

    # Equation
    if "=" in raw_norm:
        left_s, right_s = raw_norm.split("=", 1)

        # DISPLAY parse (preserve structure)
        try:
            left_disp = _parse_expr_friendly(left_s, evaluate=False)
            right_disp = _parse_expr_friendly(right_s, evaluate=False)
            problem_ltx = sp.latex(sp.Eq(left_disp, right_disp, evaluate=False))
        except Exception:
            problem_ltx = raw_norm

        # COMPUTE parse
        try:
            left = _parse_expr_friendly(left_s, evaluate=True)
            right = _parse_expr_friendly(right_s, evaluate=True)
        except Exception:
            return problem_ltx, None

        expr = sp.simplify(left - right)
        try:
            deg = sp.Poly(expr, x).degree()
        except Exception:
            deg = 1

        if deg == 2:
            return problem_ltx, _quadratic_solution_lines(left, right, x)
        else:
            return problem_ltx, _linear_solution_lines(left, right, x)

    # Expression-only
    try:
        expr_disp = _parse_expr_friendly(raw_norm, evaluate=False)
        problem_ltx = sp.latex(expr_disp)
    except Exception:
        problem_ltx = raw_norm

    try:
        expr = _parse_expr_friendly(raw_norm, evaluate=True)
        return problem_ltx, _simplify_expression_lines(expr)
    except Exception:
        return problem_ltx, None


# ============================================================
# 4) Rendering (auto-fit)
# ============================================================

def _clean_for_mathtext(latex: str) -> str:
    s = (latex or "").strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.replace(r"\displaystyle", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    # Avoid unsupported macros
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)
    return s.strip()

def _render_math_png(math_latex: str, font_size: int = 16, dpi: int = 300) -> str:
    math_latex = _clean_for_mathtext(math_latex)
    math_latex = f"${math_latex}$"

    fig = plt.figure()
    fig.patch.set_alpha(0)
    t = fig.text(0, 0, math_latex, fontsize=font_size)
    fig.canvas.draw()
    bbox = t.get_window_extent()
    w, h = bbox.width / dpi, bbox.height / dpi
    plt.close(fig)

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
                               start_font: int = 16, min_font: int = 11):
    """
    Auto-fit: if the rendered expression is too big (scale too small),
    rerender with a smaller font until it fits nicely.
    """
    target_min_scale = 0.85  # higher = prefer larger readable text
    best_path = None
    best_img = None
    best_scale = None
    best_w = best_h = None

    for fs in range(start_font, min_font - 1, -1):
        path = _render_math_png(math_latex, font_size=fs)
        img = ImageReader(path)
        iw, ih = img.getSize()
        scale = min(w / iw, h / ih, 1.0)

        # keep best so far
        if best_scale is None or scale > best_scale:
            # clean up previous best file
            if best_path and os.path.exists(best_path):
                try:
                    os.remove(best_path)
                except OSError:
                    pass
            best_path, best_img, best_scale = path, img, scale
            best_w, best_h = iw, ih
        else:
            # not best
            try:
                os.remove(path)
            except OSError:
                pass

        if scale >= target_min_scale:
            break

    # draw best
    dw, dh = best_w * best_scale, best_h * best_scale
    c.drawImage(best_img, cx - dw/2, cy - dh/2, dw, dh, mask="auto")

    # cleanup
    if best_path and os.path.exists(best_path):
        try:
            os.remove(best_path)
        except OSError:
            pass


# ============================================================
# 5) PDF layout: HORIZONTAL bands + optional teacher instructions
# ============================================================

def build_foldable_pdf(out_path: str,
                       eq1_ltx: str, eq1_lines: Optional[List[str]], instr1: str,
                       eq2_ltx: str, eq2_lines: Optional[List[str]], instr2: str,
                       eq3_ltx: str, eq3_lines: Optional[List[str]], instr3: str):
    """
    Horizontal fold lines at 2.75", 5.5", 8.25" from the TOP.

    FRONT (top to bottom bands):
      Band 1: #2 problem only + fold-center note + optional instruction
      Band 2: #3 worked + optional instruction
      Band 3: Open First
      Band 4: #1 problem only + optional instruction

    BACK (rotated 180 degrees):
      Band 1: #1 worked + optional instruction
      Band 2: empty
      Band 3: #3 problem only + optional instruction
      Band 4: #2 worked + optional instruction
    """
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
        band_top, band_bottom = band_rect(band_index)
        left = 0.6 * inch
        right = page_w - 0.6 * inch
        top = band_top - 0.90 * inch   # leave room for fold text + instruction
        bottom = band_bottom + 0.45 * inch
        return left, right, bottom, top

    def draw_instruction_in_band(band_index: int, text: str):
        # Put teacher instruction BELOW fold text area
        if not (text or "").strip():
            return
        band_top, _ = band_rect(band_index)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(page_w / 2, band_top - 0.80 * inch, text.strip())

    def draw_math_in_band_center(band_index: int, latex: str):
        left, right, bottom, top = band_content_box(band_index)
        _draw_math_centered_autofit(c, latex, (left + right) / 2, (bottom + top) / 2,
                                    (right - left), (top - bottom), start_font=16, min_font=11)

    def draw_worked_lines_in_band(band_index: int, math_lines: List[str]):
        left, right, bottom, top = band_content_box(band_index)
        cx = (left + right) / 2
        usable_h = top - bottom
        lines = (math_lines or [])[:5]
        n = max(1, len(lines))

        line_gap = usable_h / (n + 1)
        start_y = top - line_gap

        for i, ltx_line in enumerate(lines):
            cy = start_y - i * line_gap
            _draw_math_centered_autofit(
                c,
                ltx_line,
                cx,
                cy,
                (right - left) * 0.97,
                line_gap * 0.82,
                start_font=16,
                min_font=11
            )

    # ---------------- FRONT ----------------
    fold_lines_horizontal()

    # Band 1 (top): #2 problem only + fold note + instruction below it
    draw_boxed_label(ys[0], "#2")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(page_w / 2, ys[0] - 0.45 * inch, "Fold center, these two folds facing each other")
    draw_instruction_in_band(0, instr2)
    draw_math_in_band_center(0, eq2_ltx)

    # Band 2: #3 worked
    draw_boxed_label(ys[1], "#3")
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, ys[1] - 0.20 * inch, "Fold 3rd. This side out")
    draw_instruction_in_band(1, instr3)
    if eq3_lines:
        draw_worked_lines_in_band(1, eq3_lines)
    else:
        draw_math_in_band_center(1, eq3_ltx)

    # Band 3: Open First
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, ys[2] - 0.20 * inch, "Fold 1st. This side out.")
    band3_top, band3_bottom = band_rect(2)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(page_w / 2, (band3_top + band3_bottom) / 2, "Open First")

    # Band 4: #1 problem only
    draw_boxed_label(ys[3], "#1")
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, ys[3] - 0.20 * inch, "Fold 2nd. Fold under")
    draw_instruction_in_band(3, instr1)
    draw_math_in_band_center(3, eq1_ltx)

    c.showPage()

    # ---------------- BACK (rotated 180°) ----------------
    c.saveState()
    c.translate(page_w, page_h)
    c.rotate(180)

    fold_lines_horizontal()

    # Back Band 1: #1 worked
    draw_boxed_label(ys[0], "#1")
    draw_instruction_in_band(0, instr1)
    if eq1_lines:
        draw_worked_lines_in_band(0, eq1_lines)
    else:
        draw_math_in_band_center(0, eq1_ltx)

    # Back Band 2: empty

    # Back Band 3: #3 problem only
    draw_boxed_label(ys[2], "#3")
    draw_instruction_in_band(2, instr3)
    draw_math_in_band_center(2, eq3_ltx)

    # Back Band 4: #2 worked
    draw_boxed_label(ys[3], "#2")
    draw_instruction_in_band(3, instr2)
    if eq2_lines:
        draw_worked_lines_in_band(3, eq2_lines)
    else:
        draw_math_in_band_center(3, eq2_ltx)

    c.restoreState()
    c.showPage()
    c.save()


# ============================================================
# 6) Streamlit UI
# ============================================================

st.set_page_config(page_title="Foldable Generator (Word Copy/Paste)", layout="centered")
st.title("Printable Math Foldable Generator (Word Copy/Paste)")

st.write(
    "Paste from Word (Equation tool) or type normally.\n\n"
    "Now supports prompts like:\n"
    "• find f(g(x)), if g(x)=x+3 and f(x)=x^2+3\n\n"
    "Tips:\n"
    "- `1/2x` is interpreted as `(1/2)x`.\n"
    "- `√63` is supported.\n"
)

with st.expander("Examples to test"):
    st.markdown(
        """
**Radicals**
- `5√63`  → should simplify to `15√7` in the worked lines  
- `√48`   → should simplify to `4√3`

**Composition**
- `find f(g(x)), if g(x)=x+3 and f(x)=x^2+3`
- `find f(g(2)), if g(x)=2x-1 and f(x)=x^2`

**Equations**
- `1/2x + 3 = 11`
- `x^2 - 5x + 6 = 0`
"""
    )

i1 = st.text_input("Instruction for Problem #1 (optional)", value="")
p1 = st.text_area("Problem #1", value="1/2x + 3 = 11", height=70)

i2 = st.text_input("Instruction for Problem #2 (optional)", value="")
p2 = st.text_area("Problem #2", value="find f(g(x)), if g(x)=x+3 and f(x)=x^2+3", height=70)

i3 = st.text_input("Instruction for Problem #3 (optional)", value="")
p3 = st.text_area("Problem #3", value="5√63", height=70)

if st.button("Generate Foldable PDF", type="primary"):
    try:
        eq1_ltx, eq1_lines = compute_problem_and_solution_lines(p1, var="x")
        eq2_ltx, eq2_lines = compute_problem_and_solution_lines(p2, var="x")
        eq3_ltx, eq3_lines = compute_problem_and_solution_lines(p3, var="x")

        out_path = "foldable_output.pdf"
        build_foldable_pdf(
            out_path,
            eq1_ltx, eq1_lines, i1,
            eq2_ltx, eq2_lines, i2,
            eq3_ltx, eq3_lines, i3
        )

        with open(out_path, "rb") as f:
            st.download_button(
                "Download foldable_output.pdf",
                data=f,
                file_name="foldable_output.pdf",
                mime="application/pdf",
            )

        if (eq1_lines is None) or (eq2_lines is None) or (eq3_lines is None):
            st.info(
                "PDF generated. If any problem couldn't be solved automatically, "
                "it will still print as a clean equation."
            )

    except Exception as e:
        st.error("Something went wrong while generating the foldable.")
        st.exception(e)

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
# 1) Parsing utilities (Word-friendly + robust radicals/nth-roots)
# ============================================================

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,  # allow ^
)

# Unicode fractions (Word sometimes pastes these)
_UFRAC = {
    "½": "(1/2)", "⅓": "(1/3)", "⅔": "(2/3)", "¼": "(1/4)", "¾": "(3/4)",
    "⅕": "(1/5)", "⅖": "(2/5)", "⅗": "(3/5)", "⅘": "(4/5)",
    "⅙": "(1/6)", "⅚": "(5/6)", "⅛": "(1/8)", "⅜": "(3/8)", "⅝": "(5/8)", "⅞": "(7/8)",
}

def _normalize_word_text(s: str) -> str:
    """
    Normalize what teachers paste from Word.
    Supports:
      - unicode minus/multiplication
      - unicode fractions and mixed numbers
      - classroom shorthand: 1/2x -> (1/2)*x
      - square roots: √63, √(x+1)
      - cube/fourth roots: ∛, ∜
      - WORD nth-root linear format: sqrt(5&64x^11)
      - general nth-root forms: 5√(expr), 5√expr, √(5&expr) (rare)
      - superscripts: x⁶ -> x^6
      - avoids SymPy logic-word issues later by keeping this purely math normalization
    """
    s = (s or "").strip()
    if not s:
        return s

    # Normalize symbols/spaces
    s = s.replace("−", "-").replace("·", "*").replace("×", "*")
    s = re.sub(r"\s+", " ", s).strip()

    # Unicode fractions
    for k, v in _UFRAC.items():
        s = s.replace(k, v)

    # Mixed numbers: "3 1/2" -> "(3+(1/2))"
    s = re.sub(r"\b(\d+)\s+(\d+)\s*/\s*(\d+)\b", r"(\1+(\2/\3))", s)

    # Unicode superscripts digits -> ^digits (x⁶ -> x^6)
    sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    s = re.sub(
        r"([A-Za-z])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda m: m.group(1) + "^" + m.group(2).translate(sup_map),
        s
    )

    # Classroom shorthand: 1/2x means (1/2)*x
    s = re.sub(r'(\d+)\s*/\s*(\d+)\s*([A-Za-z])', r'(\1/\2)*\3', s)

    # ------------------------------------------------------------
    # BULLETPROOF NTH-ROOT SUPPORT (Word + common teacher typing)
    # ------------------------------------------------------------

    # Word nth-root linear form often appears as: sqrt(5&64x^11)
    # Sometimes with backslash: \sqrt(5&64x^11)
    s = re.sub(r"\\?sqrt\s*\(\s*(\d+)\s*&\s*([^)]+)\)", r"root(\2,\1)", s)

    # Rare: √(5&64x^11) or √( 5 & 64x^11 )
    s = re.sub(r"√\s*\(\s*(\d+)\s*&\s*([^)]+)\)", r"root(\2,\1)", s)

    # Teacher typed: 5√(64x^11)  -> root(64x^11,5)
    s = re.sub(r"(\d+)\s*√\s*\(([^)]+)\)", r"root(\2,\1)", s)

    # Teacher typed: 5√64x^11  -> root(64x^11,5) (stop at an operator/comma/equals)
    s = re.sub(
        r"(\d+)\s*√\s*([A-Za-z0-9][A-Za-z0-9\^\*\(\)\+\-\s\/]*)",
        lambda m: "root(" + m.group(2).strip() + "," + m.group(1) + ")",
        s
    )

    # Cube/fourth root unicode: ∛x, ∜x and with parentheses
    s = re.sub(r"∛\s*\(([^)]+)\)", r"root(\1,3)", s)
    s = re.sub(r"∛\s*([A-Za-z0-9]+)", r"root(\1,3)", s)
    s = re.sub(r"∜\s*\(([^)]+)\)", r"root(\1,4)", s)
    s = re.sub(r"∜\s*([A-Za-z0-9]+)", r"root(\1,4)", s)

    # Square root unicode: √(x+1) or √63
    s = re.sub(r"√\s*\(([^)]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"√\s*([A-Za-z0-9]+)", r"sqrt(\1)", s)

    # Ensure explicit multiplication when a number is directly before sqrt/root
    s = re.sub(r"(\d)\s*(sqrt|root)\(", r"\1*\2(", s)
    s = re.sub(r"\)\s*(sqrt|root)\(", r")*\1(", s)

    # Clean double spaces again
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_expr(s: str, *, evaluate: bool = True) -> sp.Expr:
    s = _normalize_word_text(s)
    local_dict = {"sqrt": sp.sqrt, "root": sp.root, "I": sp.I}
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=evaluate, local_dict=local_dict)


# ============================================================
# 2) Composition prompt support (teacher-style)
# ============================================================

def _composition_prompt(raw: str) -> Optional[Tuple[str, List[str]]]:
    """
    Accepts prompts like:
      "Composition of functions: find f(g(x)), if g(x)=x+3 and f(x)=x^2+3"
      "Find f(g(2)) if g(x)=2x-1 and f(x)=x^2+3"
    Returns (problem_latex, solution_lines) or None if not recognized.
    """
    txt = (raw or "").strip()
    if not txt:
        return None

    t = txt.replace("∘", "o")
    t = re.sub(r"\s+", " ", t)

    # Must include f(x)= and g(x)=
    if not re.search(r"g\s*\(\s*x\s*\)\s*=", t, flags=re.IGNORECASE):
        return None
    if not re.search(r"f\s*\(\s*x\s*\)\s*=", t, flags=re.IGNORECASE):
        return None

    mg = re.search(
        r"g\s*\(\s*x\s*\)\s*=\s*(.+?)(?=(?:,|\band\b)\s*f\s*\(\s*x\s*\)\s*=|$)",
        t, flags=re.IGNORECASE
    )
    mf = re.search(
        r"f\s*\(\s*x\s*\)\s*=\s*(.+?)(?=(?:,|\band\b)\s*g\s*\(\s*x\s*\)\s*=|$)",
        t, flags=re.IGNORECASE
    )
    if not (mg and mf):
        return None

    g_raw = mg.group(1).strip().rstrip(".")
    f_raw = mf.group(1).strip().rstrip(".")

    x = sp.Symbol("x")
    try:
        g = _parse_expr(g_raw, evaluate=True)
        f = _parse_expr(f_raw, evaluate=True)
        g_disp = _parse_expr(g_raw, evaluate=False)
        f_disp = _parse_expr(f_raw, evaluate=False)
    except Exception:
        return None

    # Target
    target = "f(g(x))"
    if re.search(r"g\s*\(\s*f\s*\(\s*x\s*\)\s*\)", t, flags=re.IGNORECASE):
        target = "g(f(x))"

    # Numeric target f(g(2))
    mnum = re.search(
        r"(f|g)\s*\(\s*(f|g)\s*\(\s*([\-]?\d+)\s*\)\s*\)",
        t, flags=re.IGNORECASE
    )

    prob = (
        r"\mathrm{Compose:}\ "
        + sp.latex(sp.Eq(sp.Function("g")(x), g_disp, evaluate=False))
        + r"\ ,\ "
        + sp.latex(sp.Eq(sp.Function("f")(x), f_disp, evaluate=False))
    )

    lines: List[str] = [prob]

    if mnum:
        outer = mnum.group(1).lower()
        inner = mnum.group(2).lower()
        n = int(mnum.group(3))

        inner_expr = g if inner == "g" else f
        outer_expr = f if outer == "f" else g

        inner_val = sp.simplify(inner_expr.subs(x, n))
        outer_val = sp.simplify(outer_expr.subs(x, inner_val))

        lines.append(f"{outer}({inner}({n}))")
        lines.append(r"= " + ("f(" if outer == "f" else "g(") + sp.latex(inner_val) + r")")
        lines.append(r"= " + sp.latex(outer_val))
        lines.append(r"\therefore\ " + f"{outer}({inner}({n}))=" + sp.latex(outer_val))
        return prob, lines[:5]

    if target == "g(f(x))":
        comp = sp.simplify(g.subs(x, f))
        lines.append(r"g(f(x))")
        lines.append(r"= " + sp.latex(g.subs(x, f)))
        lines.append(r"= " + sp.latex(comp))
        lines.append(r"\therefore\ g(f(x))=" + sp.latex(comp))
        return prob, lines[:5]

    comp = sp.simplify(f.subs(x, g))
    lines.append(r"f(g(x))")
    lines.append(r"= " + sp.latex(f.subs(x, g)))
    lines.append(r"= " + sp.latex(comp))
    lines.append(r"\therefore\ f(g(x))=" + sp.latex(comp))
    return prob, lines[:5]


# ============================================================
# 3) Systems support (mathtext-safe display)
# ============================================================

def _split_equations(text: str) -> List[str]:
    """
    Extract equations from teacher prompts like:
      "Solve system: x+y=5 and 2x-y=1"
      "System: x+y+z=6; 2x-y+z=3; x-2y+3z=10"
    """
    if not text:
        return []
    t = text.strip()

    # Remove common leading phrases
    t = re.sub(r"^\s*solve\s+(the\s+)?system\s*[:\-]?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*system\s*[:\-]?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*solve\s*[:\-]?\s*", "", t, flags=re.IGNORECASE)

    # Split on ; or newline or "and" (when it separates equations)
    parts = re.split(r"[;\n]+|\band\b(?=.*=)", t, flags=re.IGNORECASE)

    eqs: List[str] = []
    for p in parts:
        p = p.strip().rstrip(".")
        if not p:
            continue
        if ":" in p and "=" in p:
            p = p.split(":")[-1].strip()
        if "=" in p:
            eqs.append(p)
    return eqs


def _detect_vars(eqs: List[sp.Eq]) -> List[sp.Symbol]:
    vars_set = set()
    for e in eqs:
        vars_set |= e.free_symbols
    order = ["x", "y", "z", "a", "b", "c"]
    syms = list(vars_set)
    syms.sort(key=lambda s: (order.index(str(s)) if str(s) in order else 999, str(s)))
    return syms[:3]


def _solve_system_lines(equations_raw: List[str]) -> Optional[Tuple[str, List[str]]]:
    # Parse equations
    eqs: List[sp.Eq] = []
    for er in equations_raw:
        er = er.strip().rstrip(".")
        left_s, right_s = er.split("=", 1)
        left = _parse_expr(left_s, evaluate=True)
        right = _parse_expr(right_s, evaluate=True)
        eqs.append(sp.Eq(left, right))

    syms = _detect_vars(eqs)
    if not syms:
        return None

    # Mathtext-safe ONE LINE display for the "problem" panel
    prob = r",\; ".join(sp.latex(e) for e in eqs)

    sol = sp.solve(eqs, syms, dict=True)
    worked: List[str] = [prob]

    if not sol:
        worked.append(r"\therefore\ \mathrm{no\ solution}")
        return prob, worked[:5]

    s0 = sol[0]
    assigns = [sp.latex(v) + "=" + sp.latex(sp.simplify(s0[v])) for v in syms]
    worked.append(r"\therefore\ " + r",\ ".join(assigns))
    return prob, worked[:5]


# ============================================================
# 4) Solve/simplify router (handles teacher words safely)
# ============================================================

def _simplify_lines(expr_raw: str) -> Tuple[str, List[str]]:
    """
    Simplify an expression from a teacher prompt.
    Removes logic words like 'or/and/not' to prevent SymPy boolean parsing.
    """
    s = (expr_raw or "").strip()

    if ":" in s:
        s = s.split(":")[-1].strip()

    s = re.sub(r"\b(simplify|simplified|reduce|compute|evaluate|find|expression)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(or|and|not)\b", "", s, flags=re.IGNORECASE)

    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()

    try:
        disp = _parse_expr(s, evaluate=False)
        prob = sp.latex(disp)
    except Exception:
        prob = _normalize_word_text(s) if s else r"\mathrm{(blank)}"
        return prob, [prob]

    try:
        expr = _parse_expr(s, evaluate=True)
    except Exception:
        return prob, [prob]

    # Stronger simplification pipeline for radicals/rationals
    s1 = sp.together(expr)
    s2 = sp.radsimp(s1)
    s3 = sp.simplify(s2)

    lines = [prob]
    if s1 != expr:
        lines.append(sp.latex(s1))
    if s2 != s1:
        lines.append(sp.latex(s2))
    if s3 != s2:
        lines.append(sp.latex(s3))
    lines.append(r"\therefore\ " + sp.latex(s3))
    return prob, lines[:5]


def _solve_equation_lines(eq_raw: str) -> Tuple[str, List[str]]:
    left_s, right_s = eq_raw.split("=", 1)

    # display preserved
    left_disp = _parse_expr(left_s, evaluate=False)
    right_disp = _parse_expr(right_s, evaluate=False)
    prob = sp.latex(sp.Eq(left_disp, right_disp, evaluate=False))

    left = _parse_expr(left_s, evaluate=True)
    right = _parse_expr(right_s, evaluate=True)
    x = sp.Symbol("x")
    eq = sp.Eq(left, right)

    sols = sp.solve(eq, x)

    lines = [prob, sp.latex(sp.Eq(sp.simplify(left - right), 0))]
    if len(sols) == 0:
        lines.append(r"\therefore\ \mathrm{no\ solution}")
        return prob, lines[:5]
    if len(sols) == 1:
        lines.append(r"\therefore\ x=" + sp.latex(sp.simplify(sols[0])))
        return prob, lines[:5]
    lines.append(r"\therefore\ x=" + sp.latex(sp.simplify(sols[0])) + r",\ x=" + sp.latex(sp.simplify(sols[1])))
    return prob, lines[:5]


def solve_or_simplify_advanced_prompt(raw: str) -> Tuple[str, Optional[List[str]]]:
    t = (raw or "").strip()
    if not t:
        return r"\mathrm{(blank)}", None

    # 1) Composition prompts
    comp = _composition_prompt(t)
    if comp:
        return comp

    # 2) Systems (2+ equations)
    eqs_raw = _split_equations(t)
    if len(eqs_raw) >= 2:
        sys = _solve_system_lines(eqs_raw[:3])
        if sys:
            return sys

    # 3) If it contains '=', solve (even if they don’t type the word solve)
    if "=" in t:
        # try to extract the first equation-like chunk
        m = re.search(r"([A-Za-z0-9\(\)\+\-\*/\^\s√∛∜\.,&]+=[A-Za-z0-9\(\)\+\-\*/\^\s√∛∜\.,&]+)", t)
        eq_raw = (m.group(1) if m else t).strip().rstrip(".")
        try:
            return _solve_equation_lines(eq_raw)
        except Exception:
            # fall back to printing as-is
            safe = _normalize_word_text(eq_raw)
            return safe, None

    # 4) Otherwise simplify
    return _simplify_lines(t)


# ============================================================
# 5) Render LaTeX with matplotlib mathtext + AUTO-FIT
#    IMPORTANT: mathtext does NOT support \\ arrays/environments.
# ============================================================

def _clean_for_mathtext(latex: str) -> str:
    s = (latex or "").strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.replace(r"\displaystyle", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
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
                               start_font: int, min_font: int, target_min_scale: float):
    """
    Auto-fit: render decreasing font until the scale is acceptable.
    target_min_scale lower => keeps text larger (less shrinking).
    """
    best = None  # (path, img, scale, iw, ih)
    for fs in range(start_font, min_font - 1, -1):
        path = _render_math_png(math_latex, font_size=fs)
        img = ImageReader(path)
        iw, ih = img.getSize()
        scale = min(w / iw, h / ih, 1.0)

        if best is None or scale > best[2]:
            if best and os.path.exists(best[0]):
                try:
                    os.remove(best[0])
                except OSError:
                    pass
            best = (path, img, scale, iw, ih)
        else:
            try:
                os.remove(path)
            except OSError:
                pass

        if scale >= target_min_scale:
            break

    path, img, scale, iw, ih = best
    dw, dh = iw * scale, ih * scale
    c.drawImage(img, cx - dw/2, cy - dh/2, dw, dh, mask="auto")

    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# ============================================================
# 6) PDF layout: HORIZONTAL bands + optional teacher instructions
# ============================================================

def build_foldable_pdf(out_path: str,
                       p1_ltx: str, p1_lines: Optional[List[str]], instr1: str,
                       p2_ltx: str, p2_lines: Optional[List[str]], instr2: str,
                       p3_ltx: str, p3_lines: Optional[List[str]], instr3: str):
    page_w, page_h = letter
    c = Canvas(out_path, pagesize=letter)

    # Band boundaries from TOP: 0, 2.75, 5.5, 8.25, 11 inches
    y_top = page_h
    y_marks_from_top = [0.0, 2.75 * inch, 5.5 * inch, 8.25 * inch, 11.0 * inch]
    ys = [y_top - m for m in y_marks_from_top]

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
        top = band_top - 0.90 * inch   # space for fold text + instruction
        bottom = band_bottom + 0.45 * inch
        return left, right, bottom, top

    def draw_instruction_in_band(band_index: int, text: str):
        if not (text or "").strip():
            return
        band_top, _ = band_rect(band_index)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(page_w / 2, band_top - 0.80 * inch, text.strip())

    def draw_math_in_band_center(band_index: int, latex: str):
        left, right, bottom, top = band_content_box(band_index)
        _draw_math_centered_autofit(
            c, latex,
            (left + right) / 2, (bottom + top) / 2,
            (right - left), (top - bottom),
            start_font=18, min_font=12,
            target_min_scale=0.78
        )

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
                c, ltx_line,
                cx, cy,
                (right - left) * 0.98,
                line_gap * 0.90,
                start_font=22, min_font=14,   # BIGGER solutions
                target_min_scale=0.75         # shrink less
            )

    # ---------------- FRONT ----------------
    fold_lines_horizontal()

    # Band 1: #2 problem only + fold note
    draw_boxed_label(ys[0], "#2")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(page_w / 2, ys[0] - 0.45 * inch, "Fold center, these two folds facing each other")
    draw_instruction_in_band(0, instr2)
    draw_math_in_band_center(0, p2_ltx)

    # Band 2: #3 worked
    draw_boxed_label(ys[1], "#3")
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, ys[1] - 0.20 * inch, "Fold 3rd. This side out")
    draw_instruction_in_band(1, instr3)
    if p3_lines:
        draw_worked_lines_in_band(1, p3_lines)
    else:
        draw_math_in_band_center(1, p3_ltx)

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
    draw_math_in_band_center(3, p1_ltx)

    c.showPage()

    # ---------------- BACK (rotated 180°) ----------------
    c.saveState()
    c.translate(page_w, page_h)
    c.rotate(180)

    fold_lines_horizontal()

    # Back Band 1: #1 worked
    draw_boxed_label(ys[0], "#1")
    draw_instruction_in_band(0, instr1)
    if p1_lines:
        draw_worked_lines_in_band(0, p1_lines)
    else:
        draw_math_in_band_center(0, p1_ltx)

    # Back Band 2: empty

    # Back Band 3: #3 problem only
    draw_boxed_label(ys[2], "#3")
    draw_instruction_in_band(2, instr3)
    draw_math_in_band_center(2, p3_ltx)

    # Back Band 4: #2 worked
    draw_boxed_label(ys[3], "#2")
    draw_instruction_in_band(3, instr2)
    if p2_lines:
        draw_worked_lines_in_band(3, p2_lines)
    else:
        draw_math_in_band_center(3, p2_ltx)

    c.restoreState()
    c.showPage()
    c.save()


# ============================================================
# 7) Streamlit UI
# ============================================================

st.set_page_config(page_title="Foldable Generator (Word Copy/Paste)", layout="centered")
st.title("Printable Math Foldable Generator (Word Copy/Paste)")

st.write(
    "Paste from Word (Equation tool) or type normally.\n\n"
    "✅ Supports square roots, cube roots, 4th roots, and Word nth-roots like `sqrt(5&64x^11)`.\n"
    "✅ Solutions print larger for student readability.\n"
    "✅ Systems + composition prompts supported.\n"
)

with st.expander("Quick tests"):
    st.markdown(
        """
**Nth root (Word linear)**
- `sqrt(5&64x^11)`  → should show as \\(\\sqrt[5]{64x^{11}}\\)

**Nth root (typed)**
- `5√(64x^11)`  → \\(\\sqrt[5]{64x^{11}}\\)

**Radical simplify**
- `5√63` → \\(15\\sqrt{7}\\)

**System**
- `Solve system: x+y=5 and 2x-y=1`

**Composition**
- `Composition of functions: find f(g(x)) if g(x)=x+3 and f(x)=x^2+3`
"""
    )

i1 = st.text_input("Instruction for Problem #1 (optional)", value="")
p1 = st.text_area("Problem #1", value="sqrt(5&64x^11)", height=70)

i2 = st.text_input("Instruction for Problem #2 (optional)", value="")
p2 = st.text_area("Problem #2", value="Solve system: x+y=5 and 2x-y=1", height=70)

i3 = st.text_input("Instruction for Problem #3 (optional)", value="")
p3 = st.text_area("Problem #3", value="5√63", height=70)

if st.button("Generate Foldable PDF", type="primary"):
    try:
        l1, lines1 = solve_or_simplify_advanced_prompt(p1)
        l2, lines2 = solve_or_simplify_advanced_prompt(p2)
        l3, lines3 = solve_or_simplify_advanced_prompt(p3)

        out_path = "foldable_output.pdf"
        build_foldable_pdf(
            out_path,
            l1, lines1, i1,
            l2, lines2, i2,
            l3, lines3, i3
        )

        with open(out_path, "rb") as f:
            st.download_button(
                "Download foldable_output.pdf",
                data=f,
                file_name="foldable_output.pdf",
                mime="application/pdf",
            )

        if (lines1 is None) or (lines2 is None) or (lines3 is None):
            st.info(
                "PDF generated. If any prompt couldn't be solved automatically, "
                "it will still print as a clean expression."
            )

    except Exception as e:
        st.error("Something went wrong while generating the foldable.")
        st.exception(e)

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
# 1) Word-plain-text friendly parsing
# ============================================================

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # 2x -> 2*x, (x+1)(x-2) -> *
    convert_xor,                          # x^2 supported
)

def _normalize_word_text(s: str) -> str:
    """
    Normalize what teachers paste from Word into a textbox.
    - Keeps ^ (Word often pastes x^6 correctly)
    - Supports unicode minus and multiplication symbols
    - Leaves / for fractions
    """
    s = (s or "").strip()
    s = s.replace("−", "-").replace("·", "*").replace("×", "*")
    s = re.sub(r"\s+", " ", s).strip()

    # Optional: convert unicode superscripts if they appear (rare but harmless)
    sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    s = re.sub(
        r"([A-Za-z])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda m: m.group(1) + "^" + m.group(2).translate(sup_map),
        s
    )

    # Interpret classroom shorthand: 1/2x means (1/2)*x
    s = re.sub(r'(\d+)\s*/\s*(\d+)\s*([A-Za-z])', r'(\1/\2)*\3', s)

    return s

def _parse_expr_friendly(s: str, *, evaluate: bool = True) -> sp.Expr:
    s = _normalize_word_text(s)
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=evaluate)



# ============================================================
# 2) Build step-by-step solutions (math-only lines)
#    (No labels, no \boxed — final line uses \therefore)
# ============================================================

def _linear_solution_lines(left: sp.Expr, right: sp.Expr, x: sp.Symbol) -> List[str]:
    """
    Returns list of LaTeX math strings, each a single expression/equation line.
    """
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

def compute_problem_and_solution_lines(raw: str, var: str = "x") -> Tuple[str, Optional[List[str]]]:
    """
    Returns:
      problem_latex: LaTeX for the ORIGINAL problem (not pre-simplified)
      solution_lines: list of LaTeX lines for the worked solution (math-only), or None if parsing fails
    """
    x = sp.Symbol(var)
    raw_norm = _normalize_word_text(raw)

    if not raw_norm:
        return r"\mathrm{(blank)}", None

    # ---------------- Equation ----------------
    if "=" in raw_norm:
        left_s, right_s = raw_norm.split("=", 1)

        # Parse for DISPLAY (do not simplify)
        try:
            left_disp = _parse_expr_friendly(left_s, evaluate=False)
            right_disp = _parse_expr_friendly(right_s, evaluate=False)
            problem_ltx = sp.latex(sp.Eq(left_disp, right_disp, evaluate=False))
        except Exception:
            # If display parsing fails, still show the raw text as math
            problem_ltx = raw_norm

        # Parse for SOLVING (can simplify)
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

    # ---------------- Expression-only ----------------
    # Display version (no simplify)
    try:
        expr_disp = _parse_expr_friendly(raw_norm, evaluate=False)
        problem_ltx = sp.latex(expr_disp)
    except Exception:
        problem_ltx = raw_norm

    # Solving/simplifying version
    try:
        expr = _parse_expr_friendly(raw_norm, evaluate=True)
        simp = sp.simplify(expr)
        return problem_ltx, [sp.latex(expr), sp.latex(simp), r"\therefore\ " + sp.latex(simp)]
    except Exception:
        return problem_ltx, None

# ============================================================
# 3) Render single-line LaTeX to PNG (mathtext-safe)
# ============================================================

def _clean_for_mathtext(latex: str) -> str:
    s = (latex or "").strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.replace(r"\displaystyle", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    # IMPORTANT: no \boxed anywhere
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

def _draw_math_centered(c: Canvas, math_latex: str, cx: float, cy: float, w: float, h: float, font_size: int = 16):
    path = _render_math_png(math_latex, font_size=font_size)
    try:
        img = ImageReader(path)
        iw, ih = img.getSize()
        scale = min(w / iw, h / ih, 1.0)
        dw, dh = iw * scale, ih * scale
        c.drawImage(img, cx - dw/2, cy - dh/2, dw, dh, mask="auto")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ============================================================
# 4) PDF layout: HORIZONTAL bands like your sample
# ============================================================

def build_foldable_pdf(out_path: str,
                       eq1_ltx: str, eq1_lines: Optional[List[str]], instr1: str,
                       eq2_ltx: str, eq2_lines: Optional[List[str]], instr2: str,
                       eq3_ltx: str, eq3_lines: Optional[List[str]], instr3: str):
    """
    Horizontal fold lines at 2.75", 5.5", 8.25" from the TOP.

    FRONT (top to bottom bands):
      Band 1: #2 problem only + one-line note
      Band 2: #3 worked (math lines only)
      Band 3: Open First
      Band 4: #1 problem only

    BACK (rotated 180 degrees):
      Band 1: #1 worked
      Band 2: empty
      Band 3: #3 problem only
      Band 4: #2 worked
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
        # 0=top band, 3=bottom band
        return ys[band_index], ys[band_index + 1]

    def band_content_box(band_index: int) -> Tuple[float, float, float, float]:
        band_top, band_bottom = band_rect(band_index)
        left = 0.6 * inch
        right = page_w - 0.6 * inch
        top = band_top - 0.65 * inch
        bottom = band_bottom + 0.45 * inch
        return left, right, bottom, top

    def draw_instruction_in_band(band_index: int, text: str):
        if not (text or "").strip():
            return
        band_top, _ = band_rect(band_index)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(page_w / 2, band_top - 0.80 * inch, text.strip())


    def draw_math_in_band_center(band_index: int, latex: str, font_size: int = 16):
        left, right, bottom, top = band_content_box(band_index)
        _draw_math_centered(c, latex, (left + right) / 2, (bottom + top) / 2, (right - left), (top - bottom), font_size)

    def draw_worked_lines_in_band(band_index: int, math_lines: List[str]):
        left, right, bottom, top = band_content_box(band_index)
        cx = (left + right) / 2
        usable_h = top - bottom
        n = max(1, min(5, len(math_lines)))
        lines = math_lines[:n]

        # Even vertical spacing, centered
        line_gap = usable_h / (n + 1)
        start_y = top - line_gap

        for i, ltx_line in enumerate(lines):
            cy = start_y - i * line_gap
            _draw_math_centered(
                c,
                ltx_line,
                cx,
                cy,
                (right - left) * 0.95,
                line_gap * 0.82,
                16
            )

    # ---------------- FRONT ----------------
    fold_lines_horizontal()

    # Band 1: #2 problem only + note (one line)
    draw_boxed_label(ys[0], "#2")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(page_w / 2, ys[0] - 0.45 * inch, "Fold center, these two folds facing each other")
    draw_instruction_in_band(0, instr2)
    draw_math_in_band_center(0, eq2_ltx, 16)

    # Band 2: #3 worked
    draw_boxed_label(ys[1], "#3")
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, ys[1] - 0.20 * inch, "Fold 3rd. This side out")
    draw_instruction_in_band(1, instr3)
    if eq3_lines:
        draw_worked_lines_in_band(1, eq3_lines)
    else:
        draw_math_in_band_center(1, eq3_ltx, 16)

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
    draw_math_in_band_center(3, eq1_ltx, 16)

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
        draw_math_in_band_center(0, eq1_ltx, 16)

    # Back Band 2: empty

    # Back Band 3: #3 problem only
    draw_boxed_label(ys[2], "#3")
    draw_instruction_in_band(2, instr3)
    draw_math_in_band_center(2, eq3_ltx, 16)

    # Back Band 4: #2 worked
    draw_boxed_label(ys[3], "#2")
    draw_instruction_in_band(3, instr2)
    if eq2_lines:
        draw_worked_lines_in_band(3, eq2_lines)
    else:
        draw_math_in_band_center(3, eq2_ltx, 16)

    c.restoreState()
    c.showPage()
    c.save()


# ============================================================
# 5) Streamlit UI
# ============================================================

st.set_page_config(page_title="Foldable Generator (Word Copy/Paste)", layout="centered")
st.title("Printable Math Foldable Generator (Word Copy/Paste)")

st.write(
    "Type equations in **Word** using the Equation tool, then copy/paste here.\n\n"
    "Tips:\n"
    "- `1/2x` is interpreted as `(1/2)x` automatically.\n"
    "- Exponents from Word like `x^6` work great.\n"
)

with st.expander("Optional Instructions"):
    st.markdown(
        "You can add an optional instruction for each problem (example: **Simplify**, **Solve**, **Factor**). "
        "Whatever you type will print exactly above that panel."
    )

i1 = st.text_input("Instruction for Problem #1 (optional)", value="")
p1 = st.text_area("Problem #1", value="1/2x + 3 = 11", height=70)

i2 = st.text_input("Instruction for Problem #2 (optional)", value="")
p2 = st.text_area("Problem #2", value="3x - 5 = 16", height=70)

i3 = st.text_input("Instruction for Problem #3 (optional)", value="")
p3 = st.text_area("Problem #3", value="x^2 - 5x + 6 = 0", height=70)

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



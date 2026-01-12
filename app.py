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
# 1) Word-plain-text friendly preprocessing
# ============================================================

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # 2x -> 2*x, (x+1)(x-2) -> *
    convert_xor,                          # allows x^2
)

def _normalize_word_text(s: str) -> str:
    """
    Takes what Word pastes into a textbox (plain text) and normalizes it so:
    - x6 becomes x^6  (treat letter+digits as exponent)
    - unicode minus becomes normal minus
    - keeps / for fractions
    - keeps spaces flexible
    """
    s = (s or "").strip()
    s = s.replace("−", "-").replace("·", "*").replace("×", "*")
    # Force classroom interpretation: 1/2x → (1/2)*x
    s = re.sub(r'(\d+)\s*/\s*(\d+)\s*([A-Za-z])', r'(\1/\2)*\3', s)

    
    # Remove extra spaces around operators to simplify pattern matching
    s = re.sub(r"\s+", " ", s).strip()

    # Convert letter followed by digits into exponent: x6 -> x^6, y12 -> y^12
    # This assumes teachers write 6x as "6x" (not "x6"), which is standard.
    s = re.sub(r"([A-Za-z])(\d+)\b", r"\1^\2", s)

    # Convert superscript unicode digits if they appear (optional safety)
    sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    # Replace patterns like x⁶ -> x^6
    s = re.sub(r"([A-Za-z])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: m.group(1) + "^" + m.group(2).translate(sup_map), s)

    return s


def _parse_expr_friendly(s: str) -> sp.Expr:
    s = _normalize_word_text(s)
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=True)


# ============================================================
# 2) Notebook-style steps (rendered line-by-line)
# ============================================================

def _linear_steps(left: sp.Expr, right: sp.Expr, x: sp.Symbol):
    expr = sp.simplify(left - right)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x)
    b = poly.coeff_monomial(1)

    steps = []
    steps.append(("Given", sp.latex(sp.Eq(left, right))))
    steps.append(("Step 1", sp.latex(sp.Eq(expr, 0))))
    steps.append(("Step 2", sp.latex(sp.Eq(a * x, -b))))

    if a == 0:
        steps.append(("Therefore", r"\therefore\ \mathrm{no\ unique\ solution}"))
        return steps

    sol = sp.simplify(-b / a)
    steps.append(("Step 3", sp.latex(sp.Eq(x, sol))))
    steps.append(("Therefore", r"\therefore\ x=" + sp.latex(sol)))
    return steps


def _quadratic_steps(left: sp.Expr, right: sp.Expr, x: sp.Symbol):
    expr = sp.simplify(left - right)
    steps = []
    steps.append(("Given", sp.latex(sp.Eq(left, right))))
    steps.append(("Step 1", sp.latex(sp.Eq(expr, 0))))

    factored = sp.factor(expr)
    if factored != expr:
        steps.append(("Step 2", sp.latex(sp.Eq(factored, 0))))
        sols = sp.solve(sp.Eq(expr, 0), x)
        sols = [sp.simplify(s) for s in sols]

        if len(sols) == 1:
            steps.append(("Therefore", r"\therefore\ x=" + sp.latex(sols[0])))
        elif len(sols) >= 2:
            steps.append(
                ("Therefore",
                 r"\therefore\ x=" + sp.latex(sols[0]) + r",\ x=" + sp.latex(sols[1]))
            )
        else:
            steps.append(("Therefore", r"\therefore\ \mathrm{no\ real\ solutions}"))

        return steps[:5]

    # Quadratic Formula path
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x**2)
    b = poly.coeff_monomial(x)
    c = poly.coeff_monomial(1)
    disc = sp.simplify(b**2 - 4*a*c)

    steps.append(("Step 2", r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}"))
    steps.append(("Step 3",
                  r"x=\frac{" + sp.latex(-b) +
                  r"\pm\sqrt{" + sp.latex(disc) +
                  r"}}{" + sp.latex(2*a) + r"}"))

    sols = sp.solve(sp.Eq(expr, 0), x)
    sols = [sp.simplify(s) for s in sols]

    if len(sols) == 1:
        steps.append(("Therefore", r"\therefore\ x=" + sp.latex(sols[0])))
    elif len(sols) >= 2:
        steps.append(
            ("Therefore",
             r"\therefore\ x=" + sp.latex(sols[0]) + r",\ x=" + sp.latex(sols[1]))
        )
    else:
        steps.append(("Therefore", r"\therefore\ \mathrm{no\ real\ solutions}"))

    return steps[:5]


def _expression_steps(expr: sp.Expr):
    simp = sp.simplify(expr)
    return [
        ("Given", sp.latex(expr)),
        ("Step 1", sp.latex(simp)),
        ("Therefore", r"\therefore\ " + sp.latex(simp)),
    ]







def compute_steps_from_input(raw: str, var: str = "x") -> Tuple[str, Optional[List[Tuple[str, str]]]]:
    """
    Returns:
      display_latex: what to render for the problem statement
      steps: list of (label, latex) if solvable; otherwise None (still renders the problem)
    """
    x = sp.Symbol(var)
    raw_norm = _normalize_word_text(raw)

    # Display: we render the *normalized* version so x6 displays as x^6 nicely
    display_latex = raw_norm

    if "=" in raw_norm:
        left_s, right_s = raw_norm.split("=", 1)
        try:
            left = _parse_expr_friendly(left_s)
            right = _parse_expr_friendly(right_s)
        except Exception:
            return display_latex, None

        expr = sp.simplify(left - right)
        try:
            deg = sp.Poly(expr, x).degree()
        except Exception:
            deg = 1

        if deg == 2:
            return sp.latex(sp.Eq(left, right)), _quadratic_steps(left, right, x)
        return sp.latex(sp.Eq(left, right)), _linear_steps(left, right, x)

    # Expression
    try:
        expr = _parse_expr_friendly(raw_norm)
        return sp.latex(expr), _expression_steps(expr)
    except Exception:
        return display_latex, None


# ============================================================
# 3) Render single LaTeX expressions to PNG (no environments)
# ============================================================

def _render_math_png(math_latex: str, font_size: int = 16, dpi: int = 300) -> str:
    math_latex = (math_latex or "").strip()
    if math_latex.startswith("$") and math_latex.endswith("$"):
        math_latex = math_latex[1:-1].strip()

    # Always render as math
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


def _draw_worked_lines(c: Canvas, steps: List[Tuple[str, str]], x0: float, x1: float, page_h: float):
    pad_x = 0.25 * inch
    box_x0 = x0 + pad_x
    box_x1 = x1 - pad_x
    box_w = box_x1 - box_x0

    top = page_h - 1.2 * inch
    line_gap = 1.05 * inch

    for idx, (label, math_ltx) in enumerate(steps[:5]):
        y = top - idx * line_gap

        c.setFont("Helvetica-Bold", 11)
        c.drawString(box_x0, y, f"{label}:")

        # Math to the right
        math_x_center = box_x0 + (box_w * 0.64)
        _draw_math_centered(c, math_ltx, math_x_center, y + 0.10 * inch, box_w * 0.80, 0.70 * inch, 16)


# ============================================================
# 4) Build PDF with your fold layout
# ============================================================
def build_foldable_pdf(out_path: str,
                       eq1_ltx: str, eq1_steps: Optional[List[Tuple[str, str]]],
                       eq2_ltx: str, eq2_steps: Optional[List[Tuple[str, str]]],
                       eq3_ltx: str, eq3_steps: Optional[List[Tuple[str, str]]]):
    """
    Horizontal bands layout to match the sample:
      Fold lines are HORIZONTAL at y = 2.75", 5.5", 8.25" from the TOP.

    FRONT (top to bottom bands):
      Band 1: #2 problem only + "Fold center, These 2 folds facing each other"
      Band 2: #3 worked + "Fold 3rd. This side out"
      Band 3: Open First + "Fold 1st. This side out."
      Band 4: #1 problem only + "Fold 2nd. Fold under"

    BACK (rotated 180 degrees):
      Band 1: #1 worked
      Band 2: empty
      Band 3: #3 problem only
      Band 4: #2 worked
    """
    page_w, page_h = letter
    c = Canvas(out_path, pagesize=letter)

    # Horizontal band boundaries (measured from TOP to match your directions)
    # y_from_top in inches: 0, 2.75, 5.5, 8.25, 11
    y_top = page_h
    y_marks_from_top = [0.0, 2.75 * inch, 5.5 * inch, 8.25 * inch, 11.0 * inch]
    ys = [y_top - m for m in y_marks_from_top]  # convert to canvas y coords
    # ys = [top, line1, line2, line3, bottom]

    def fold_lines_horizontal():
        c.saveState()
        c.setDash(6, 6)
        c.setLineWidth(1)
        for y in (ys[1], ys[2], ys[3]):
            c.line(0.5 * inch, y, page_w - 0.5 * inch, y)
        c.restoreState()

    def draw_boxed_label(x: float, y: float, text: str):
        # top-left label box inside a band (y is band top)
        box_w, box_h = 0.55 * inch, 0.35 * inch
        pad_x = 0.35 * inch
        pad_y = 0.35 * inch
        c.saveState()
        c.setLineWidth(1.2)
        c.rect(x + pad_x, y - pad_y - box_h, box_w, box_h, stroke=1, fill=0)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(x + pad_x + box_w / 2, y - pad_y - box_h + 0.10 * inch, text)
        c.restoreState()

    def band_rect(band_index: int):
        # band_index: 0=top band, 3=bottom band
        band_top = ys[band_index]
        band_bottom = ys[band_index + 1]
        return band_top, band_bottom

    def band_content_box(band_index: int):
        # returns (content_left, content_right, content_bottom, content_top)
        band_top, band_bottom = band_rect(band_index)
        left = 0.6 * inch
        right = page_w - 0.6 * inch
        top = band_top - 0.55 * inch
        bottom = band_bottom + 0.45 * inch
        return left, right, bottom, top

    def draw_math_in_band_center(band_index: int, latex: str, font_size: int = 16):
        left, right, bottom, top = band_content_box(band_index)
        cx = (left + right) / 2
        cy = (bottom + top) / 2
        _draw_math_centered(c, latex, cx, cy, (right - left), (top - bottom), font_size)

    def draw_worked_in_band(band_index: int, steps: List[Tuple[str, str]]):
        # Reuse your line-by-line notebook renderer but mapped to band box
        left, right, bottom, top = band_content_box(band_index)

        # Temporarily adapt the existing _draw_worked_lines by creating a wrapper
        # that uses x0/x1 and page_h consistent with current coordinate system.
        # We'll place labels/math starting near the top of the band.
        pad_x = 0.15 * inch
        x0 = left + pad_x
        x1 = right - pad_x

        # Custom vertical placement inside the band:
        line_gap = (top - bottom) / 4.2
        start_y = top - 0.15 * inch

        for idx, (label, math_ltx) in enumerate(steps[:5]):
            y = start_y - idx * line_gap
            c.setFont("Helvetica-Bold", 11)
            c.drawString(x0, y, f"{label}:")

            math_region_w = (x1 - x0) * 0.78
            math_region_h = min(0.70 * inch, line_gap * 0.9)
            math_x_center = x0 + (x1 - x0) * 0.62
            math_y_center = y + 0.08 * inch

            _draw_math_centered(c, math_ltx, math_x_center, math_y_center, math_region_w, math_region_h, 16)

    # ---------------- FRONT ----------------
    fold_lines_horizontal()

    # Band 1 (top): #2 problem only + top note
    draw_boxed_label(0, ys[0], "#2")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(page_w / 2, ys[0] - 0.35 * inch, "Fold center, These 2 folds")
    c.drawCentredString(page_w / 2, ys[0] - 0.55 * inch, "facing each other")
    draw_math_in_band_center(0, eq2_ltx, 16)

    # Band 2: #3 worked + fold note on the fold line above it
    # Put the note centered on the fold line between band 1 and 2 (ys[1])
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, ys[1] - 0.20 * inch, "Fold 3rd. This side out")
    draw_boxed_label(0, ys[1], "#3")
    if eq3_steps:
        draw_worked_in_band(1, eq3_steps)
    else:
        draw_math_in_band_center(1, eq3_ltx, 16)

    # Band 3: Open First + fold note on fold line between band 2 and 3 (ys[2])
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, ys[2] - 0.20 * inch, "Fold 1st. This side out.")
    c.setFont("Helvetica-Bold", 24)
    band3_top, band3_bottom = band_rect(2)
    c.drawCentredString(page_w / 2, (band3_top + band3_bottom) / 2, "Open First")

    # Band 4 (bottom): #1 problem only + fold note on fold line between band 3 and 4 (ys[3])
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, ys[3] - 0.20 * inch, "Fold 2nd. Fold under")
    draw_boxed_label(0, ys[3], "#1")
    draw_math_in_band_center(3, eq1_ltx, 16)

    c.showPage()

    # ---------------- BACK (rotated 180°) ----------------
    c.saveState()
    c.translate(page_w, page_h)
    c.rotate(180)

    fold_lines_horizontal()

    # Back Band 1: #1 worked
    draw_boxed_label(0, ys[0], "#1")
    if eq1_steps:
        draw_worked_in_band(0, eq1_steps)
    else:
        draw_math_in_band_center(0, eq1_ltx, 16)

    # Back Band 2: empty (do nothing)

    # Back Band 3: #3 problem only
    draw_boxed_label(0, ys[2], "#3")
    draw_math_in_band_center(2, eq3_ltx, 16)

    # Back Band 4: #2 worked
    draw_boxed_label(0, ys[3], "#2")
    if eq2_steps:
        draw_worked_in_band(3, eq2_steps)
    else:
        draw_math_in_band_center(3, eq2_ltx, 16)

    c.restoreState()
    c.showPage()
    c.save()

# ============================================================
# 5) Streamlit UI
# ============================================================

st.set_page_config(page_title="Word Copy/Paste Math Foldable", layout="centered")
st.title("Printable Math Foldable (Word Copy/Paste Friendly)")

st.write(
    "Type in Word using the Equation button, then copy/paste here.\n\n"
    "✅ This app will automatically treat **x6** as **x^6** (exponent) and render real math."
)

with st.expander("What to paste (examples)"):
    st.markdown(
        """
- Linear: `1/2x + 3 = 11`
- Linear: `3x - 5 = 16`
- Exponent from Word (plain): `x6 + 2x = 10`  (interprets as x^6)
- Quadratic: `x2 - 5x + 6 = 0`  (interprets as x^2)
"""
    )

p1 = st.text_area("Problem #1", value="1/2x + 3 = 11", height=70)
p2 = st.text_area("Problem #2", value="3x - 5 = 16", height=70)
p3 = st.text_area("Problem #3", value="x2 - 5x + 6 = 0", height=70)

if st.button("Generate Foldable PDF", type="primary"):
    try:
        eq1_ltx, eq1_steps = compute_steps_from_input(p1, var="x")
        eq2_ltx, eq2_steps = compute_steps_from_input(p2, var="x")
        eq3_ltx, eq3_steps = compute_steps_from_input(p3, var="x")

        out_path = "foldable_output.pdf"
        build_foldable_pdf(out_path, eq1_ltx, eq1_steps, eq2_ltx, eq2_steps, eq3_ltx, eq3_steps)

        with open(out_path, "rb") as f:
            st.download_button(
                "Download foldable_output.pdf",
                data=f,
                file_name="foldable_output.pdf",
                mime="application/pdf",
            )

        if (eq1_steps is None) or (eq2_steps is None) or (eq3_steps is None):
            st.info(
                "PDF generated. If any problem couldn't be solved automatically, "
                "it will still print as a clean equation."
            )

    except Exception as e:
        st.error("Something went wrong while generating the foldable.")
        st.exception(e)



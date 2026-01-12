# foldable_engine.py
# Engine for Streamlit app: generates a 2-page foldable PDF with LaTeX-rendered math
# and notebook-style multi-step solutions.

from __future__ import annotations

import os
import re
import tempfile
from typing import Tuple, List, Optional

import sympy as sp

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Input normalization
# ----------------------------

def _normalize_math(s: str) -> str:
    s = s.strip()
    s = s.replace("−", "-").replace("^", "**").replace("·", "*")

    # Insert * for implicit multiplication patterns: 2x -> 2*x, 3(x+1) -> 3*(x+1)
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)
    s = re.sub(r"([a-zA-Z])\(", r"\1*(", s)
    s = re.sub(r"\)([a-zA-Z0-9])", r")*\1", s)
    return s


# ----------------------------
# LaTeX helpers
# ----------------------------

def ltx(x) -> str:
    return sp.latex(x)

def eq_ltx(L, R) -> str:
    return sp.latex(sp.Eq(L, R))

def step(label: str, math: str) -> str:
    # aligned line: "Step 1:  <math>"
    return rf"\text{{{label}:}}\; & {math}"

def final_box(math: str) -> str:
    return rf"\boxed{{{math}}}"

def notebook_aligned(step_lines: List[str]) -> str:
    body = r" \\ ".join(step_lines)
    return r"\begin{aligned}" + body + r"\end{aligned}"


# ----------------------------
# Notebook-style step generation
# ----------------------------

def _linear_steps(left: sp.Expr, right: sp.Expr, x: sp.Symbol) -> List[str]:
    expr = sp.simplify(left - right)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x)
    b = poly.coeff_monomial(1)

    lines: List[str] = []
    lines.append(step("Given", eq_ltx(left, right)))
    lines.append(step("Step 1", eq_ltx(expr, 0)))

    if b != 0:
        lines.append(step("Step 2", eq_ltx(a * x, -b)))
    else:
        lines.append(step("Step 2", eq_ltx(a * x, 0)))

    if a != 0:
        sol = sp.simplify((-b) / a)
        lines.append(step("Step 3", eq_ltx(x, sol)))
        lines.append(step("Answer", final_box(rf"x={ltx(sol)}")))
    else:
        lines.append(step("Answer", r"\text{No unique solution}"))

    return lines[:5]


def _quadratic_steps(left: sp.Expr, right: sp.Expr, x: sp.Symbol) -> List[str]:
    expr = sp.simplify(left - right)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x**2)
    b = poly.coeff_monomial(x)
    c = poly.coeff_monomial(1)

    lines: List[str] = []
    lines.append(step("Given", eq_ltx(left, right)))
    lines.append(step("Step 1", eq_ltx(expr, 0)))

    factored = sp.factor(expr)
    sols = sp.solve(sp.Eq(expr, 0), x)

    # If it factors nicely, use factoring path
    if factored != expr:
        lines.append(step("Step 2", eq_ltx(factored, 0)))
        if sols:
            shown = [sp.simplify(s) for s in sols[:2]]
            if len(shown) == 1:
                lines.append(step("Step 3", rf"x={ltx(shown[0])}"))
                lines.append(step("Answer", final_box(rf"x={ltx(shown[0])}")))
            else:
                soltxt = rf"x={ltx(shown[0])},\; x={ltx(shown[1])}"
                lines.append(step("Step 3", soltxt))
                lines.append(step("Answer", final_box(soltxt)))
        else:
            lines.append(step("Answer", r"\text{No real solutions}"))
        return lines[:5]

    # Otherwise, quadratic formula path
    lines.append(step("Step 2", r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}"))
    disc = sp.simplify(b**2 - 4*a*c)
    lines.append(step("Step 3", rf"x=\frac{{{ltx(-b)}\pm\sqrt{{{ltx(disc)}}}}}{{{ltx(2*a)}}}"))

    if sols:
        shown = [sp.simplify(s) for s in sols[:2]]
        if len(shown) == 1:
            lines.append(step("Answer", final_box(rf"x={ltx(shown[0])}")))
        else:
            soltxt = rf"x={ltx(shown[0])},\; x={ltx(shown[1])}"
            lines.append(step("Answer", final_box(soltxt)))
    else:
        lines.append(step("Answer", r"\text{No real solutions}"))

    return lines[:5]


def _expression_steps(expr: sp.Expr) -> List[str]:
    lines: List[str] = []
    lines.append(step("Given", ltx(expr)))

    expanded = sp.expand(expr)
    if expanded != expr:
        lines.append(step("Step 1", ltx(expanded)))
        current = expanded
    else:
        current = sp.simplify(expr)
        lines.append(step("Step 1", ltx(current)))

    # Try simplifying rational expressions (cancel)
    together = sp.together(current)
    canceled = sp.cancel(together)
    if canceled != current:
        lines.append(step("Step 2", ltx(canceled)))
        current = canceled
    else:
        factored = sp.factor(current)
        if factored != current:
            lines.append(step("Step 2", ltx(factored)))
            current = factored

    final = sp.simplify(current)
    lines.append(step("Answer", final_box(ltx(final))))
    return lines[:5]


def problem_and_steps(raw: str, var: str = "x") -> Tuple[str, List[str]]:
    """
    Returns:
      problem_latex (for problem-only panels)
      steps (list of aligned 'notebook' lines for worked-solution panels)
    """
    x = sp.Symbol(var)
    norm = _normalize_math(raw)

    if "=" in norm:
        left_str, right_str = norm.split("=", 1)
        left = sp.sympify(left_str)
        right = sp.sympify(right_str)

        expr = sp.simplify(left - right)
        deg = sp.Poly(expr, x).degree() if expr.has(x) else 0

        problem_latex = eq_ltx(left, right)

        if deg == 1:
            steps = _linear_steps(left, right, x)
        elif deg == 2:
            steps = _quadratic_steps(left, right, x)
        else:
            sols = sp.solve(sp.Eq(expr, 0), x)
            steps = [
                step("Given", eq_ltx(left, right)),
                step("Step 1", eq_ltx(expr, 0)),
            ]
            if sols:
                shown = sols[:2]
                soltxt = r",\; ".join([rf"x={ltx(sp.simplify(s))}" for s in shown])
                steps.append(step("Step 2", soltxt))
                steps.append(step("Answer", final_box(soltxt)))
            else:
                steps.append(step("Answer", r"\text{No solution}"))
            steps = steps[:5]

        return problem_latex, steps

    # Expression
    expr = sp.sympify(norm)
    problem_latex = ltx(expr)
    steps = _expression_steps(expr)
    return problem_latex, steps


# ----------------------------
# LaTeX -> PNG rendering
# ----------------------------

def _render_to_png(math_latex: str, font_size: int = 16, dpi: int = 300) -> str:
    s = math_latex.strip()
    if not (s.startswith("$") and s.endswith("$")):
        s = f"${s}$"

    # Measure
    fig = plt.figure()
    fig.patch.set_alpha(0)
    t = fig.text(0, 0, s, fontsize=font_size)
    fig.canvas.draw()
    bbox = t.get_window_extent()
    pad_px = 14
    w_in = (bbox.width + pad_px * 2) / dpi
    h_in = (bbox.height + pad_px * 2) / dpi
    plt.close(fig)

    # Render
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, s, fontsize=font_size, ha="center", va="center")

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def _draw_math_centered(
    c: Canvas,
    math_latex: str,
    x_center: float,
    y_center: float,
    box_w: float,
    box_h: float,
    target_font_size: int = 16
) -> None:
    png_path = _render_to_png(math_latex, font_size=target_font_size)
    try:
        img = ImageReader(png_path)
        iw, ih = img.getSize()
        scale = min(box_w / iw, box_h / ih, 1.0)
        dw, dh = iw * scale, ih * scale
        c.drawImage(img, x_center - dw/2, y_center - dh/2, width=dw, height=dh, mask="auto")
    finally:
        try:
            os.remove(png_path)
        except OSError:
            pass


# ----------------------------
# PDF layout for your foldable
# ----------------------------

def _draw_dashed_fold_lines(c: Canvas, page_h: float) -> None:
    xs = [2.75 * inch, 5.5 * inch, 8.25 * inch]
    c.saveState()
    c.setDash(6, 6)
    c.setLineWidth(1)
    for x in xs:
        c.line(x, 0.5 * inch, x, page_h - 0.5 * inch)
    c.restoreState()


def _draw_boxed_label(c: Canvas, x: float, y: float, text: str) -> None:
    box_w, box_h = 0.55 * inch, 0.35 * inch
    c.saveState()
    c.setLineWidth(1.2)
    c.rect(x, y - box_h, box_w, box_h, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(x + box_w / 2, y - box_h + 0.10 * inch, text)
    c.restoreState()


def _quadrant_content_box(
    c: Canvas,
    x0: float,
    x1: float,
    page_h: float,
    label: Optional[str],
    top_text: Optional[str],
    bottom_text: Optional[str],
):
    pad_x = 0.20 * inch
    quad_center_x = (x0 + x1) / 2

    if label:
        _draw_boxed_label(c, x0 + pad_x, page_h - 0.75 * inch, label)

    if top_text:
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(quad_center_x, page_h - 0.60 * inch, top_text)

    if bottom_text:
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(quad_center_x, 0.75 * inch, bottom_text)

    # Content area for centering math
    content_w = (x1 - x0) - 2 * pad_x
    content_h = page_h - 2.35 * inch
    content_y = 1.20 * inch

    return content_y, content_w, content_h


def build_foldable(
    out_path: str,
    eq1: Tuple[str, str, List[str]],
    eq2: Tuple[str, str, List[str]],
    eq3: Tuple[str, str, List[str]],
) -> str:
    page_w, page_h = letter
    c = Canvas(out_path, pagesize=letter)

    xA = 0.0
    xB = 2.75 * inch
    xC = 5.5 * inch
    xD = 8.25 * inch
    xE = page_w

    p1, block1, _ = eq1
    p2, block2, _ = eq2
    p3, block3, _ = eq3

    # -------- FRONT --------
    _draw_dashed_fold_lines(c, page_h)

    # Front Q1: #2 problem only
    cy, cw, ch = _quadrant_content_box(
        c, xA, xB, page_h, "#2",
        "These two folds face each other",
        "Fold 3rd.  This side out."
    )
    _draw_math_centered(c, f"${p2}$", (xA + xB)/2, cy + ch/2, cw, ch, 16)

    # Front Q2: #3 worked (multi-step)
    cy, cw, ch = _quadrant_content_box(c, xB, xC, page_h, "#3", None, None)
    _draw_math_centered(c, f"${block3}$", (xB + xC)/2, cy + ch/2, cw, ch, 16)

    # Front Q3: Open First
    _quadrant_content_box(c, xC, xD, page_h, None, "Fold 1st.  This side out.", None)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString((xC + xD)/2, page_h/2, "Open First")

    # Front Q4: #1 problem only
    cy, cw, ch = _quadrant_content_box(c, xD, xE, page_h, "#1", "Fold 2nd.  Fold under.", None)
    _draw_math_centered(c, f"${p1}$", (xD + xE)/2, cy + ch/2, cw, ch, 16)

    c.showPage()

    # -------- BACK (rotated 180°) --------
    c.saveState()
    c.translate(page_w, page_h)
    c.rotate(180)

    _draw_dashed_fold_lines(c, page_h)

    # Back Q1: #1 worked
    cy, cw, ch = _quadrant_content_box(c, xA, xB, page_h, "#1", None, None)
    _draw_math_centered(c, f"${block1}$", (xA + xB)/2, cy + ch/2, cw, ch, 16)

    # Back Q2: empty (do nothing)

    # Back Q3: #3 problem only
    cy, cw, ch = _quadrant_content_box(c, xC, xD, page_h, "#3", None, None)
    _draw_math_centered(c, f"${p3}$", (xC + xD)/2, cy + ch/2, cw, ch, 16)

    # Back Q4: #2 worked
    cy, cw, ch = _quadrant_content_box(c, xD, xE, page_h, "#2", None, None)
    _draw_math_centered(c, f"${block2}$", (xD + xE)/2, cy + ch/2, cw, ch, 16)

    c.restoreState()
    c.showPage()
    c.save()
    return out_path

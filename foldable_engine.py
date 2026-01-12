from __fu"""
foldable_trifold_latex_notebook_steps.py

Teacher enters 3 equations/expressions.
Creates a 2-page foldable PDF (Front + Back) with:
- true stacked fractions/exponents (LaTeX rendering)
- notebook-style solutions with Step labels
- dashed fold lines at 2.75", 5.5", 8.25" on front and back
- back page rotated 180 degrees

Install:
  pip install reportlab sympy matplotlib pillow
Run:
  python foldable_trifold_latex_notebook_steps.py
Output:
  foldable_output.pdf
"""

ture__ import annotations

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

    # Insert * for implicit multiplication patterns
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)     # 2x, 3(x+1)
    s = re.sub(r"([a-zA-Z])\(", r"\1*(", s)         # x(x+1)
    s = re.sub(r"\)([a-zA-Z0-9])", r")*\1", s)      # )( or )2
    return s


# ----------------------------
# LaTeX helpers
# ----------------------------

def ltx(x) -> str:
    return sp.latex(x)

def eq_ltx(L, R) -> str:
    return sp.latex(sp.Eq(L, R))


def notebook_aligned(step_lines: List[str]) -> str:
    """
    Builds a notebook-style aligned block with:
      Step 1:  ...
      Step 2:  ...
    Uses an alignment point so steps line up nicely.
    """
    # Each line should already contain "&" for alignment.
    body = r" \\ ".join(step_lines)
    return r"\begin{aligned}" + body + r"\end{aligned}"


def step(label: str, math: str) -> str:
    """
    One aligned line:
      \text{Step 1: } & <math>
    """
    return rf"\text{{{label}}}\; & {math}"


def final_box(math: str) -> str:
    return rf"\boxed{{{math}}}"


# ----------------------------
# Step generation (Notebook style)
# ----------------------------

def linear_steps(left: sp.Expr, right: sp.Expr, x: sp.Symbol) -> List[str]:
    """
    Notebook steps for linear equations:
      Step 1: combine like terms (move to one side)
      Step 2: isolate ax
      Step 3: divide
      Final: boxed
    """
    expr = sp.simplify(left - right)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x)
    b = poly.coeff_monomial(1)

    lines: List[str] = []
    lines.append(step("Given", eq_ltx(left, right)))

    # Step 1: move everything to left side
    lines.append(step("Step 1", eq_ltx(expr, 0)))

    # Step 2: move constant term
    if b != 0:
        lines.append(step("Step 2", eq_ltx(a*x, -b)))
    else:
        lines.append(step("Step 2", eq_ltx(a*x, 0)))

    # Step 3: divide by a
    if a != 0:
        sol = sp.simplify((-b)/a)
        lines.append(step("Step 3", eq_ltx(x, sol)))
        lines.append(step("Answer", final_box(rf"x={ltx(sol)}")))
    else:
        lines.append(step("Answer", r"\text{No unique solution}"))

    # Keep it tight
    return lines[:5]


def quadratic_steps(left: sp.Expr, right: sp.Expr, x: sp.Symbol) -> List[str]:
    """
    Notebook steps for quadratics:
    - Step 1: standard form
    - Step 2: factor OR quadratic formula setup
    - Step 3: solve
    - Answer: boxed solutions
    """
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

    # If factors nicely, use zero product property
    if factored != expr:
        lines.append(step("Step 2", eq_ltx(factored, 0)))

        if sols:
            # show solutions line
            if len(sols) == 1:
                s0 = sp.simplify(sols[0])
                lines.append(step("Step 3", eq_ltx(x, s0)))
                lines.append(step("Answer", final_box(rf"x={ltx(s0)}")))
            else:
                s0 = sp.simplify(sols[0])
                s1 = sp.simplify(sols[1])
                lines.append(step("Step 3", rf"x={ltx(s0)},\; x={ltx(s1)}"))
                lines.append(step("Answer", final_box(rf"x={ltx(s0)},\; x={ltx(s1)}")))
        else:
            lines.append(step("Answer", r"\text{No real solutions}"))

        return lines[:5]

    # Otherwise: quadratic formula (still notebook style)
    lines.append(step("Step 2", r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}"))

    disc = sp.simplify(b**2 - 4*a*c)
    lines.append(step(
        "Step 3",
        rf"x=\frac{{{ltx(-b)}\pm\sqrt{{{ltx(disc)}}}}}{{{ltx(2*a)}}}"
    ))

    if sols:
        if len(sols) == 1:
            s0 = sp.simplify(sols[0])
            lines.append(step("Answer", final_box(rf"x={ltx(s0)}")))
        else:
            s0 = sp.simplify(sols[0])
            s1 = sp.simplify(sols[1])
            lines.append(step("Answer", final_box(rf"x={ltx(s0)},\; x={ltx(s1)}")))
    else:
        lines.append(step("Answer", r"\text{No real solutions}"))

    return lines[:5]


def expression_steps(expr: sp.Expr) -> List[str]:
    """
    Notebook-style simplification:
      Given
      Step 1: expand OR combine
      Step 2: factor OR cancel
      Answer: boxed final
    """
    lines: List[str] = []
    lines.append(step("Given", ltx(expr)))

    expanded = sp.expand(expr)
    if expanded != expr:
        lines.append(step("Step 1", ltx(expanded)))
        current = expanded
    else:
        current = expr
        lines.append(step("Step 1", ltx(sp.simplify(current))))

    # Try rational simplification/cancel
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
      step_lines (notebook aligned lines for worked panels)
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
            steps = linear_steps(left, right, x)
        elif deg == 2:
            steps = quadratic_steps(left, right, x)
        else:
            # Fallback: show move to zero + solve
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
    steps = expression_steps(expr)
    return problem_latex, steps


# ----------------------------
# LaTeX -> PNG rendering
# ----------------------------

def render_to_png(math_latex: str, font_size: int = 16, dpi: int = 300) -> str:
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


def draw_math_centered(c: Canvas, math_latex: str, x_center: float, y_center: float,
                       box_w: float, box_h: float, target_font_size: int = 16) -> None:
    png_path = render_to_png(math_latex, font_size=target_font_size)
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
# PDF layout
# ----------------------------

def draw_dashed_fold_lines(c: Canvas, page_h: float) -> None:
    xs = [2.75 * inch, 5.5 * inch, 8.25 * inch]
    c.saveState()
    c.setDash(6, 6)
    c.setLineWidth(1)
    for x in xs:
        c.line(x, 0.5 * inch, x, page_h - 0.5 * inch)
    c.restoreState()


def draw_boxed_label(c: Canvas, x: float, y: float, text: str) -> None:
    box_w, box_h = 0.55 * inch, 0.35 * inch
    c.saveState()
    c.setLineWidth(1.2)
    c.rect(x, y - box_h, box_w, box_h, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(x + box_w / 2, y - box_h + 0.10 * inch, text)
    c.restoreState()


def quadrant_content_box(c: Canvas, x0: float, x1: float, page_h: float,
                         label: Optional[str], top_text: Optional[str], bottom_text: Optional[str]):
    pad_x = 0.20 * inch
    quad_center_x = (x0 + x1) / 2

    if label:
        draw_boxed_label(c, x0 + pad_x, page_h - 0.75 * inch, label)

    if top_text:
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(quad_center_x, page_h - 0.60 * inch, top_text)

    if bottom_text:
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(quad_center_x, 0.75 * inch, bottom_text)

    # generous content area for centering
    content_x = x0 + pad_x
    content_w = (x1 - x0) - 2 * pad_x
    content_y = 1.20 * inch
    content_h = page_h - 2.35 * inch
    return content_x, content_y, content_w, content_h


def build_foldable(out_path: str,
                   eq1: Tuple[str, str, List[str]],
                   eq2: Tuple[str, str, List[str]],
                   eq3: Tuple[str, str, List[str]]) -> str:

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
    draw_dashed_fold_lines(c, page_h)

    # Front Q1: #2 problem only
    cx, cy, cw, ch = quadrant_content_box(
        c, xA, xB, page_h, "#2",
        "These two folds face each other",
        "Fold 3rd.  This side out."
    )
    draw_math_centered(c, f"${p2}$", (xA + xB)/2, cy + ch/2, cw, ch, 16)

    # Front Q2: #3 worked
    cx, cy, cw, ch = quadrant_content_box(c, xB, xC, page_h, "#3", None, None)
    draw_math_centered(c, f"${block3}$", (xB + xC)/2, cy + ch/2, cw, ch, 16)

    # Front Q3: Open First
    cx, cy, cw, ch = quadrant_content_box(c, xC, xD, page_h, None, "Fold 1st.  This side out.", None)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString((xC + xD)/2, page_h/2, "Open First")

    # Front Q4: #1 problem only
    cx, cy, cw, ch = quadrant_content_box(c, xD, xE, page_h, "#1", "Fold 2nd.  Fold under.", None)
    draw_math_centered(c, f"${p1}$", (xD + xE)/2, cy + ch/2, cw, ch, 16)

    c.showPage()

    # -------- BACK (rotated 180°) --------
    c.saveState()
    c.translate(page_w, page_h)
    c.rotate(180)

    draw_dashed_fold_lines(c, page_h)

    # Back Q1: #1 worked
    cx, cy, cw, ch = quadrant_content_box(c, xA, xB, page_h, "#1", None, None)
    draw_math_centered(c, f"${block1}$", (xA + xB)/2, cy + ch/2, cw, ch, 16)

    # Back Q2: empty

    # Back Q3: #3 problem only
    cx, cy, cw, ch = quadrant_content_box(c, xC, xD, page_h, "#3", None, None)
    draw_math_centered(c, f"${p3}$", (xC + xD)/2, cy + ch/2, cw, ch, 16)

    # Back Q4: #2 worked
    cx, cy, cw, ch = quadrant_content_box(c, xD, xE, page_h, "#2", None, None)
    draw_math_centered(c, f"${block2}$", (xD + xE)/2, cy + ch/2, cw, ch, 16)

    c.restoreState()
    c.showPage()
    c.save()
    return out_path


def main():
    print("\n=== Foldable Generator (Notebook Steps + LaTeX) ===")
    print("Enter 3 expressions or equations.")
    print("Examples:")
    print("  2x+3=11")
    print("  x^2-5x+6=0")
    print("  (x^3*y^2)^4/(x*y^5)\n")

    raw1 = input("Enter Equation/Expression #1: ").strip()
    raw2 = input("Enter Equation/Expression #2: ").strip()
    raw3 = input("Enter Equation/Expression #3: ").strip()

    p1, steps1 = problem_and_steps(raw1, var="x")
    p2, steps2 = problem_and_steps(raw2, var="x")
    p3, steps3 = problem_and_steps(raw3, var="x")

    block1 = notebook_aligned(steps1)
    block2 = notebook_aligned(steps2)
    block3 = notebook_aligned(steps3)

    build_foldable(
        "foldable_output.pdf",
        (p1, block1, steps1),
        (p2, block2, steps2),
        (p3, block3, steps3),
    )

    print("\nDone! Created: foldable_output.pdf")
    print("Printing tip: Print double-sided. If alignment is off, try 'Flip on short edge'.\n")



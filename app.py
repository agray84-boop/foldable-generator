import streamlit as st
import os
import re
import tempfile
from typing import List

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


# ----------------------------
# Math parsing (teacher-friendly)
# ----------------------------

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,  # lets teachers use x^2
)

def _normalize_math(s: str) -> str:
    s = s.strip()
    s = s.replace("−", "-").replace("·", "*")
    return s

def _parse(s: str) -> sp.Expr:
    s = _normalize_math(s)
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=True)


# ----------------------------
# LaTeX helpers (mathtext-safe)
# ----------------------------

def ltx(x) -> str:
    return sp.latex(x)

def eq_ltx(L, R) -> str:
    return sp.latex(sp.Eq(L, R))

def _label(label: str) -> str:
    # mathtext supports \mathrm and escaped spaces
    return r"\mathrm{" + label.replace(" ", r"\ ") + r"}"

def notebook_block(lines: List[str]) -> str:
    """
    Matplotlib mathtext does NOT reliably support \begin{aligned} / \text{} on Streamlit Cloud.
    Use array environment instead (much more compatible).
    """
    body = r" \\ ".join(lines)
    return r"\begin{array}{l}" + body + r"\end{array}"


# ----------------------------
# Notebook-style steps (Alg I / Alg II friendly)
# ----------------------------

def linear_steps(left, right, x):
    # Solve ax + b = 0 style after simplifying
    expr = sp.simplify(left - right)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x)
    b = poly.coeff_monomial(1)

    steps: List[str] = []
    steps.append(_label("Given:") + r"\quad " + eq_ltx(left, right))
    steps.append(_label("Step 1:") + r"\quad " + eq_ltx(expr, 0))
    steps.append(_label("Step 2:") + r"\quad " + eq_ltx(a * x, -b))

    # Avoid division by zero edge cases
    if a == 0:
        steps.append(_label("Answer:") + r"\quad " + r"\mathrm{No\ unique\ solution}")
        return steps

    sol = sp.simplify(-b / a)
    steps.append(_label("Step 3:") + r"\quad " + eq_ltx(x, sol))
    steps.append(_label("Answer:") + r"\quad " + r"\boxed{x=" + ltx(sol) + r"}")
    return steps


def quadratic_steps(left, right, x):
    expr = sp.simplify(left - right)
    steps: List[str] = []
    steps.append(_label("Given:") + r"\quad " + eq_ltx(left, right))
    steps.append(_label("Step 1:") + r"\quad " + eq_ltx(expr, 0))

    # Try factoring first
    factored = sp.factor(expr)
    if factored != expr:
        steps.append(_label("Step 2:") + r"\quad " + eq_ltx(factored, 0))
        sols = sp.solve(sp.Eq(expr, 0), x)
        if sols:
            sols = [sp.simplify(s) for s in sols[:2]]
            if len(sols) == 1:
                steps.append(_label("Answer:") + r"\quad " + r"\boxed{x=" + ltx(sols[0]) + r"}")
            else:
                steps.append(
                    _label("Answer:") + r"\quad " +
                    r"\boxed{x=" + ltx(sols[0]) + r",\;x=" + ltx(sols[1]) + r"}"
                )
        else:
            steps.append(_label("Answer:") + r"\quad " + r"\mathrm{No\ real\ solutions}")
        return steps[:5]

    # If not factorable nicely, use quadratic formula (compact)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x**2)
    b = poly.coeff_monomial(x)
    c = poly.coeff_monomial(1)
    disc = sp.simplify(b**2 - 4*a*c)

    steps.append(_label("Step 2:") + r"\quad " + r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}")
    steps.append(
        _label("Step 3:") + r"\quad " +
        r"x=\frac{" + ltx(-b) + r"\pm\sqrt{" + ltx(disc) + r"}}{" + ltx(2*a) + r"}"
    )

    sols = sp.solve(sp.Eq(expr, 0), x)
    if sols:
        sols = [sp.simplify(s) for s in sols[:2]]
        if len(sols) == 1:
            steps.append(_label("Answer:") + r"\quad " + r"\boxed{x=" + ltx(sols[0]) + r"}")
        else:
            steps.append(
                _label("Answer:") + r"\quad " +
                r"\boxed{x=" + ltx(sols[0]) + r",\;x=" + ltx(sols[1]) + r"}"
            )
    else:
        steps.append(_label("Answer:") + r"\quad " + r"\mathrm{No\ real\ solutions}")

    return steps[:5]


def expression_steps(expr):
    simp = sp.simplify(expr)
    steps: List[str] = []
    steps.append(_label("Given:") + r"\quad " + ltx(expr))
    steps.append(_label("Step 1:") + r"\quad " + ltx(simp))
    steps.append(_label("Answer:") + r"\quad " + r"\boxed{" + ltx(simp) + r"}")
    return steps


def problem_and_steps(raw: str, var="x"):
    x = sp.Symbol(var)
    norm = _normalize_math(raw)

    # Equation case
    if "=" in norm:
        left_str, right_str = norm.split("=", 1)
        left = _parse(left_str)
        right = _parse(right_str)

        # Detect degree in x to choose linear vs quadratic steps
        expr = sp.simplify(left - right)
        try:
            deg = sp.Poly(expr, x).degree()
        except Exception:
            deg = 1  # safe fallback

        if deg == 2:
            return eq_ltx(left, right), quadratic_steps(left, right, x)
        else:
            return eq_ltx(left, right), linear_steps(left, right, x)

    # Expression (no equals)
    expr = _parse(norm)
    return ltx(expr), expression_steps(expr)


# ----------------------------
# LaTeX rendering to PNG (mathtext)
# ----------------------------

def render_png(math_latex: str, font_size: int = 16, dpi: int = 300) -> str:
    if not math_latex.startswith("$"):
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


def draw_centered(c: Canvas, math_latex: str, cx: float, cy: float, w: float, h: float):
    img_path = render_png(math_latex)
    img = ImageReader(img_path)
    iw, ih = img.getSize()
    scale = min(w / iw, h / ih)
    c.drawImage(
        img,
        cx - (iw * scale) / 2,
        cy - (ih * scale) / 2,
        iw * scale,
        ih * scale,
        mask="auto",
    )
    os.remove(img_path)


# ----------------------------
# PDF foldable layout
# ----------------------------

def build_foldable(out_path: str, eq1, eq2, eq3):
    """
    eqN = (problem_latex, steps_list)
    Front:
      Q1: #2 problem only
      Q2: #3 worked
      Q3: Open First (text)
      Q4: #1 problem only
    Back (rotated 180):
      Q1: #1 worked
      Q2: empty
      Q3: #3 problem only
      Q4: #2 worked
    """
    page_w, page_h = letter
    c = Canvas(out_path, pagesize=letter)

    xs = [0, 2.75 * inch, 5.5 * inch, 8.25 * inch, page_w]

    def folds():
        c.setDash(6, 6)
        for x in xs[1:-1]:
            c.line(x, 0.5 * inch, x, page_h - 0.5 * inch)
        c.setDash()

    # -------- FRONT --------
    folds()

    front = [eq2, eq3, None, eq1]
    for i in range(4):
        x0, x1 = xs[i], xs[i + 1]
        if front[i] is None:
            # Open First panel
            c.setFont("Helvetica-Bold", 24)
            c.drawCentredString((x0 + x1) / 2, page_h / 2, "Open First")
            continue

        prob_ltx, steps = front[i]

        # Q2 (index 1) is worked, others are problem-only
        if i == 1:
            block = notebook_block(steps)
            draw_centered(
                c,
                block,
                (x0 + x1) / 2,
                page_h / 2,
                (x1 - x0) - 0.4 * inch,
                page_h - 2.0 * inch,
            )
        else:
            draw_centered(
                c,
                prob_ltx,
                (x0 + x1) / 2,
                page_h / 2,
                (x1 - x0) - 0.4 * inch,
                page_h - 2.0 * inch,
            )

    c.showPage()

    # -------- BACK (rotated 180°) --------
    c.translate(page_w, page_h)
    c.rotate(180)
    folds()

    back = [eq1, None, eq3, eq2]
    for i in range(4):
        if back[i] is None:
            continue
        x0, x1 = xs[i], xs[i + 1]
        prob_ltx, steps = back[i]

        # Worked panels: Q1 and Q4 (indices 0 and 3). Q3 (index 2) is problem-only.
        if i in (0, 3):
            block = notebook_block(steps)
            draw_centered(
                c,
                block,
                (x0 + x1) / 2,
                page_h / 2,
                (x1 - x0) - 0.4 * inch,
                page_h - 2.0 * inch,
            )
        else:
            draw_centered(
                c,
                prob_ltx,
                (x0 + x1) / 2,
                page_h / 2,
                (x1 - x0) - 0.4 * inch,
                page_h - 2.0 * inch,
            )

    c.showPage()
    c.save()


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="Math Foldable Generator", layout="centered")
st.title("Printable Math Foldable Generator")
st.write("Enter **3 math problems** (equations or expressions), then click **Generate**.")

with st.expander("Input tips"):
    st.markdown(
        """
- Linear equations: `3x-5=16`
- Quadratics: `x^2-5x+6=0`
- Expressions: `(x^3*y^2)^4/(x*y^5)`
- Use `^` for exponents, and parentheses for grouping: `3(x-2)=12`
"""
    )

raw1 = st.text_input("Problem #1", value="2x+3=11")
raw2 = st.text_input("Problem #2", value="-x+7=2")
raw3 = st.text_input("Problem #3", value="3x-5=16")

if st.button("Generate Foldable PDF", type="primary"):
    try:
        eq1 = problem_and_steps(raw1)
        eq2 = problem_and_steps(raw2)
        eq3 = problem_and_steps(raw3)

        build_foldable("foldable_output.pdf", eq1, eq2, eq3)

        with open("foldable_output.pdf", "rb") as f:
            st.download_button(
                "Download foldable_output.pdf",
                f,
                file_name="foldable_output.pdf",
                mime="application/pdf",
            )

        st.success("Your foldable is ready!")

    except Exception as e:
        st.error("Error reading one of the inputs.")
        st.exception(e)

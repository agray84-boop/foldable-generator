import streamlit as st
import os
import re
import tempfile
from typing import Tuple, List, Optional

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
    convert_xor,
)

def _normalize_math(s: str) -> str:
    s = s.strip()
    s = s.replace("−", "-").replace("·", "*")
    return s

def _parse(s: str) -> sp.Expr:
    s = _normalize_math(s)
    return parse_expr(s, transformations=_TRANSFORMS, evaluate=True)


# ----------------------------
# LaTeX helpers
# ----------------------------

def ltx(x) -> str:
    return sp.latex(x)

def eq_ltx(L, R) -> str:
    return sp.latex(sp.Eq(L, R))

def step(label: str, math: str) -> str:
    return rf"\text{{{label}:}}\; & {math}"

def final_box(math: str) -> str:
    return rf"\boxed{{{math}}}"

def notebook_aligned(lines: List[str]) -> str:
    body = r" \\ ".join(lines)
    return r"\begin{aligned}" + body + r"\end{aligned}"


# ----------------------------
# Notebook-style steps
# ----------------------------

def linear_steps(left, right, x):
    expr = sp.simplify(left - right)
    poly = sp.Poly(expr, x)
    a = poly.coeff_monomial(x)
    b = poly.coeff_monomial(1)

    steps = [
        step("Given", eq_ltx(left, right)),
        step("Step 1", eq_ltx(expr, 0)),
        step("Step 2", eq_ltx(a * x, -b)),
    ]

    sol = sp.simplify(-b / a)
    steps.append(step("Step 3", eq_ltx(x, sol)))
    steps.append(step("Answer", final_box(rf"x={ltx(sol)}")))

    return steps


def expression_steps(expr):
    steps = [step("Given", ltx(expr))]
    simp = sp.simplify(expr)
    steps.append(step("Step 1", ltx(simp)))
    steps.append(step("Answer", final_box(ltx(simp))))
    return steps


def problem_and_steps(raw: str, var="x"):
    x = sp.Symbol(var)
    norm = _normalize_math(raw)

    if "=" in norm:
        left_str, right_str = norm.split("=", 1)
        left = _parse(left_str)
        right = _parse(right_str)
        return eq_ltx(left, right), linear_steps(left, right, x)

    expr = _parse(norm)
    return ltx(expr), expression_steps(expr)


# ----------------------------
# LaTeX rendering to PNG
# ----------------------------

def render_png(math_latex, font_size=16, dpi=300):
    if not math_latex.startswith("$"):
        math_latex = f"${math_latex}$"

    fig = plt.figure()
    fig.patch.set_alpha(0)
    t = fig.text(0, 0, math_latex, fontsize=font_size)
    fig.canvas.draw()
    bbox = t.get_window_extent()
    w, h = bbox.width / dpi, bbox.height / dpi
    plt.close(fig)

    fig = plt.figure(figsize=(w + 0.3, h + 0.3), dpi=dpi)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, math_latex, fontsize=font_size, ha="center", va="center")

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_centered(c, math_latex, cx, cy, w, h):
    img_path = render_png(math_latex)
    img = ImageReader(img_path)
    iw, ih = img.getSize()
    scale = min(w / iw, h / ih)
    c.drawImage(img, cx - (iw * scale) / 2, cy - (ih * scale) / 2,
                iw * scale, ih * scale, mask="auto")
    os.remove(img_path)


# ----------------------------
# PDF foldable layout
# ----------------------------

def build_foldable(out_path, eq1, eq2, eq3):
    page_w, page_h = letter
    c = Canvas(out_path, pagesize=letter)

    xs = [0, 2.75 * inch, 5.5 * inch, 8.25 * inch, page_w]

    def folds():
        c.setDash(6, 6)
        for x in xs[1:-1]:
            c.line(x, 0.5 * inch, x, page_h - 0.5 * inch)
        c.setDash()

    folds()

    problems = [eq2, eq3, None, eq1]

    for i in range(4):
        if problems[i]:
            draw_centered(
                c,
                problems[i][0] if i != 1 else notebook_aligned(problems[i][1]),
                (xs[i] + xs[i+1]) / 2,
                page_h / 2,
                xs[i+1] - xs[i] - 0.4 * inch,
                page_h - 2 * inch,
            )

    c.showPage()

    c.translate(page_w, page_h)
    c.rotate(180)
    folds()

    backs = [eq1, None, eq3, eq2]

    for i in range(4):
        if backs[i]:
            draw_centered(
                c,
                notebook_aligned(backs[i][1]) if i != 2 else backs[i][0],
                (xs[i] + xs[i+1]) / 2,
                page_h / 2,
                xs[i+1] - xs[i] - 0.4 * inch,
                page_h - 2 * inch,
            )

    c.showPage()
    c.save()


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="Math Foldable Generator", layout="centered")

st.title("Printable Math Foldable Generator")
st.write("Enter **3 math problems** (equations or expressions).")

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

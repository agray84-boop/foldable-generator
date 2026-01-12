import streamlit as st
from foldable_engine import problem_and_steps, notebook_aligned, build_foldable

st.set_page_config(page_title="Math Foldable Generator", layout="centered")

st.title("Printable Math Foldable Generator")

st.write(
    "Enter **3 math problems** (equations or expressions), then click **Generate**.\n\n"
    "**Examples:**\n"
    "- `2x + 3 = 11`\n"
    "- `x^2 - 5x + 6 = 0`\n"
    "- `(x^3*y^2)^4/(x*y^5)`"
)

with st.expander("Input Tips"):
    st.markdown(
        """
- Use **^** for exponents → `x^2`
- Use parentheses → `3(x-2)=12`
- Include `=` for equations
- Expressions without `=` will be simplified
"""
    )

raw1 = st.text_input("Problem #1", value="2x + 3 = 11")
raw2 = st.text_input("Problem #2", value="x^2 - 5x + 6 = 0")
raw3 = st.text_input("Problem #3", value="(x^3*y^2)^4/(x*y^5)")

if st.button("Generate Foldable PDF", type="primary"):
    try:
        p1, steps1 = problem_and_steps(raw1, var="x")
        p2, steps2 = problem_and_steps(raw2, var="x")
        p3, steps3 = problem_and_steps(raw3, var="x")

        block1 = notebook_aligned(steps1)
        block2 = notebook_aligned(steps2)
        block3 = notebook_aligned(steps3)

        output_path = "foldable_output.pdf"
        build_foldable(
            output_path,
            (p1, block1, steps1),
            (p2, block2, steps2),
            (p3, block3, steps3),
        )

        with open(output_path, "rb") as f:
            st.download_button(
                "Download foldable_output.pdf",
                data=f,
                file_name="foldable_output.pdf",
                mime="application/pdf",
            )

        st.success("Your foldable is ready!")

    except Exception as e:
        st.error(
            "One of the inputs could not be read. "
            "Try adding parentheses or using ^ for exponents."
        )

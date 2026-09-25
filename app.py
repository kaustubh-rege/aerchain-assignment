"""
Aerchain assignment - UI.

A thin Streamlit front end over everything the pipeline already built:
- Tab 1: the price comparison table, cheapest-per-row highlighted, with
  every flag (currency conversion, unit conversion, footnote discounts,
  missing quotes) visible rather than hidden behind a clean-looking number.
- Tab 2: vendor quality pass/fail, with the actual reason for every fail -
  this is what makes the Tab 3 agent's filtering trustworthy rather than
  a black box.
- Tab 3: the natural-language agent from agent.py, with the pandas code
  it ran shown alongside every answer.

Run:
    pip install streamlit
    export ANTHROPIC_API_KEY=sk-...
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import build_dataframes, run_agent

BASE = Path(__file__).parent

st.set_page_config(page_title="Aerchain RFx Comparison", layout="wide")


@st.cache_data
def load_data():
    comparison_path = BASE / "comparison_table.json"
    master_path = BASE / "master_items.json"
    extracted_dir = BASE / "extracted"

    missing = [p.name for p in [comparison_path, master_path] if not p.exists()]
    if missing or not extracted_dir.exists():
        return None  # caller checks for this and shows setup instructions

    comparison_table = json.loads(comparison_path.read_text())
    master_items = json.loads(master_path.read_text())
    extractions = {f.stem: json.loads(f.read_text()) for f in extracted_dir.glob("*.json")}
    return comparison_table, master_items, extractions


def highlight_cheapest(row: pd.Series) -> list[str]:
    """Green-highlight the lowest price in each SKU row; leaves NaN cells
    (vendor didn't quote that SKU) unstyled rather than treating them as 0."""
    numeric = row.apply(lambda x: x if pd.notna(x) else float("inf"))
    if numeric.min() == float("inf"):
        return [""] * len(row)
    min_idx = list(row.index).index(numeric.idxmin())
    return ["background-color: #d4f7d4" if i == min_idx else "" for i in range(len(row))]


def main():
    st.title("Kill the Quote Spreadsheet")
    st.caption("RFx-2026-CORR-014 - corrugated packaging, 30 SKUs, 5 vendors")

    data = load_data()
    if data is None:
        st.error(
            "Missing pipeline output. Before running this UI, from this same folder run:\n\n"
            "1. `python extraction_pipeline.py`  (needs the 5 Vendor* files + master_items.json)\n"
            "2. `python normalize.py`  (needs the `extracted/` folder from step 1)\n\n"
            "Then re-run `streamlit run app.py`."
        )
        st.stop()

    comparison_table, master_items, extractions = data
    dataframes = build_dataframes(comparison_table, extractions, master_items)
    price_df = dataframes["price_df"]
    flags_df = dataframes["flags_df"]
    quality_df = dataframes["quality_df"]

    tab1, tab2, tab3 = st.tabs(["Comparison Table", "Vendor Quality", "Ask a Question"])

    with tab1:
        st.subheader("Price comparison (INR per piece)")
        st.caption("Cheapest quoted price per SKU highlighted in green. Blank = not quoted by that vendor.")
        styled = price_df.style.apply(highlight_cheapest, axis=1).format(precision=2, na_rep="—")
        st.dataframe(styled, use_container_width=True)

        n_flagged = int((flags_df != "").sum().sum())
        with st.expander(f"Show all flags ({n_flagged} flagged cells - unit/currency conversions, footnote conditions, missing quotes, illegible fields)"):
            st.dataframe(flags_df, use_container_width=True)

    with tab2:
        st.subheader("Quality questionnaire - pass/fail")
        st.caption("Pass = ISO 9001 confirmed AND defect-rate guarantee ≤ 2.0%. Missing/illegible data fails closed, not open.")
        display_cols = [c for c in ["quality_pass", "iso_9001", "defect_rate_pct", "fail_reasons"]
                         if c in quality_df.columns]
        st.dataframe(quality_df[display_cols], use_container_width=True)

    with tab3:
        st.subheader("Ask the analyst agent")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("ANTHROPIC_API_KEY is not set in this terminal's environment. Set it and "
                      "restart `streamlit run app.py`.")
            st.stop()

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input(
                "Your question",
                placeholder='e.g. "cheapest per line among vendors who cleared the quality questionnaire?"'
            )
            submitted = st.form_submit_button("Ask")

        if submitted and question.strip():
            with st.spinner("Running analysis..."):
                result = run_agent(client, question, dataframes, verbose=False)
            st.session_state.chat_history.append((question, result))

        for q, result in reversed(st.session_state.chat_history):
            st.markdown(f"**Q: {q}**")
            if result.get("is_partial"):
                st.warning("Best-effort answer - the agent hit its turn limit before fully "
                           "resolving this one. Treat it as provisional.")
            st.markdown(result["answer"])
            with st.expander("Show the code the agent ran (audit trail)"):
                for step in result["trace"]:
                    if step.get("purpose"):
                        st.caption(step["purpose"])
                    st.code(step["code"], language="python")
                    st.text(step["result"])
            st.divider()


if __name__ == "__main__":
    main()

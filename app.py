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


def build_display_df(dataframes: dict, eligible_vendors: list[str]) -> pd.DataFrame:
    """One row per SKU: description, the cheapest price AMONG eligible_vendors
    only (so the quality-only toggle actually changes the winner, not just
    the color), every vendor's raw price as its own sortable column, and a
    flag count so a buyer can sort straight to the riskiest lines."""
    price_df = dataframes["price_df"]
    flags_df = dataframes["flags_df"]
    items_df = dataframes["items_df"]
    vendors = list(price_df.columns)

    rows = []
    for sku in price_df.index:
        row = price_df.loc[sku]
        eligible = row[eligible_vendors].dropna()
        cheapest_vendor = eligible.idxmin() if not eligible.empty else "— none quoted —"
        cheapest_price = eligible.min() if not eligible.empty else None
        flag_count = int((flags_df.loc[sku] != "").sum())
        entry = {
            "SKU": sku,
            "Description": items_df.loc[sku, "description"],
            "Cheapest Vendor": cheapest_vendor,
            "Cheapest Price": cheapest_price,
        }
        for v in vendors:
            entry[v] = row[v]
        entry["Flags"] = flag_count
        rows.append(entry)
    return pd.DataFrame(rows).set_index("SKU")


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
    confidence_df = dataframes["confidence_df"]
    quality_df = dataframes["quality_df"]
    items_df = dataframes["items_df"]
    all_vendors = list(price_df.columns)

    tab1, tab2, tab3 = st.tabs(["Comparison Table", "Vendor Quality", "Ask a Question"])

    with tab1:
        st.subheader("Price comparison (INR per piece)")

        with st.sidebar:
            st.header("Filters")
            vendor_filter = st.multiselect("Vendor columns to show", all_vendors, default=all_vendors)
            quality_only = st.checkbox(
                "Only count quality-passed vendors as 'cheapest'",
                help="When checked, 'Cheapest Vendor'/'Cheapest Price' below ignore any vendor "
                     "that failed the quality questionnaire - even if their raw price is lower."
            )
            search = st.text_input("Search SKU or description")
            only_flagged = st.checkbox("Only show SKUs with at least one flag")

        eligible_vendors = (
            quality_df[quality_df["quality_pass"]].index.tolist() if quality_only else all_vendors
        )
        eligible_vendors = eligible_vendors or all_vendors  # guard against an empty filter result

        display_df = build_display_df(dataframes, eligible_vendors)

        if search.strip():
            s = search.strip().lower()
            mask = display_df.index.str.lower().str.contains(s) | \
                   display_df["Description"].str.lower().str.contains(s)
            display_df = display_df[mask]
        if only_flagged:
            display_df = display_df[display_df["Flags"] > 0]

        shown_cols = ["Description", "Cheapest Vendor", "Cheapest Price"] + vendor_filter + ["Flags"]
        st.caption(f"{len(display_df)} of {len(price_df)} SKUs shown. Click a row for full detail. "
                   f"Click any column header to sort.")

        event = st.dataframe(
            display_df[shown_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Cheapest Price": st.column_config.NumberColumn(format="₹%.2f"),
                **{v: st.column_config.NumberColumn(format="₹%.2f") for v in vendor_filter},
            },
            key="price_table",
        )

        selected_rows = event.selection.rows if event and event.selection else []
        if selected_rows:
            sku = display_df.index[selected_rows[0]]
            st.divider()
            st.subheader(f"{sku} — {items_df.loc[sku, 'description']}")
            detail_rows = []
            for v in all_vendors:
                detail_rows.append({
                    "Vendor": v,
                    "Price (INR/pc)": price_df.loc[sku, v],
                    "Confidence": confidence_df.loc[sku, v],
                    "Quality Pass": quality_df.loc[v, "quality_pass"] if v in quality_df.index else None,
                    "Flags": flags_df.loc[sku, v] or "—",
                })
            st.dataframe(pd.DataFrame(detail_rows).set_index("Vendor"), use_container_width=True)
        else:
            st.info("No row selected yet - click any SKU above to see every vendor's price, "
                    "confidence, and flags for that line side by side.")

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

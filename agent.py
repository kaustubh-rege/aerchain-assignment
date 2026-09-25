"""
Aerchain assignment - Stage 4: natural-language Q&A agent.

Loads comparison_table.json (Stage 3 output) plus the per-vendor
questionnaire answers (Stage 2 output) into pandas DataFrames, then lets
Claude answer free-text buyer questions by actually WRITING AND RUNNING
pandas code against them - via a tool call, in a loop, same as a human
analyst would. The code it ran is shown alongside the answer so a buyer
can audit it, not just trust it.

Run interactively:
    export ANTHROPIC_API_KEY=sk-...
    python agent.py
"""

import io
import json
import os
import re
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------
# DATA LAYER - build DataFrames from Stage 2 + Stage 3 output
# --------------------------------------------------------------------------

def parse_pct(s: str | None) -> float | None:
    """'1.5%' -> 1.5, 'illegible' / None / 'not stated' -> None. Never guesses."""
    if not s:
        return None
    m = re.search(r"[\d.]+", s)
    return float(m.group()) if m else None


def compute_quality_pass(questionnaire: dict) -> dict:
    """Pass criteria (documented assumption, state this in your one-pager):
    ISO 9001 certified AND defect-rate guarantee <= 2.0%. Missing or
    illegible data means FAIL, not benefit-of-the-doubt PASS - a buyer
    acting on this shouldn't have unverifiable claims read as compliant."""
    iso_text = (questionnaire.get("iso_9001") or "").lower()
    iso_ok = iso_text.startswith("yes")
    defect_pct = parse_pct(questionnaire.get("defect_rate_pct"))
    defect_ok = defect_pct is not None and defect_pct <= 2.0

    reasons = []
    if not iso_ok:
        reasons.append("ISO 9001 not confirmed")
    if defect_pct is None:
        reasons.append("defect rate not legible/stated")
    elif not defect_ok:
        reasons.append(f"defect rate {defect_pct}% exceeds 2.0% threshold")

    return {
        "iso_9001_ok": iso_ok,
        "defect_rate_pct": defect_pct,
        "quality_pass": iso_ok and defect_ok,
        "fail_reasons": reasons,
    }


def build_dataframes(comparison_table: list[dict], extractions: dict, master_items: list[dict]):
    # items_df - the RFx master list
    items_df = pd.DataFrame(master_items).set_index("sku")

    # price_df / confidence_df / flags_df - wide, one row per SKU, one column per vendor
    vendor_names = set()
    for row in comparison_table:
        vendor_names.update(k for k in row.keys() if k != "sku")
    vendor_names = sorted(vendor_names)

    price_rows, conf_rows, flag_rows = {}, {}, {}
    for row in comparison_table:
        sku = row["sku"]
        price_rows[sku] = {v: row.get(v, {}).get("price_inr_per_piece") for v in vendor_names}
        conf_rows[sku] = {v: row.get(v, {}).get("confidence") for v in vendor_names}
        flag_rows[sku] = {v: "; ".join(row.get(v, {}).get("flags", []) or []) for v in vendor_names}

    price_df = pd.DataFrame.from_dict(price_rows, orient="index")[vendor_names]
    price_df.index.name = "sku"
    confidence_df = pd.DataFrame.from_dict(conf_rows, orient="index")[vendor_names]
    flags_df = pd.DataFrame.from_dict(flag_rows, orient="index")[vendor_names]

    # quality_df - one row per vendor: questionnaire answers + computed pass/fail
    quality_rows = {}
    for vendor_key, data in extractions.items():
        vendor_name = data.get("vendor_name_as_stated", vendor_key)
        q = data.get("questionnaire", {}) or {}
        result = compute_quality_pass(q)
        quality_rows[vendor_name] = {**q, **result}
    quality_df = pd.DataFrame.from_dict(quality_rows, orient="index")

    return {
        "price_df": price_df,           # SKU x vendor -> INR per piece (None if not quoted)
        "confidence_df": confidence_df,  # SKU x vendor -> 'high'/'medium'/'low'/None
        "flags_df": flags_df,            # SKU x vendor -> semicolon-joined flag text
        "quality_df": quality_df,        # vendor -> questionnaire answers + quality_pass bool
        "items_df": items_df,            # sku -> description, ply, gsm, dims, annual_qty_pcs
    }


# --------------------------------------------------------------------------
# SANDBOXED EXECUTION - the tool the agent actually calls
# --------------------------------------------------------------------------

SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "round": round,
    "sorted": sorted, "list": list, "dict": dict, "range": range,
    "enumerate": enumerate, "zip": zip, "print": print, "abs": abs,
    "__import__": lambda name, *a, **k: (
        pd if name == "pandas" else (_ for _ in ()).throw(
            ImportError(f"import of '{name}' is not allowed in this sandbox - "
                        f"pandas is already available as `pd`, you don't need to import it"))
    ),
}


def build_scope(dataframes: dict) -> dict:
    """One persistent namespace per question. Passed to run_analysis_code on
    every tool call within that question, so variables the model defines in
    one turn (e.g. res_df) are still there in the next turn - a REPL/notebook
    session, not a fresh interpreter per call. This is what a real analyst's
    workflow looks like, and what the model naturally assumes."""
    return {"pd": pd, "__builtins__": SAFE_BUILTINS, **dataframes}


def run_analysis_code(code: str, scope: dict) -> str:
    """Executes buyer-analysis code the model wrote, against a scope dict
    that persists across calls within one question (see build_scope) - no
    file, network, or system access. This is a scoping sandbox for a
    trusted single-user prototype, not a hardened multi-tenant sandbox -
    worth saying explicitly in your one-pager rather than implying more
    security than this has."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, scope)
        output = buf.getvalue().strip()
        return output if output else "(code ran with no printed output - use print() to show results)"
    except Exception as e:
        return f"ERROR running code: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# AGENT LOOP
# --------------------------------------------------------------------------

ANALYSIS_TOOL = {
    "name": "run_pandas_analysis",
    "description": "Write and run Python/pandas code against the procurement data to answer "
                    "the buyer's question. Always compute real numbers this way - never state "
                    "a figure you haven't actually calculated with this tool.",
    "input_schema": {
        "type": "object",
        "required": ["code", "purpose"],
        "properties": {
            "purpose": {"type": "string", "description": "One line: what this code checks"},
            "code": {"type": "string", "description": "Python code. Use print() to show "
                                                          "results - only printed output is "
                                                          "returned to you."}
        }
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a procurement analyst agent helping a buyer compare vendor
quotes for RFx-2026-CORR-014. You have these pandas DataFrames available via the
run_pandas_analysis tool:

- price_df: index=SKU, columns=vendor name, values=INR per piece (None if not quoted by that vendor)
- confidence_df: same shape, values='high'/'medium'/'low'/None - the extraction confidence for that cell
- flags_df: same shape, values=text describing any issue with that cell (unit conversion, currency, illegible, etc), empty string if none
- quality_df: index=vendor name, columns include iso_9001, defect_rate_pct, quality_pass (bool), fail_reasons
- items_df: index=SKU, columns=description, ply, gsm, dims_mm, annual_qty_pcs

Rules:
- Always use the tool to compute actual numbers. Never state a price, total, or comparison from
  memory or estimation - only from code you just ran.
- Variables you define in one tool call PERSIST into your next tool call within this
  conversation (like a notebook session) - you do not need to redefine price_df or rebuild a
  result you already built. pandas is already available as `pd` - do not import it.
- Before showing a "cheapest" or "best" answer, mention if any of the winning cells have
  low confidence or a non-empty flag - the buyer needs to know if a number is shaky.
- If a SKU is missing from a vendor's data (NaN), say so explicitly - don't skip it silently.
- Give your final answer in plain language once you have the numbers. You can call the tool
  more than once if you need to check something before finalizing.

Buyer's question: {question}"""


def run_agent(client, question: str, dataframes: dict, max_turns: int = 10, verbose: bool = True) -> dict:
    """Returns {"answer": str, "trace": [...], "is_partial": bool}. If the
    agent doesn't reach a final text-only answer within max_turns, it makes
    ONE more call with tool_choice='none' to force a best-effort summary of
    whatever it learned - so you always get something usable instead of a
    dead end, with is_partial=True marking it as such."""
    system = SYSTEM_PROMPT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": question}]
    trace = []
    scope = build_scope(dataframes)  # persists across all tool calls for THIS question

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=2048,
            system=system,
            tools=[ANALYSIS_TOOL],
            tool_choice={"type": "auto"},
            messages=messages,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_parts = [b.text for b in response.content if b.type == "text"]

        if not tool_uses:
            # Model gave a final answer with no further tool calls.
            return {"answer": "\n".join(text_parts).strip(), "trace": trace, "is_partial": False}

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tu in tool_uses:
            result = run_analysis_code(tu.input["code"], scope)
            step = {"purpose": tu.input.get("purpose", ""), "code": tu.input["code"], "result": result}
            trace.append(step)
            if verbose:
                print(f"\n[turn {turn+1}] {step['purpose']}\n{step['code']}\n-> {step['result']}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    # Ran out of turns - force a plain-text best-effort answer instead of
    # giving up silently, using only what's already in the conversation.
    messages.append({"role": "user", "content": "You're out of tool-call turns. Give your best "
                      "answer in plain text now based only on what you've already computed above "
                      "- do not call the tool again. If you genuinely can't answer, say exactly "
                      "what's missing."})
    final = client.messages.create(
        model="claude-sonnet-5", max_tokens=1024, system=system,
        tools=[ANALYSIS_TOOL], tool_choice={"type": "none"}, messages=messages,
    )
    text = "\n".join(b.text for b in final.content if b.type == "text").strip()
    return {"answer": text or "(agent could not produce an answer even after forcing one - "
                                "this question likely needs a different approach)",
            "trace": trace, "is_partial": True}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    import anthropic

    base = Path(__file__).parent
    comparison_table = json.loads((base / "comparison_table.json").read_text())
    master_items = json.loads((base / "master_items.json").read_text())
    extractions = {f.stem: json.loads(f.read_text()) for f in (base / "extracted").glob("*.json")}

    dataframes = build_dataframes(comparison_table, extractions, master_items)
    print(f"Loaded {len(dataframes['items_df'])} SKUs x {len(dataframes['price_df'].columns)} vendors.\n")
    print("Quality pass/fail:")
    print(dataframes["quality_df"][["quality_pass", "fail_reasons"]].to_string())
    print()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("Ask a question (or 'quit'):")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        # verbose=True inside run_agent already prints each step live as it
        # happens, so you see progress DURING a slow/stuck question instead
        # of only after it finishes.
        result = run_agent(client, q, dataframes)
        if result["is_partial"]:
            print("\n(NOTE: this answer is best-effort - the agent hit its turn limit "
                  "before fully resolving the question. Treat it as provisional.)")
        print("\n--- answer ---")
        print(result["answer"])


if __name__ == "__main__":
    main()

"""
Aerchain assignment - Stage 3: normalization.

Takes the raw extraction JSON produced by extraction_pipeline.py (one file
per vendor, units/currency preserved exactly as the vendor stated them) and
produces ONE comparison table: every vendor, every RFx SKU, one currency
(INR), one unit (per piece), with every adjustment applied and disclosed.

This step is deliberately NOT an LLM call for the arithmetic - currency and
unit conversion is done in plain Python so it's auditable line by line. The
one exception is resolving a vendor's free-text pricing RULES (e.g. Vendor
E's "3-ply above 400x300x250mm - Rs 9/pc") into implied per-SKU prices,
which genuinely needs language understanding - that's a second, narrowly
scoped LLM call, kept separate from the deterministic math so you can tell
which numbers came from code and which came from model reasoning.

Run after extraction_pipeline.py has populated extracted/*.json:
    python normalize.py
"""

import json
import os
from pathlib import Path

# Documented assumption - the one number in this whole pipeline that isn't
# extracted from anywhere. State it plainly wherever the table is shown.
USD_TO_INR = 83.0


def load_extractions(extracted_dir: Path) -> dict:
    out = {}
    for f in sorted(extracted_dir.glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def convert_unit_price_to_inr_per_piece(item: dict, currency_stated: str) -> tuple[float | None, list[str]]:
    """Returns (price_in_inr_per_piece, notes). Never silently guesses -
    if a unit can't be converted with what's known, price comes back None
    and the reason goes in notes so it shows up as a flag, not a blank."""
    notes = []
    price = item.get("unit_price")
    if price is None:
        return None, ["no price stated"]

    # Currency - currency_stated is now a strict enum (INR/USD/EUR/GBP/OTHER/NOT_SPECIFIED)
    # from the extraction schema, so this is an exact match, not string-guessing.
    currency = (currency_stated or "NOT_SPECIFIED").strip().upper()
    if currency in ("NOT_SPECIFIED", "INR"):
        price_inr = price
    elif currency == "USD":
        price_inr = price * USD_TO_INR
        notes.append(f"converted USD->INR at assumed rate {USD_TO_INR}")
    else:
        return None, [f"unhandled currency '{currency_stated}' - needs manual review"]

    # Unit of measure
    uom = (item.get("unit_of_measure") or "").strip().lower()
    if uom in ("", "per piece", "per pc", "unspecified", "per unit"):
        price_per_piece = price_inr
        if uom in ("", "unspecified"):
            notes.append("unit of measure not stated - ASSUMED per piece, verify before use")
    elif "100" in uom:
        price_per_piece = price_inr / 100.0
        notes.append(f"converted '{item.get('unit_of_measure')}' -> per piece (divided by 100)")
    else:
        return None, [f"unhandled unit '{item.get('unit_of_measure')}' - needs manual review"]

    return round(price_per_piece, 4), notes


def apply_document_adjustments(price: float, adjustments: list[dict]) -> tuple[float, list[str]]:
    notes = []
    net = price
    for adj in adjustments:
        pct = adj.get("adjustment_pct")
        if pct is None:
            notes.append(f"unquantified adjustment noted but NOT applied: {adj.get('description')}")
            continue
        net = net * (1 + pct / 100.0)
        notes.append(f"applied {pct:+.1f}% ({adj.get('description')})")
    return round(net, 4), notes


def resolve_pricing_rules_with_llm(client, vendor_name: str, pricing_rules: list[str],
                                     master_items: list[dict]) -> list[dict]:
    """The one place normalization uses an LLM: turning free-text category
    rules into implied per-SKU prices. Kept separate from the deterministic
    math above and clearly labeled in the output so a buyer can see which
    numbers are arithmetic and which are model interpretation."""
    tool = {
        "name": "record_rule_resolution",
        "description": "Apply the vendor's stated pricing rules to each RFx SKU.",
        "input_schema": {
            "type": "object",
            "required": ["resolved_prices"],
            "properties": {
                "resolved_prices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["rfx_sku", "implied_price_inr_per_piece", "rule_applied",
                                     "confidence"],
                        "properties": {
                            "rfx_sku": {"type": "string"},
                            "implied_price_inr_per_piece": {"type": ["number", "null"]},
                            "rule_applied": {"type": "string",
                                              "description": "Which rule text this came from"},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                        }
                    }
                }
            }
        }
    }
    system = ("You resolve a vendor's category-level pricing rules into an implied price for "
              "each specific SKU in the buyer's RFx. Only use the rules given - don't invent "
              "numbers. If a rule is genuinely ambiguous for a SKU, still give your best-effort "
              "price but mark confidence low and say why in rule_applied.")
    user_text = (f"Vendor: {vendor_name}\n\nPricing rules as stated:\n" +
                 "\n".join(f"- {r}" for r in pricing_rules) +
                 "\n\nRFx SKUs to price:\n" + json.dumps(master_items, indent=2))

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": "record_rule_resolution"},
        messages=[{"role": "user", "content": user_text}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input["resolved_prices"]
    return []


def build_comparison_table(extractions: dict, master_items: list[dict], client=None) -> list[dict]:
    sku_list = [it["sku"] for it in master_items]
    rows = {sku: {"sku": sku} for sku in sku_list}

    for vendor_key, data in extractions.items():
        vendor_name = data.get("vendor_name_as_stated", vendor_key)
        currency = data.get("currency_stated", "INR")
        adjustments = data.get("document_level_adjustments", [])

        matched_skus = set()
        for item in data.get("line_items", []):
            sku = item.get("matched_rfx_sku")
            price_pp, unit_notes = convert_unit_price_to_inr_per_piece(item, currency)
            if price_pp is not None and adjustments:
                price_pp, adj_notes = apply_document_adjustments(price_pp, adjustments)
                unit_notes += adj_notes

            cell = {
                "price_inr_per_piece": price_pp,
                "confidence": item.get("confidence"),
                "flags": item.get("flags", []) + unit_notes,
                "source": "line_item",
            }
            if sku and sku in rows:
                rows[sku][vendor_name] = cell
                matched_skus.add(sku)

        # Rule-based vendors (e.g. Vendor E) - only resolved if a client was passed in
        rules = data.get("pricing_rules", [])
        if rules and client is not None:
            resolved = resolve_pricing_rules_with_llm(client, vendor_name, rules, master_items)
            for r in resolved:
                sku = r["rfx_sku"]
                if sku in rows and sku not in matched_skus:
                    rows[sku][vendor_name] = {
                        "price_inr_per_piece": r["implied_price_inr_per_piece"],
                        "confidence": r["confidence"],
                        "flags": [f"derived from rule: {r['rule_applied']}"],
                        "source": "resolved_rule",
                    }
                    matched_skus.add(sku)
        elif rules:
            for sku in sku_list:
                if sku not in matched_skus:
                    rows[sku].setdefault(vendor_name, {
                        "price_inr_per_piece": None,
                        "confidence": None,
                        "flags": ["vendor gave pricing rules, not resolved - run with API client"],
                        "source": "unresolved_rule",
                    })

        # Anything the RFx asked for that this vendor never priced at all
        for sku in sku_list:
            if sku not in matched_skus and vendor_name not in rows[sku]:
                rows[sku][vendor_name] = {
                    "price_inr_per_piece": None,
                    "confidence": None,
                    "flags": ["NOT QUOTED by this vendor"],
                    "source": "missing",
                }

    return list(rows.values())


def main():
    base = Path(__file__).parent
    extracted_dir = base / "extracted"
    master_items = json.loads((base / "master_items.json").read_text())

    if not extracted_dir.exists() or not any(extracted_dir.glob("*.json")):
        print("No files in extracted/ yet - run extraction_pipeline.py first.")
        return

    extractions = load_extractions(extracted_dir)

    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        print("No ANTHROPIC_API_KEY set - rule-based vendors (e.g. category pricing) "
              "will be left unresolved with a flag instead of an LLM-derived price.")

    table = build_comparison_table(extractions, master_items, client=client)

    out_path = base / "comparison_table.json"
    out_path.write_text(json.dumps(table, indent=2))
    print(f"Wrote {out_path} ({len(table)} SKU rows)")


if __name__ == "__main__":
    main()

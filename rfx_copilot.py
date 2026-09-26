"""
Aerchain assignment - Stage 0: RFx drafting co-pilot.

A buyer talks an RFx into existence conversationally - category, line
items, quality questionnaire, commercial terms - instead of filling out a
form. Once there's enough to work with, the model calls finalize_rfx with
the complete structured spec (same shape as master_items.json, so it's
compatible with every later stage). Sending is stubbed (the brief
explicitly allows faking the SMTP server) - it writes to outbox.json
rather than actually emailing anyone.

IMPORTANT SCOPING NOTE: sending a freshly-drafted RFx here does NOT produce
matching vendor responses - the 5 Vendor* files in this repo are pre-
fabricated demo data for a DIFFERENT, already-sent RFx (used to demonstrate
extraction/normalization/Q&A), since real vendors take up to 9 days to
reply. This co-pilot is genuinely functional end-to-end for drafting +
sending; the round-trip back to vendor replies is where the demo
necessarily switches to canned data. Said explicitly here and in the UI
rather than left implicit.

Run interactively:
    export ANTHROPIC_API_KEY=sk-...
    python rfx_copilot.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path


FINALIZE_TOOL = {
    "name": "finalize_rfx",
    "description": "Record the complete, buyer-confirmed RFx once enough detail has been "
                    "gathered conversationally. Call this only when you have at least one "
                    "fully-specified line item and the buyer has indicated they're ready - "
                    "not on the first message.",
    "input_schema": {
        "type": "object",
        "required": ["scope", "line_items", "questionnaire_questions", "commercial_terms"],
        "properties": {
            "scope": {
                "type": "string",
                "description": "One or two sentences describing what this RFx covers."
            },
            "line_items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["sku", "description", "ply", "gsm", "dims_mm", "annual_qty_pcs"],
                    "properties": {
                        "sku": {"type": "string", "description": "Assign sequential codes yourself, e.g. PKG-001, PKG-002..."},
                        "description": {"type": "string"},
                        "ply": {"type": "integer"},
                        "gsm": {"type": "integer"},
                        "dims_mm": {"type": "string", "description": "Format LxWxH, e.g. '300x200x150'"},
                        "annual_qty_pcs": {"type": "integer"},
                    }
                }
            },
            "questionnaire_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Quality/compliance questions to send to every vendor, e.g. "
                                "'ISO 9001:2015 certified?', 'Defect rate guarantee?'"
            },
            "commercial_terms": {
                "type": "object",
                "properties": {
                    "payment_terms_requested": {"type": "string"},
                    "delivery_terms_requested": {"type": "string"},
                    "quote_validity_days": {"type": "integer"},
                }
            }
        }
    }
}

SYSTEM_PROMPT = """You are an RFx drafting co-pilot helping a procurement buyer create a new
RFx by talking it through with you, instead of filling out a form.

Have a natural back-and-forth - don't demand everything in one message. Cover, over as many
turns as it takes:
- What category/product this is for, and roughly how many distinct line items (sizes/variants)
- Key specs per item (dimensions, material, ply, whatever's relevant to the category)
- Annual quantity per item
- What you should ask every vendor in a quality/compliance questionnaire
- Commercial terms the buyer wants (payment terms, delivery expectations, quote validity period)

If the buyer is vague on a detail, propose a sensible default and confirm it rather than
blocking on it. Once you have at least one fully-specified line item and the buyer indicates
they're ready to finalize, call finalize_rfx - and ALSO include a short plain-text summary of
what you're finalizing in the same turn, so the buyer can review it before it's sent."""


def process_user_message(client, messages: list, user_text: str) -> dict:
    """One turn of the drafting conversation. Returns {"reply_text": str,
    "finalized_rfx": dict|None, "messages": list} - messages is the updated
    history, already including the required tool_result if a tool was
    called, so the conversation stays valid for a follow-up turn."""
    messages.append({"role": "user", "content": user_text})

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        tools=[FINALIZE_TOOL],
        tool_choice={"type": "auto"},
        messages=messages,
    )

    if response.stop_reason == "max_tokens":
        # Same failure mode as the extraction pipeline hit earlier: the tool call got cut
        # off mid-generation (likely while writing out several line items), so whatever
        # came back is missing required fields. Don't let that reach send_rfx() as if it
        # were a complete RFx - drop this turn's assistant message and surface a clear error.
        raise RuntimeError(
            "The RFx draft got cut off before finishing (stop_reason=max_tokens) - likely "
            "too many line items in one go. Try asking for fewer items at once, or ask the "
            "co-pilot to finalize in smaller batches."
        )

    messages.append({"role": "assistant", "content": response.content})

    text_parts = [b.text for b in response.content if b.type == "text"]
    finalize_blocks = [b for b in response.content if b.type == "tool_use" and b.name == "finalize_rfx"]

    result = {"reply_text": "\n".join(text_parts).strip(), "finalized_rfx": None, "messages": messages}

    if finalize_blocks:
        tb = finalize_blocks[0]
        if "line_items" not in tb.input or not tb.input["line_items"]:
            # Belt-and-suspenders: even without hitting max_tokens, don't trust an
            # incomplete finalize call. Still send the required tool_result so the
            # conversation stays valid, but never hand this back as finalized_rfx.
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tb.id,
                 "content": "This RFx is missing line_items - please ask the buyer for at "
                            "least one complete item before finalizing again.",
                 "is_error": True}
            ]})
            result["reply_text"] = (result["reply_text"] or
                "(The draft was incomplete - missing line items. Let's go over that again.)")
            return result
        result["finalized_rfx"] = tb.input
        # Every tool_use MUST be followed by a tool_result before the conversation can
        # continue (e.g. if the buyer wants to revise something after finalizing).
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tb.id,
             "content": "RFx captured and shown to the buyer for review."}
        ]})

    return result


def send_rfx(rfx: dict, vendor_emails: list[str], outbox_path: Path) -> dict:
    """Stubbed send - the brief explicitly allows faking the SMTP server.
    Writes a record to outbox.json instead of actually emailing anyone."""
    body_lines = [f"Subject: RFx - {rfx['scope']}", "", rfx["scope"], "", "Line items:"]
    for li in rfx["line_items"]:
        body_lines.append(f"  {li['sku']}: {li['description']} "
                          f"({li['dims_mm']}mm, {li['ply']}-ply, {li['gsm']}gsm) "
                          f"- qty {li['annual_qty_pcs']}/yr")
    body_lines.append("\nQuality questionnaire:")
    for q in rfx.get("questionnaire_questions", []):
        body_lines.append(f"  - {q}")
    ct = rfx.get("commercial_terms", {})
    if ct:
        body_lines.append(f"\nRequested terms: {ct}")
    body = "\n".join(body_lines)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "to": vendor_emails,
        "body": body,
        "rfx": rfx,
    }

    outbox = json.loads(outbox_path.read_text()) if outbox_path.exists() else []
    outbox.append(record)
    outbox_path.write_text(json.dumps(outbox, indent=2))

    return record


def rfx_to_master_items(rfx: dict) -> list[dict]:
    """Converts a finalized RFx into the same shape as master_items.json,
    so a freshly-drafted RFx is structurally compatible with every later
    pipeline stage (even though, per the scoping note above, it won't have
    real vendor responses waiting for it)."""
    return [
        {"sku": li["sku"], "description": li["description"], "ply": li["ply"],
         "gsm": li["gsm"], "dims_mm": li["dims_mm"], "annual_qty_pcs": li["annual_qty_pcs"]}
        for li in rfx["line_items"]
    ]


def main():
    import anthropic
    import os

    base = Path(__file__).parent
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = []

    print("RFx drafting co-pilot. Describe what you need (category, specs, quantities). "
          "Type 'quit' to exit.\n")

    finalized = None
    while True:
        user_text = input("> ").strip()
        if user_text.lower() in ("quit", "exit"):
            break

        result = process_user_message(client, messages, user_text)
        print(f"\n{result['reply_text']}\n")

        if result["finalized_rfx"]:
            finalized = result["finalized_rfx"]
            answer = input("Send this RFx to vendors now? (y/n): ").strip().lower()
            if answer == "y":
                emails_raw = input("Vendor emails (comma-separated, or Enter for defaults): ").strip()
                vendor_emails = [e.strip() for e in emails_raw.split(",")] if emails_raw else [
                    "vendor-a@example.com", "vendor-b@example.com", "vendor-c@example.com"
                ]
                record = send_rfx(finalized, vendor_emails, base / "outbox.json")
                (base / "drafted_master_items.json").write_text(
                    json.dumps(rfx_to_master_items(finalized), indent=2))
                print(f"\n(stubbed) Sent to {vendor_emails} - logged in outbox.json")
                print("Also wrote drafted_master_items.json (pipeline-compatible line items).\n")
                print("NOTE: this repo's Comparison/Q&A tabs use pre-fabricated vendor "
                      "responses for a DIFFERENT demo RFx - a freshly sent RFx here has no "
                      "real vendor replies waiting for it.\n")


if __name__ == "__main__":
    main()

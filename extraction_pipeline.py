"""
Aerchain assignment - vendor response extraction pipeline.

Stage 1 (ingest_file) converts any supported vendor document into either
raw text or a base64 image, whichever the source format actually is.
Stage 2 (extract_vendor_document) sends that content to Claude with a
FORCED tool call, so the model cannot return anything except schema-valid
JSON - no regex-scraping of free text.

Run:
    export ANTHROPIC_API_KEY=sk-...
    pip install anthropic pdfplumber python-docx pandas openpyxl pillow
    python extraction_pipeline.py

Requires the RFx master item list (master_items.json) as extraction
context, so the model can match a vendor's own SKU refs / prose mentions
back to canonical RFx SKUs.
"""

import base64
import json
import mimetypes
import os
from pathlib import Path

import pandas as pd
import pdfplumber
from docx import Document as DocxDocument

# anthropic is only needed for the actual API call (Stage 2) - imported
# lazily so Stage 1 (ingestion) can be tested without it installed.


# --------------------------------------------------------------------------
# STAGE 1 - INGESTION: turn any file format into {"kind": "text"|"image", ...}
# --------------------------------------------------------------------------

def ingest_file(path: str) -> dict:
    ext = Path(path).suffix.lower()

    if ext == ".xlsx":
        return _ingest_xlsx(path)
    if ext == ".pdf":
        return _ingest_pdf(path)
    if ext == ".docx":
        return _ingest_docx(path)
    if ext in (".txt", ".eml"):
        return _ingest_text(path)
    if ext in (".jpg", ".jpeg", ".png"):
        return _ingest_image(path)

    raise ValueError(f"Unsupported vendor file type: {ext}")


def _ingest_xlsx(path: str) -> dict:
    # Dump every sheet as markdown-ish text. Don't try to be clever about
    # layout here - let the LLM read it the way a human would open the
    # sheet and read it top to bottom. This also means a WELL-FORMED sheet
    # goes through the *same* extraction path as a messy PDF, which is the
    # point: no special-cased fast path that quietly skips the AI loop.
    xls = pd.ExcelFile(path)
    chunks = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        chunks.append(f"--- Sheet: {sheet} ---\n{df.to_csv(index=False, header=False)}")
    return {"kind": "text", "content": "\n\n".join(chunks)}


def _ingest_pdf(path: str) -> dict:
    # Real text extraction (this PDF has a genuine text layer, not a scan).
    # A scanned/flattened PDF would instead need page rasterization + the
    # image path below - worth handling if your real vendors send those.
    texts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            texts.append(f"--- Page {i} ---\n{page.extract_text() or '[no extractable text]'}")
    return {"kind": "text", "content": "\n\n".join(texts)}


def _ingest_docx(path: str) -> dict:
    doc = DocxDocument(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    # Tables in a .docx (if any) need separate handling - not needed for
    # this dataset's prose-only vendor, but flagged here as a known gap.
    return {"kind": "text", "content": "\n".join(paras)}


def _ingest_text(path: str) -> dict:
    return {"kind": "text", "content": Path(path).read_text()}


def _ingest_image(path: str) -> dict:
    mime, _ = mimetypes.guess_type(path)
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
    return {"kind": "image", "media_type": mime or "image/jpeg", "content": data}


# --------------------------------------------------------------------------
# STAGE 2 - EXTRACTION: forced tool-use call, schema-valid JSON out
# --------------------------------------------------------------------------

EXTRACTION_TOOL = {
    "name": "record_vendor_extraction",
    "description": "Record structured data extracted from one vendor's RFx response.",
    "input_schema": {
        "type": "object",
        "required": ["vendor_name_as_stated", "currency_stated", "line_items", "pricing_rules",
                     "questionnaire", "extraction_notes"],
        "properties": {
            "vendor_name_as_stated": {"type": "string"},
            "currency_stated": {
                "type": "string",
                "description": "Currency exactly as stated in the source (e.g. 'INR', 'USD', "
                                "'not specified'). Never assume INR by default."
            },
            "line_items": {
                "type": "array",
                "description": "Only items with an explicit, extractable price. Do not invent "
                                "a price for an item the vendor didn't quote.",
                "items": {
                    "type": "object",
                    "required": ["description_as_quoted", "matched_rfx_sku", "unit_price",
                                 "unit_of_measure", "confidence", "flags", "source_snippet"],
                    "properties": {
                        "description_as_quoted": {"type": "string"},
                        "matched_rfx_sku": {
                            "type": ["string", "null"],
                            "description": "Best-guess RFx SKU (e.g. 'CB-014') this line "
                                            "corresponds to, using the supplied master item "
                                            "list. Null if you cannot confidently match it."
                        },
                        "unit_price": {"type": ["number", "null"]},
                        "unit_of_measure": {
                            "type": "string",
                            "description": "Exactly as implied by the source: 'per piece', "
                                            "'per 100 pieces', 'per kg', 'unspecified', etc. "
                                            "Never silently normalize this yourself."
                        },
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "e.g. 'illegible', 'unit_ambiguous', "
                                           "'price_in_footnote', 'currency_not_inr'"
                        },
                        "source_snippet": {
                            "type": "string",
                            "description": "The verbatim text (or a description of the image "
                                            "region) that supports this extraction, for audit."
                        }
                    }
                }
            },
            "pricing_rules": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Use this INSTEAD of / IN ADDITION TO line_items when the "
                                "vendor gives category-level rules rather than a per-SKU price "
                                "list (e.g. 'add Rs 3.50/pc over 3-ply rate for 5-ply'). Quote "
                                "each rule close to verbatim. Do not try to resolve these into "
                                "per-SKU prices yourself - a separate step does that."
            },
            "questionnaire": {
                "type": "object",
                "properties": {
                    "iso_9001": {"type": ["string", "null"]},
                    "moq": {"type": ["string", "null"]},
                    "lead_time_days": {"type": ["string", "null"]},
                    "recycled_content_pct": {"type": ["string", "null"]},
                    "defect_rate_pct": {"type": ["string", "null"]},
                    "payment_terms": {"type": ["string", "null"]},
                    "freight_terms": {"type": ["string", "null"]}
                }
            },
            "extraction_notes": {
                "type": "string",
                "description": "Plain-language summary of anything ambiguous, illegible, "
                                "missing, or inconsistent in this document - this is what "
                                "gets shown to the buyer as a trust signal."
            }
        }
    }
}

SYSTEM_PROMPT = """You are a meticulous procurement data-extraction agent. You will be shown
one vendor's response to an RFx, plus the buyer's master line-item list for reference.

Rules:
- Extract only what is actually stated. Never invent a price, a SKU match, or a certification
  the vendor didn't provide.
- Preserve the vendor's own units and currency exactly as stated - do not convert INR/USD or
  per-piece/per-100 yourself. That normalization happens in a later step, deterministically.
- If a document has a discount, surcharge, or condition in a footnote, endnote, or small print,
  you must find it and include it - buried conditions are exactly what this system exists to
  catch.
- If a price applies to a size/ply bracket rather than a specific SKU, record it under
  pricing_rules, not by guessing which SKUs it covers.
- Flag anything illegible, ambiguous, or missing rather than silently working around it.
- Call the record_vendor_extraction tool exactly once with your findings."""


def extract_vendor_document(client, file_path: str, master_items: list[dict]) -> dict:
    ingested = ingest_file(file_path)
    master_items_text = json.dumps(master_items, indent=2)

    if ingested["kind"] == "text":
        user_content = [{
            "type": "text",
            "text": (f"Buyer's master RFx line-item list (for SKU matching):\n"
                      f"{master_items_text}\n\n"
                      f"--- VENDOR DOCUMENT ({Path(file_path).name}) ---\n"
                      f"{ingested['content']}")
        }]
    else:  # image
        user_content = [
            {"type": "text",
             "text": (f"Buyer's master RFx line-item list (for SKU matching):\n"
                       f"{master_items_text}\n\n"
                       f"The following image is a vendor's rate card "
                       f"({Path(file_path).name}). Read it carefully, including any small "
                       f"print - it may be a photo taken at an angle.")},
            {"type": "image",
             "source": {"type": "base64", "media_type": ingested["media_type"],
                        "data": ingested["content"]}}
        ]

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_vendor_extraction"},
        messages=[{"role": "user", "content": user_content}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_vendor_extraction":
            return block.input

    raise RuntimeError("Model did not return the expected tool call.")


# --------------------------------------------------------------------------
# MAIN - run extraction across all vendor files in a folder
# --------------------------------------------------------------------------

def main():
    import anthropic  # deferred so Stage 1 can be unit-tested without this installed

    data_dir = Path(__file__).parent
    master_items = json.loads((data_dir / "master_items.json").read_text())

    vendor_files = [
        "VendorA_Alpha_Packaging_Quote.xlsx",
        "VendorB_Bharat_Corrugators_Quote.pdf",
        "VendorC_Continental_Packaging_Quote.docx",
        "VendorD_Swift_RateCard_Photo.jpg",
        "VendorE_Rapid_Cartons_Email.txt",
    ]

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    out_dir = data_dir / "extracted"
    out_dir.mkdir(exist_ok=True)

    for fname in vendor_files:
        fpath = data_dir / fname
        print(f"Extracting {fname} ...")
        result = extract_vendor_document(client, str(fpath), master_items)
        out_path = out_dir / f"{Path(fname).stem}.json"
        out_path.write_text(json.dumps(result, indent=2))

        n_items = len(result.get("line_items", []))
        n_rules = len(result.get("pricing_rules", []))
        n_flags = sum(len(li.get("flags", [])) for li in result.get("line_items", []))
        print(f"  -> {n_items} line items, {n_rules} pricing rules, {n_flags} flags raised")


if __name__ == "__main__":
    main()

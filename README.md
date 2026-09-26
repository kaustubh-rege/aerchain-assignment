# Kill the Quote Spreadsheet

Aerchain Product Management take-home — Kaustubh Rege
RFx-2026-CORR-014: corrugated packaging, 30 SKUs, 5 vendors, 5 different response formats.

**Build notes / what I decided and left out:** see [`one_pager.pdf`](./one_pager.pdf)
**Recorded walkthrough:** https://www.loom.com/share/8e09b4c18984422884554bf0a90f2371 

## What's in this repo

| File | Stage | What it does |
|---|---|---|
| `rfx_copilot.py` | 0. Draft & Send | Conversational RFx drafting - buyer describes what they need, a forced Claude tool-call finalizes scope/line items/questionnaire/terms once ready; sending is stubbed (logs to `outbox.json`, no real SMTP) |
| `master_items.json` | — | The demo RFx already sent: 30 SKUs with dimensions, ply, GSM, annual quantity |
| `Vendor*` (xlsx/pdf/docx/jpg/txt) | — | 5 fabricated vendor responses to that demo RFx, each stress-testing a different real-world edge case |
| `extraction_pipeline.py` | 1. Ingest + Extract | Converts any vendor file to text/image, then a forced Claude tool-call extracts structured line items, pricing rules, document-level adjustments, and questionnaire answers |
| `normalize.py` | 2. Normalize | Deterministic Python: converts every vendor to one currency/unit (INR per piece), applies footnote-style adjustments, resolves category-level pricing rules to implied per-SKU prices |
| `agent.py` | 3. Q&A | Natural-language agent that writes and runs real pandas code against the normalized table, in a persistent sandboxed session, showing its code alongside every answer |
| `app.py` | 4. UI | Streamlit front end with 4 tabs: Draft & Send RFx, the comparison table with row drill-down, vendor quality pass/fail, and the Q&A agent as a chat interface |

**Important scoping note:** the RFx drafting co-pilot (tab 0) is genuinely live end-to-end - draft a new RFx, it gets finalized and "sent." But the Comparison/Q&A tabs are wired to `master_items.json` and the 5 `Vendor*` files, which simulate a *different*, already-sent RFx - since real vendors take up to 9 days to reply, there's no way to get live responses to whatever you draft in the demo itself. This is a deliberate scoping decision, not a hidden gap - see `one_pager.pdf`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install anthropic pdfplumber python-docx pandas openpyxl pillow streamlit
export ANTHROPIC_API_KEY=sk-...  # Windows: $env:ANTHROPIC_API_KEY="sk-..."
```

## Run order

```bash
python extraction_pipeline.py   # writes extracted/*.json
python normalize.py             # writes comparison_table.json
streamlit run app.py            # opens the UI in your browser
```

`agent.py` can also be run standalone (`python agent.py`) for a terminal-only Q&A session without the UI.

## Try the RFx co-pilot (tab 0)

Describe something like: *"I need corrugated mailer boxes for a Q1 apparel launch, three sizes, roughly 5000-8000 units each, ISO certification required."* The co-pilot will ask follow-up questions, then finalize a structured RFx you can review and "send."

## Try asking the agent (tab 3)

- *"What if we split it, cheapest per line, but only among vendors who cleared the quality questionnaire?"*
- *"Which SKUs does no quality-passed vendor quote?"*
- *"How much would we save splitting the award vs. sole-sourcing the cheapest single vendor?"*

## Design decisions worth knowing before the demo

- **Currency/unit conversion and the quality pass/fail bar are plain Python, not LLM output** — auditable line by line. See `one_pager.pdf` for why.
- **The sandbox is scoped, not hardened** — no file/network access, but built for a trusted single user, not multi-tenant.
- **USD→INR uses a documented constant (83)**, not a live rate — see `USD_TO_INR` in `normalize.py`.

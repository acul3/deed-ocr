# Simplified Deed OCR Workflow

This implements a simplified workflow for deed OCR processing:

**PDF Input → PDF-Image → OCR Engine (Gemini AI) → Structured JSON**

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Set Up API Key

Either set your Google AI API key as an environment variable:
```bash
export GOOGLE_AI_API_KEY="your_api_key_here"
```

Or pass it directly to the function.

### 3. Run the Workflow

#### Option 1: Use the test script
```bash
python test_simple_workflow.py
```

#### Option 2: Use in your own code
```python
from pathlib import Path
from deed_ocr import process_deed_pdf_simple

# Process a PDF
result = process_deed_pdf_simple(
    pdf_path=Path("your_deed.pdf"),
    api_key="your_api_key",  # Optional if using env var
    output_dir=Path("output"),  # Optional
    dpi=300  # Image quality
)

# Access results
print(f"Pages processed: {result.total_pages}")
print(f"Legal descriptions found: {len(result.all_legal_descriptions)}")
print(f"Full text: {result.combined_full_text}")
```

## Output Structure

The workflow produces a `SimplifiedDeedResult` with:

- `source_pdf`: Original PDF filename
- `total_pages`: Number of pages processed
- `pages_data`: List of OCR results for each page
- `combined_full_text`: All text combined from all pages
- `all_legal_descriptions`: All legal descriptions found
- `all_details`: Combined deed details

Each page result contains:
```json
{
  "page_number": 1,
  "full_text": "Complete OCR text...",
  "legal_description_block": ["Legal description text..."],
  "details": {
    "grantor": "...",
    "grantee": "...",
    "date": "...",
    "etc": "..."
  }
}
```

## Files Created

When you specify an `output_dir`, the workflow saves:

- `{pdf_name}_result.json` - Complete structured result
- `{pdf_name}_full_text.txt` - Combined text from all pages
- `pages/page_N.json` - Individual page results

## Configuration

- `dpi`: Image resolution (default: 300) - higher = better quality but slower
- `api_key`: Google AI API key
- `output_dir`: Where to save results (optional)

## Error Handling

The workflow continues processing even if individual pages fail. Failed pages will have an `error` field in their results. 
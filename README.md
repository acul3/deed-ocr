# Deed OCR - Phase 1

Extract legal descriptions from scanned deed PDFs using OCR and pattern matching with optional LLM enhancement.

## Features

- OCR via Google Cloud Vision or Amazon Textract
- Multiple extraction methods:
  - Regex patterns for known formats
  - Hybrid approach with NLP (spaCy)
  - LLM-enhanced extraction (Gemini Pro, ChatGPT, Qwen3)
- Layout detection for improved accuracy
- Outputs page-level TXT files and structured JSON
- Confidence scoring for all extractions

## Setup

1. Install dependencies:
```bash
pip install poetry
poetry install  # Basic installation
# OR
poetry install --extras llm  # With LLM support
poetry install --extras all  # All features
```

2. Configure API credentials:
```bash
cp env.example .env
# Edit .env with your API keys
```

Required credentials:
- **OCR**: Google Cloud Vision or AWS Textract
- **LLM** (optional): Gemini, OpenAI, or local Qwen3 path

## Usage

Basic usage:
```bash
python -m deed_ocr --pdf input.pdf --out_dir out/
```

With LLM enhancement:
```bash
python -m deed_ocr --pdf input.pdf --out_dir out/ --extraction-method llm --llm-provider gemini
```

Options:
- `--ocr-provider`: Choose 'google' or 'textract'
- `--extraction-method`: 'regex', 'hybrid', or 'llm'
- `--llm-provider`: 'gemini', 'chatgpt', or 'qwen3'
- `--use-layout/--no-layout`: Enable layout detection
- `--verbose`: Debug logging

## Output

- `out/page_N.txt`: OCR text per page
- `out/result.json`: Legal descriptions with confidence scores

## Testing

```bash
poetry run pytest
```
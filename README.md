# Deed OCR - AI-Powered Legal Document Processing

Extract legal descriptions and structured data from deed PDFs using AI OCR engines.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install Python packages
pip install google-genai google-cloud-vision pymupdf pillow pydantic python-dotenv click
```

### 2. Get API Keys
- **Gemini AI**: Get key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Google Vision** (optional): Set up [Google Cloud Vision API](https://cloud.google.com/vision/docs/setup) (use Authentication with service accounts, download json)

### 3. Configure Credentials
Create a `.env` file in your project root:
```bash
# Required for all engines
GOOGLE_AI_API_KEY=your_gemini_api_key_here

# Required only for vision-gemini engine (your sevice-accoun json path)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### 4. Run
```bash
# Process single PDF
python -m deed_ocr -i deed.pdf -o ./results

# Process folder of PDFs
python -m deed_ocr -i ./pdf_folder -o ./results --recursive
```

## 🔧 OCR Engines

### Gemini (Direct Image-to-Structure)
```bash
# Fast processing using Gemini AI directly on images
python -m deed_ocr -i deed.pdf -o ./results --ocr-engine gemini

# Use a specific Gemini model
python -m deed_ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05

# High-accuracy mode (better results, slower, uses more tokens)
python -m deed_ocr -i deed.pdf -o ./results --ocr-engine gemini --high-accuracy
```

### Vision + Gemini (Two-Step Process)
```bash
# More accurate: Google Vision extracts text, Gemini structures it
python -m deed_ocr -i deed.pdf -o ./results --ocr-engine vision-gemini
```

## 📖 Usage Examples

```bash
# Single PDF
python -m deed_ocr -i deed.pdf -o ./results

# Batch processing
python -m deed_ocr -i ./pdf_folder -o ./results --recursive

# Separate folders for each PDF
python -m deed_ocr -i ./pdf_folder -o ./results --separate-folders

# Use specific Gemini model
python -m deed_ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05

# High-accuracy mode for better OCR results
python -m deed_ocr -i deed.pdf -o ./results --high-accuracy

# Continue on errors + high quality
python -m deed_ocr -i ./pdf_folder -o ./results --continue-on-error --dpi 600

# Verbose output for debugging
python -m deed_ocr -i deed.pdf -o ./results --verbose
```

## 🛠️ Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input (-i)` | PDF file or folder path | *Required* |
| `--output (-o)` | Output directory | *Required* |
| `--ocr-engine` | `gemini` or `vision-gemini` | `gemini` |
| `--model` | Gemini model to use | `gemini-2.5-flash-preview-05-20` |
| `--recursive (-r)` | Process subdirectories | `False` |
| `--separate-folders` | Separate folder per PDF | `False` |
| `--continue-on-error` | Continue on errors | `False` |
| `--dpi` | Image resolution (200-600) | `300` |
| `--high-accuracy` | Better results (slower, more tokens) | `False` |
| `--verbose (-v)` | Debug logging | `False` |

## 🎯 High-Accuracy Mode

The `--high-accuracy` flag enables deeper AI processing for better extraction results:

**Default Mode (Fast):**
- Uses Gemini AI with thinking budget limitation
- Faster processing and lower token usage
- Good for most standard documents

**High-Accuracy Mode (Slow but Better):**
- Removes thinking budget limitations, allowing deeper AI analysis
- More accurate extraction of complex legal descriptions
- Higher token usage and processing time
- Recommended for difficult or critical documents

```bash
# Standard processing
python -m deed_ocr -i complex_deed.pdf -o ./results

# High-accuracy processing for better results
python -m deed_ocr -i complex_deed.pdf -o ./results --high-accuracy
```

**When to use high-accuracy:**
- Complex handwritten documents
- Poor scan quality
- Critical legal documents requiring precision
- When standard mode produces incomplete results

## 🤖 Available Gemini Models

You can specify different Gemini models using the `--model` option:

**Available Models:**
- `gemini-2.5-flash-preview-05-20` (Default) - Fast and efficient
- `gemini-2.5-pro-preview-06-05` - More powerful, better accuracy
- Other Gemini models as they become available

```bash
# Use the default fast model
python -m deed_ocr -i deed.pdf -o ./results

# Use the more powerful pro model
python -m deed_ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05
```

**Model Not Found Error:**
If you see a "404 NOT_FOUND" error with a model name, the model may not be available or supported:
```
❌ Model Error: The model 'gemini-2.5-pro-preview-06-05' was not found or is not supported.
💡 Try using a different model like 'gemini-2.5-flash-preview-05-20' with --model option
```

Try using the default model or check [Google AI Studio](https://aistudio.google.com/) for available models.

## 📁 Output Structure

Each PDF creates a folder with:

**Gemini Engine:**
```
{pdf_name}/
├── full_text.txt                    # Combined text from all pages
├── full_pdf_analysis.json           # Complete structured data
├── final_result.json                # Final merged result
└── pages/                           # Individual page results
```

**Vision + Gemini Engine:**
```
{pdf_name}_vision_gemini/
├── full_text.txt                    # Text extracted by Google Vision
├── gemini_structured.json           # Structured data from Gemini AI
├── complete_result.json             # Complete workflow result
└── pages_text/                      # Individual page text files
```

### Extracted Data Structure
```json
{
  "legal_description_block": ["Legal property descriptions..."],
  "details": {
    "document_type": "Deed",
    "document_subtype": "Warranty Deed",
    "parties": {
      "grantor": ["John Doe"],
      "grantee": ["Jane Smith"]
    },
    "TRS": ["T2N R3W S14"],
    "deed_details": {
      "grantors_interest": "Fee simple",
      "subject_to": "Existing easements..."
    }
  }
}
```

## 🐛 Common Issues

**API key not found**
- Check your `.env` file exists and contains `GOOGLE_AI_API_KEY`
- For vision-gemini: ensure `GOOGLE_APPLICATION_CREDENTIALS` is set

**No PDF files found**
- Verify file extensions (.pdf, .PDF)
- Use `--recursive` for subdirectories

**Memory issues**
- Reduce DPI: `--dpi 200`
- Process smaller batches

**Debug mode**
```bash
python -m deed_ocr -i input.pdf -o output --verbose
```
# Deed OCR - AI-Powered Legal Document Processing

Extract legal descriptions and structured data from deed PDFs using advanced AI OCR engines.


## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- Google AI API key (Gemini)
- Google Cloud Vision API credentials (optional, for vision-gemini engine)

### 2. Installation

#### Option A: Using pip with virtual environment (Recommended)

**Linux/macOS:**
```bash
# Create and activate virtual environment
python -m venv deed-ocr-env
source deed-ocr-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
# Create and activate virtual environment
python -m venv deed-ocr-env
deed-ocr-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
# Create and activate virtual environment
python -m venv deed-ocr-env
deed-ocr-env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Poetry (Alternative)

**Linux/macOS:**
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies with Poetry
poetry install

# Activate the Poetry environment
poetry shell
```

**Windows (PowerShell - Run as Administrator):**
```powershell
# Install Poetry if you haven't already
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install dependencies with Poetry
poetry install

# Activate the Poetry environment
poetry shell
```

### 3. Configuration

Copy the example environment file and add your API keys:

**Linux/macOS:**
```bash
# Copy the example file
cp env.example .env

# Edit .env with your preferred editor
nano .env
```

**Windows (Command Prompt):**
```cmd
# Copy the example file
copy env.example .env

# Edit .env with notepad or your preferred editor
notepad .env
```

**Windows (PowerShell):**
```powershell
# Copy the example file
Copy-Item env.example .env

# Edit .env with notepad or your preferred editor
notepad .env
```

Add your API credentials to `.env`:

```bash
# Google AI API (for Gemini) - Required
GOOGLE_AI_API_KEY=your_gemini_api_key_here

# Google Cloud Vision API - Required only for vision-gemini engine
# Linux/macOS format:
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
# Windows format:
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\service-account.json
```

#### Getting API Keys

- **Gemini AI**: Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Google Vision** (optional): Set up [Google Cloud Vision API](https://cloud.google.com/vision/docs/setup) with service account authentication

### 4. Basic Usage

**Linux/macOS:**
```bash
# Process a single PDF
python -m deed_ocr -i deed.pdf -o ./results

# Process a folder of PDFs
python -m deed_ocr -i ./pdf_folder -o ./results --recursive
```

**Windows:**
```cmd
# Process a single PDF
python -m deed_ocr -i deed.pdf -o .\results

# Process a folder of PDFs  
python -m deed_ocr -i .\pdf_folder -o .\results --recursive

# Alternative with forward slashes (also works on Windows)
python -m deed_ocr -i deed.pdf -o ./results
```

## 🔧 OCR Engines

### 🚀 Gemini Engine (Default)
Direct image-to-structure processing using Gemini AI - fast and efficient.

**Linux/macOS:**
```bash
# Standard processing
python -m deed_ocr -i deed.pdf -o ./results --ocr-engine gemini

# High-accuracy mode (slower, more accurate)
python -m deed_ocr -i deed.pdf -o ./results --ocr-engine gemini --high-accuracy

# Use specific Gemini model
python -m deed_ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05
```

**Windows:**
```cmd
# Standard processing
python -m deed_ocr -i deed.pdf -o .\results --ocr-engine gemini

# High-accuracy mode (slower, more accurate)
python -m deed_ocr -i deed.pdf -o .\results --ocr-engine gemini --high-accuracy

# Use specific Gemini model
python -m deed_ocr -i deed.pdf -o .\results --model gemini-2.5-pro-preview-06-05
```

### 🔍 Vision + Gemini Engine
Two-step process: Google Vision extracts text, then Gemini structures it for higher accuracy.

**Linux/macOS:**
```bash
# More accurate extraction
python -m deed_ocr -i deed.pdf -o ./results --ocr-engine vision-gemini
```

**Windows:**
```cmd
# More accurate extraction
python -m deed_ocr -i deed.pdf -o .\results --ocr-engine vision-gemini
```

## 📖 Usage Examples

### Basic Operations

**Linux/macOS:**
```bash
# Single PDF with default settings
python -m deed_ocr -i deed.pdf -o ./results

# Batch processing with recursive folder scanning
python -m deed_ocr -i ./pdf_folder -o ./results --recursive

# Create separate output folders for each PDF
python -m deed_ocr -i ./pdf_folder -o ./results --separate-folders
```

**Windows:**
```cmd
# Single PDF with default settings
python -m deed_ocr -i deed.pdf -o .\results

# Batch processing with recursive folder scanning
python -m deed_ocr -i .\pdf_folder -o .\results --recursive

# Create separate output folders for each PDF
python -m deed_ocr -i .\pdf_folder -o .\results --separate-folders
```

### Advanced Options

**Linux/macOS:**
```bash
# High-accuracy mode for better results
python -m deed_ocr -i deed.pdf -o ./results --high-accuracy

# High-quality image conversion
python -m deed_ocr -i deed.pdf -o ./results --dpi 600

# Continue processing on errors + verbose logging
python -m deed_ocr -i ./pdf_folder -o ./results --continue-on-error --verbose

# Use specific Gemini model
python -m deed_ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05
```

**Windows:**
```cmd
# High-accuracy mode for better results
python -m deed_ocr -i deed.pdf -o .\results --high-accuracy

# High-quality image conversion
python -m deed_ocr -i deed.pdf -o .\results --dpi 600

# Continue processing on errors + verbose logging
python -m deed_ocr -i .\pdf_folder -o .\results --continue-on-error --verbose

# Use specific Gemini model
python -m deed_ocr -i deed.pdf -o .\results --model gemini-2.5-pro-preview-06-05
```

## ⚙️ Configuration Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | PDF file or folder path | *Required* |
| `--output` | `-o` | Output directory | *Required* |
| `--ocr-engine` | | `gemini` or `vision-gemini` | `gemini` |
| `--model` | | Gemini model to use | `gemini-2.5-flash-preview-05-20` |
| `--recursive` | `-r` | Process subdirectories | `False` |
| `--separate-folders` | | Separate folder per PDF | `False` |
| `--continue-on-error` | | Continue processing on errors | `False` |
| `--dpi` | | Image resolution (200-600) | `300` |
| `--high-accuracy` | | Enable high-accuracy mode | `False` |
| `--verbose` | `-v` | Enable debug logging | `False` |

## 🎯 High-Accuracy Mode

Enable enhanced AI processing for better extraction results:

**Standard Mode:**
- ✅ Fast processing and lower token usage
- ✅ Good for most standard documents
- ✅ Uses thinking budget limitations for efficiency

**High-Accuracy Mode (`--high-accuracy`):**
- 🎯 Removes thinking budget limitations
- 🎯 Deeper AI analysis for complex documents
- 🎯 Better handling of handwritten or poor-quality scans
- ⚠️ Higher token usage and processing time

**Linux/macOS:**
```bash
# When to use high-accuracy mode
python -m deed_ocr -i complex_deed.pdf -o ./results --high-accuracy
```

**Windows:**
```cmd
# When to use high-accuracy mode
python -m deed_ocr -i complex_deed.pdf -o .\results --high-accuracy
```

**Use high-accuracy for:**
- Complex handwritten documents
- Poor scan quality or faded text
- Critical legal documents requiring precision
- When standard mode produces incomplete results

## 🤖 Available Gemini Models

Choose the right model for your needs:

| Model | Speed | Accuracy | Use Case |
|-------|--------|----------|----------|
| `gemini-2.5-flash-preview-05-20` | ⚡ Fast | Good | Standard processing (default) |
| `gemini-2.5-pro-preview-06-05` | 🐌 Slower | Better | Complex documents, high accuracy |

**Linux/macOS:**
```bash
# Use the powerful pro model
python -m deed_ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05
```

**Windows:**
```cmd
# Use the powerful pro model
python -m deed_ocr -i deed.pdf -o .\results --model gemini-2.5-pro-preview-06-05
```

> **Note**: If you encounter a "404 NOT_FOUND" error, the model may not be available. Check [Google AI Studio](https://aistudio.google.com/) for current model availability.

## 📁 Output Structure

### Gemini Engine Output
```
{pdf_name}/
├── full_text.txt                    # Combined text from all pages
├── full_pdf_analysis.json           # Complete structured data
├── final_result.json                # Final merged result
└── pages/                           # Individual page results
    ├── page_1.json
    ├── page_2.json
    └── ...
```

### Vision + Gemini Engine Output
```
{pdf_name}_vision_gemini/
├── full_text.txt                    # Text extracted by Google Vision
├── gemini_structured.json           # Structured data from Gemini AI
├── complete_result.json             # Complete workflow result
└── pages_text/                      # Individual page text files
    ├── page_1.txt
    ├── page_2.txt
    └── ...
```

### 📊 Extracted Data Schema

```json
{
  "legal_description_block": [
    "LOT 1, BLOCK 5, RIVERSIDE SUBDIVISION, ACCORDING TO THE PLAT THEREOF..."
  ],
  "details": {
    "document_type": "Deed",
    "document_subtype": "Warranty Deed",
    "parties": {
      "grantor": ["John Doe", "Jane Doe"],
      "grantee": ["Smith Family Trust"]
    },
    "TRS": ["T2N R3W S14"],
    "deed_details": {
      "grantors_interest": "Fee simple absolute",
      "subject_to": "Existing easements and restrictions of record",
      "consideration": "$150,000.00"
    },
    "recording_info": {
      "book": "123",
      "page": "456",
      "date": "2023-01-15"
    }
  }
}
```

## 🔧 Development

### Setting up for Development

**Linux/macOS:**
```bash
# Using Poetry (recommended for development)
poetry install --with dev

# Or using pip
pip install -r requirements.txt
# Install dev dependencies manually if needed
```

**Windows:**
```cmd
# Using Poetry (recommended for development)
poetry install --with dev

# Or using pip
pip install -r requirements.txt
# Install dev dependencies manually if needed
```

### Running Tests

**Linux/macOS:**
```bash
# With Poetry
poetry run pytest

# With pip
pytest
```

**Windows:**
```cmd
# With Poetry
poetry run pytest

# With pip
pytest
```

### Code Formatting

**Linux/macOS:**
```bash
# Format code with Black
poetry run black deed_ocr/

# Lint with Ruff
poetry run ruff check deed_ocr/
```

**Windows:**
```cmd
# Format code with Black
poetry run black deed_ocr/

# Lint with Ruff
poetry run ruff check deed_ocr/
```

## 🐛 Troubleshooting

### Common Issues

**❌ API Key Not Found**
```
Solution: Check your .env file exists and contains GOOGLE_AI_API_KEY
```

**❌ No PDF Files Found**
```
Solutions:
- Verify file extensions (.pdf, .PDF)
- Use --recursive flag for subdirectories
- Check file permissions
```

**❌ Memory Issues with Large PDFs**
```
Solutions:
- Reduce DPI: --dpi 200
- Process smaller batches
- Use separate-folders option
```

**❌ Model Not Found Error**
```
Error: 404 NOT_FOUND: The model 'model-name' was not found
Solution: Try the default model or check Google AI Studio for available models
```

### Windows-Specific Issues

**❌ PowerShell Execution Policy Error**
```
Error: cannot be loaded because running scripts is disabled on this system
Solutions:
- Run PowerShell as Administrator
- Execute: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
- Or use Command Prompt instead of PowerShell
```

**❌ Python Command Not Found (Windows)**
```
Solutions:
- Try 'py' instead of 'python': py -m deed_ocr -i deed.pdf -o .\results
- Add Python to PATH during installation
- Use full path: C:\Users\YourName\AppData\Local\Programs\Python\Python39\python.exe
```

**❌ Path Issues with Backslashes**
```
Solutions:
- Use forward slashes (works on Windows): python -m deed_ocr -i deed.pdf -o ./results
- Or use double backslashes: python -m deed_ocr -i deed.pdf -o .\\results
- Or use Windows-style paths: python -m deed_ocr -i deed.pdf -o .\results
```

**❌ Google Cloud Credentials Path (Windows)**
```
Error: Could not load credentials file
Solutions:
- Use full Windows path: GOOGLE_APPLICATION_CREDENTIALS=C:\Users\YourName\path\to\service-account.json
- Avoid spaces in path or use quotes: GOOGLE_APPLICATION_CREDENTIALS="C:\Program Files\My App\service-account.json"
- Use forward slashes: GOOGLE_APPLICATION_CREDENTIALS=C:/Users/YourName/path/to/service-account.json
```

### Debug Mode

Enable verbose logging for troubleshooting:

**Linux/macOS:**
```bash
python -m deed_ocr -i input.pdf -o output --verbose
```

**Windows:**
```cmd
python -m deed_ocr -i input.pdf -o output --verbose
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
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

# Stage-2 batch processing with combined outputs
python -m deed_ocr -i ./pdf_folder -o ./results --stage-2
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

# Stage-2 batch processing with combined outputs
python -m deed_ocr -i .\pdf_folder -o .\results --stage-2
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
| `--stage-2` | | Batch processing with combined outputs | `False` |
| `--geodatabase-path` | | Path to PLSS geodatabase for TRS validation | *Optional* |
| `--counties-json-path` | | Path to counties JSON file for TRS validation | *Optional* |
| `--disable-trs-validation` | | Disable TRS validation | `False` |
| `--verbose` | `-v` | Enable debug logging | `False` |

## 🎯 High-Accuracy Mode

Enable enhanced AI processing for significantly better extraction results:

**Standard Mode (Default):**
- ✅ Fast processing and lower token usage
- ⚠️ **Less accurate** - good for basic document extraction
- ✅ Uses thinking budget limitations for efficiency
- ⚠️ May miss complex details or produce incomplete results

**High-Accuracy Mode (`--high-accuracy`):**
- 🎯 **Significantly more accurate** extraction
- 🎯 Removes thinking budget limitations for deeper analysis
- 🎯 Better handling of handwritten or poor-quality scans
- 🎯 More complete and precise data extraction
- ⚠️ Higher token usage and processing time

**🏆 Most Accurate Setup:** Combine `--high-accuracy` with `gemini-2.5-pro-preview-06-05` model

**Linux/macOS:**
```bash
# High-accuracy mode (more accurate than standard)
python -m deed_ocr -i complex_deed.pdf -o ./results --high-accuracy

# Most accurate setup: high-accuracy + pro model
python -m deed_ocr -i complex_deed.pdf -o ./results --high-accuracy --model gemini-2.5-pro-preview-06-05
```

**Windows:**
```cmd
# High-accuracy mode (more accurate than standard)
python -m deed_ocr -i complex_deed.pdf -o .\results --high-accuracy

# Most accurate setup: high-accuracy + pro model
python -m deed_ocr -i complex_deed.pdf -o .\results --high-accuracy --model gemini-2.5-pro-preview-06-05
```

**Use high-accuracy mode for:**
- **Any document where accuracy is important** (recommended over standard mode)
- Complex handwritten documents
- Poor scan quality or faded text
- Critical legal documents requiring precision
- When standard mode produces incomplete or inaccurate results

## 🚀 Stage-2 Processing

Stage-2 processing is designed for batch operations that need combined outputs from multiple PDFs, ideal for large-scale document processing workflows.

### What Stage-2 Does

Stage-2 automatically:
1. ✅ **Processes all PDFs** in a folder with high-accuracy mode enabled
2. ✅ **Creates separate folders** for each PDF's individual results  
3. ✅ **Generates Index_Output.xlsx** - Combined data from all processed PDFs
4. ✅ **Creates doc_texts/ folder** - All full_text.txt files renamed with document identifiers
5. ✅ **Uses optimized batch processing** - Loads geodatabase once for all files

### Stage-2 Output Structure
```
results/
├── Index_Output.xlsx           ← Combined data from all PDFs
├── doc_texts/                  ← All text files renamed 
│   ├── Document1_12345.txt     ← {pdf_name}_{reception_number}.txt
│   ├── Document2_67890.txt
│   └── ...
├── Document1/                  ← Individual PDF results
│   ├── final_result.xlsx
│   ├── full_text.txt
│   └── ...
├── Document2/
│   ├── final_result.xlsx
│   ├── full_text.txt
│   └── ...
└── ...
```

### Usage

**Linux/macOS:**
```bash
# Stage-2 processing (folder input required)
python -m deed_ocr -i ./pdf_folder -o ./results --stage-2

# Stage-2 automatically enables:
# --high-accuracy    (better extraction results)
# --separate-folders (individual PDF folders)
# --ocr-engine gemini (most reliable engine)
```

**Windows:**
```cmd
# Stage-2 processing (folder input required)
python -m deed_ocr -i .\pdf_folder -o .\results --stage-2

# Stage-2 automatically enables:
# --high-accuracy    (better extraction results)
# --separate-folders (individual PDF folders)  
# --ocr-engine gemini (most reliable engine)
```

### Stage-2 Features

- **🎯 High-Accuracy Mode**: Automatically enabled for best results
- **📊 Combined Excel Output**: All extracted data merged into Index_Output.xlsx
- **📝 Text File Collection**: All full_text.txt files collected and renamed systematically
- **⚡ Optimized Performance**: Geodatabase loaded once for all files (when TRS validation enabled)
- **🔧 Automatic Configuration**: Optimal settings applied automatically

### When to Use Stage-2

Use Stage-2 processing when you need:
- **Batch processing** of multiple deed PDFs
- **Combined output** in a single Excel file for analysis
- **Systematic text file organization** for further processing
- **High-quality extraction** across all documents
- **Consistent naming** and organization of results

**Example Workflow:**
```bash
# Process 50 deed PDFs with stage-2
python -m deed_ocr -i ./deed_batch -o ./batch_results --stage-2 --verbose

# Results:
# - 50 individual folders with complete analysis
# - Index_Output.xlsx with all combined data  
# - doc_texts/ folder with 50 renamed text files
# - TRS validation for all documents (if enabled)
```

## 🤖 Available Gemini Models

Choose the right model for your needs:

| Model | Speed | Accuracy | Use Case |
|-------|--------|----------|----------|
| `gemini-2.5-flash-preview-05-20` | ⚡ Fast | Standard | Default processing (less accurate) |
| `gemini-2.5-pro-preview-06-05` | 🐌 Slower | **Highest** | **Most accurate results** - recommended for quality |

**Linux/macOS:**
```bash
# Use the most accurate pro model
python -m deed_ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05

# Maximum accuracy: pro model + high-accuracy mode
python -m deed_ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05 --high-accuracy
```

**Windows:**
```cmd
# Use the most accurate pro model
python -m deed_ocr -i deed.pdf -o .\results --model gemini-2.5-pro-preview-06-05

# Maximum accuracy: pro model + high-accuracy mode
python -m deed_ocr -i deed.pdf -o .\results --model gemini-2.5-pro-preview-06-05 --high-accuracy
```

> **Note**: If you encounter a "404 NOT_FOUND" error, the model may not be available. Check [Google AI Studio](https://aistudio.google.com/) for current model availability.

## 🗺️ TRS Validation

Validate extracted Township, Range, and Section details against the official PLSS (Public Land Survey System) geodatabase to ensure accuracy.

### Features
- ✅ Validates Township/Range combinations against PLSS data
- ✅ Validates Section numbers within townships
- ✅ **Handles multiple counties** (e.g., "Weld, Morgan County")
- ✅ Generates Excel reports with validation results
- ✅ Integrated into the main workflow

### Requirements
```bash
# Install GeoPandas for TRS validation (optional)
pip install geopandas>=0.14.0
```

### Setup

#### 1. Download PLSS Geodatabase

Download the official PLSS geodatabase from BLM:

**Download Link:** https://gbp-blm-egis.hub.arcgis.com/datasets/283939812bc34c11bad695a1c8152faf/about

1. Click "Download" on the webpage
2. Extract the downloaded file
3. Locate the `ilmocplss.gdb` folder
4. Place it in your desired location (e.g., `/path/to/geodata/ilmocplss.gdb`)

#### 2. Configure Environment Variables

Add TRS validation paths to your `.env` file:
```bash
# Add to .env file
GEODATABASE_PATH=/path/to/geodata/ilmocplss.gdb
COUNTIES_JSON_PATH=counties_list.json  # Already included in repo root
```

**Note:** The `counties_list.json` file is already included in the repository root directory.

### Usage

**Linux/macOS:**
```bash
# TRS validation automatically enabled if GEODATABASE_PATH is set
python -m deed_ocr -i deed.pdf -o ./results

# Or specify paths directly (overrides environment variables)
python -m deed_ocr -i deed.pdf -o ./results --geodatabase-path /path/to/plss.gdb
```

**Windows:**
```cmd
# TRS validation automatically enabled if GEODATABASE_PATH is set
python -m deed_ocr -i deed.pdf -o .\results

# Or specify paths directly (overrides environment variables)
python -m deed_ocr -i deed.pdf -o .\results --geodatabase-path C:\path\to\plss.gdb
```

### Output
When TRS validation is enabled, an additional file is created:
```
{pdf_name}/
├── final_result.json
├── final_result.xlsx
├── trs_validation.xlsx    ← TRS validation results
└── ...
```

📖 **For detailed TRS validation documentation, see [docs/TRS_VALIDATION_README.md](docs/TRS_VALIDATION_README.md)**

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

### Stage-2 Processing Output
```
results/
├── Index_Output.xlsx                # Combined data from all PDFs
├── doc_texts/                       # All full_text.txt files renamed
│   ├── Document1_12345.txt          # {pdf_name}_{reception_number}.txt
│   ├── Document2_67890.txt
│   └── ...
├── Document1/                       # Individual PDF results
│   ├── final_result.json
│   ├── final_result.xlsx
│   ├── full_text.txt
│   ├── trs_validation.xlsx          # If TRS validation enabled
│   └── pages/
├── Document2/
│   ├── final_result.json
│   ├── final_result.xlsx
│   ├── full_text.txt
│   ├── trs_validation.xlsx          # If TRS validation enabled
│   └── pages/
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
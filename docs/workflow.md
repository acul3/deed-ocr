# Deed OCR Phase 1 - Development Workflow

## Architecture Overview

### Traditional Architecture
```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   PDF Input │ --> │ OCR Engine  │ --> │ Text Output  │
└─────────────┘     └─────────────┘     └──────────────┘
                           |                     |
                           v                     v
                    ┌─────────────┐     ┌──────────────┐
                    │ Page Images │     │ Legal Desc.  │
                    └─────────────┘     │  Extractor   │
                                        └──────────────┘
                                                |
                                                v
                                        ┌──────────────┐
                                        │ JSON Output  │
                                        └──────────────┘
```

### LLM-Enhanced Hybrid Architecture
```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   PDF Input │ --> │ OCR Engine  │ --> │ Text Output  │
└─────────────┘     └─────────────┘     └──────────────┘
        |                   |                     |
        v                   v                     v
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│Layout Model │     │ Page Images │     │ Regex/NLP    │
│(LayoutLM/   │     └─────────────┘     │ Extraction   │
│ Surya)      │            |             └──────────────┘
└─────────────┘            v                     |
        |           ┌─────────────┐              |
        └────────> │ LLM Engine  │ <────────────┘
                   │(Qwen3/Gemini│
                   │ /ChatGPT)   │
                   └─────────────┘
                           |
                           v
                   ┌──────────────┐
                   │ Validation & │
                   │ Confidence   │
                   │   Scoring    │
                   └──────────────┘
                           |
                           v
                   ┌──────────────┐
                   │ JSON Output  │
                   └──────────────┘
```

## Development Phases

### 1. Repository Structure
```
deed-ocr/
├── deed_ocr/           # Main package
│   ├── __init__.py
│   ├── __main__.py     # CLI entry point
│   ├── cli.py          # Click CLI implementation
│   ├── main.py         # Core processing logic
│   ├── ocr/            # OCR provider implementations
│   │   ├── google.py   # Google Cloud Vision
│   │   └── textract.py # Amazon Textract
│   ├── extractors/     # Legal description extractors
│   │   ├── regex.py    # Regex-based patterns
│   │   ├── nlp.py      # spaCy NLP extraction
│   │   └── llm.py      # LLM-based extraction
│   └── models/         # Model integrations
│       ├── layout.py   # Layout detection models
│       └── vision_llm.py # Vision LLM integration
├── tests/              # Test suite
├── docs/               # Documentation
├── pyproject.toml      # Poetry configuration
└── README.md
```

### 2. Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Individual features
- `fix/*`: Bug fixes

Workflow: feature → develop → main (via PR)

### 3. Dependency Management
- **Poetry** for Python dependencies
- Lock file ensures reproducible builds
- Separate dev dependencies for testing/linting

### 4. Secret Management
- `.env` file for local development (gitignored)
- Environment variables for API credentials:
  - `GOOGLE_APPLICATION_CREDENTIALS`
  - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
  - `OPENAI_API_KEY` (for ChatGPT)
  - `GEMINI_API_KEY` (for Gemini Pro)
  - `QWEN_MODEL_PATH` (for local Qwen3)
- Never commit credentials to repository

## Legal Description Extraction Logic

### Approach 1: Regex Patterns
Common legal description formats:
1. **Metes & Bounds**: "Beginning at... thence..."
2. **Lot & Block**: "Lot X, Block Y, [Subdivision Name]"
3. **Section/Township/Range**: "Section X, Township Y, Range Z"

### Approach 2: Statistical Model Approach
- Use Light NER model (for ex: spacy) for legal entities 
- Use dependency parsing for structure
- Score based on legal terminology density

### Approach 3: Hybrid
1. Detect page layout using layoutlm , surya or any similiar
2. Apply regex patterns first (high precision)
3. Use NLP for ambiguous cases
4. Score and rank all candidates
5. Return highest confidence matches

### Approach 4: High Precision Approach (Needs GPU or Token-based API)
1. Use vision transformers model like qwen3, phi4 (for local Deployment), gemini pro ,claude) to get layout, location from the pdf format, instruct it to give score
2. compare it to approach 2

### Approach 5: LLM-Enhanced Hybrid (Recommended for Production)
This approach combines the best of traditional methods with modern LLM capabilities:

#### Phase 1: Multi-Modal Extraction
1. **Layout Understanding**:
   - Use LayoutLM/Surya for document structure analysis
   - Identify text blocks, headers, and spatial relationships
   - Create bounding boxes for potential legal description areas

2. **Traditional Extraction**:
   - Apply regex patterns for known formats
   - Run spaCy NLP for entity recognition
   - Generate initial candidates with confidence scores

#### Phase 2: LLM Enhancement
3. **Context-Aware Processing**:
   ```python
   # Pseudo-code for LLM integration
   prompt = f"""
   Document Type: Real Estate Deed
   Task: Extract legal property description
   
   OCR Text:
   {ocr_text}
   
   Layout Information:
   {layout_boxes}
   
   Initial Candidates:
   {regex_matches}
   
   Please identify and extract the complete legal description.
   Return confidence score (0-1) and reasoning.
   """
   ```

4. **Model Selection**:
   - **Local Deployment**: Qwen3-VL (Vision-Language model)
     - Pros: Privacy, no API costs, full control
     - Cons: Requires GPU (min 24GB VRAM for 7B model)
   - **API-Based**: 
     - Gemini Pro Vision: Best for complex layouts
     - GPT-4 Vision: Highest accuracy, more expensive
     - Claude 3: Good balance of cost and performance

#### Phase 3: Validation & Consensus
5. **Cross-Validation**:
   - Compare LLM output with traditional extraction
   - Use ensemble voting for final decision
   - Flag discrepancies for human review

6. **Confidence Scoring Formula**:
   ```
   final_confidence = (
       0.3 * regex_confidence +
       0.2 * nlp_confidence +
       0.4 * llm_confidence +
       0.1 * layout_confidence
   )
   ```

### Implementation Details for LLM Integration

#### Qwen3 Local Setup
```python
# deed_ocr/models/vision_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen3Extractor:
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def extract_legal_description(self, image, text, layout):
        # Implementation details...
```

#### Gemini Pro Integration
```python
# deed_ocr/models/vision_llm.py
import google.generativeai as genai

class GeminiExtractor:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')
    
    def extract_legal_description(self, image_path, ocr_text):
        # Implementation details...
```

### Confidence Scoring
- Pattern match strength (0-1)
- Context indicators (headers, keywords)
- Length and structure validation
- Boundary detection accuracy
- LLM reasoning quality
- Cross-method agreement score

## Testing Strategy

1. **Unit Tests**: Individual function testing
2. **Integration Tests**: OCR + extraction pipeline
3. **Accuracy Tests**: Against gold standard dataset
4. **Performance Tests**: Processing speed benchmarks
5. **LLM Tests**: Prompt consistency and output validation

## Error Handling

- Graceful degradation for OCR failures
- Fallback extraction methods
- LLM timeout and rate limit handling
- Detailed logging for debugging
- User-friendly error messages

## Performance Optimization

1. **Caching Strategy**:
   - Cache OCR results per page
   - Store LLM responses for similar queries
   - Reuse layout analysis results

2. **Batch Processing**:
   - Process multiple pages in parallel
   - Batch LLM API calls when possible
   - Optimize GPU memory for local models

3. **Fallback Chain**:
   ```
   Try LLM-Enhanced → Try Hybrid → Try Regex-only → Manual Review
   ```

## Future Enhancements (Phase 2 & 3)
- Document type classification
- Reservation detection
- Chain-of-title analysis
- Multi-document correlation
- Fine-tuned domain-specific LLM
- Active learning from corrections 
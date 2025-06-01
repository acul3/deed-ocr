"""Main module for deed OCR processing."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PageText(BaseModel):
    """Model for OCR page text."""

    page_number: int = Field(description="Page number (1-indexed)")
    text: str = Field(description="Full OCR text of the page")
    confidence: Optional[float] = Field(
        default=None, description="Overall confidence score"
    )


class LegalDescription(BaseModel):
    """Model for extracted legal description."""

    text: str = Field(description="Extracted legal description text")
    page_number: int = Field(description="Page number where found")
    start_char: int = Field(description="Start character position in page text")
    end_char: int = Field(description="End character position in page text")
    confidence: float = Field(description="Confidence score of extraction")


class DeedOCRResult(BaseModel):
    """Model for complete deed OCR result."""

    source_pdf: str = Field(description="Source PDF filename")
    total_pages: int = Field(description="Total number of pages")
    legal_descriptions: List[LegalDescription] = Field(
        description="All extracted legal descriptions"
    )
    pages_text: List[PageText] = Field(description="OCR text for all pages")


def extract_text_google_vision(
    pdf_path: Path, credentials_path: Optional[Path] = None
) -> List[PageText]:
    """
    Extract text from PDF using Google Cloud Vision API.

    Args:
        pdf_path: Path to the PDF file
        credentials_path: Optional path to Google Cloud credentials JSON

    Returns:
        List of PageText objects containing OCR results
    """
    logger.info(f"Extracting text from {pdf_path} using Google Cloud Vision")
    # TODO: Implement Google Cloud Vision OCR
    # 1. Convert PDF pages to images
    # 2. Send each image to Google Vision API
    # 3. Extract text and confidence scores
    # 4. Return list of PageText objects
    raise NotImplementedError("Google Cloud Vision OCR not yet implemented")


def extract_text_textract(pdf_path: Path) -> List[PageText]:
    """
    Extract text from PDF using Amazon Textract.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of PageText objects containing OCR results
    """
    logger.info(f"Extracting text from {pdf_path} using Amazon Textract")
    # TODO: Implement Amazon Textract OCR
    # 1. Upload PDF to S3 or process locally
    # 2. Call Textract API
    # 3. Parse response and extract text
    # 4. Return list of PageText objects
    raise NotImplementedError("Amazon Textract OCR not yet implemented")


def find_legal_descriptions(
    page_texts: List[PageText],
    extraction_method: str = "hybrid",
    llm_provider: Optional[str] = None,
    use_layout: bool = True,
    pdf_path: Optional[Path] = None,
) -> List[LegalDescription]:
    """
    Find and extract legal descriptions from OCR text.

    This function uses various methods to locate legal descriptions:
    - Regex patterns for common legal description formats
    - NLP-based entity recognition
    - Layout analysis (if using advanced models)
    - LLM-enhanced extraction

    Args:
        page_texts: List of OCR page texts
        extraction_method: Method to use ('regex', 'hybrid', 'llm')
        llm_provider: LLM provider if using LLM method
        use_layout: Whether to use layout detection
        pdf_path: Path to original PDF (needed for LLM method)

    Returns:
        List of extracted legal descriptions
    """
    logger.info(f"Searching for legal descriptions using {extraction_method} method")
    
    if extraction_method == "regex":
        # Use regex-only extraction
        from deed_ocr.extractors.regex import extract_with_regex
        
        legal_descriptions = []
        for page in page_texts:
            matches = extract_with_regex(page.text)
            for match_text, start, end in matches:
                legal_descriptions.append(
                    LegalDescription(
                        text=match_text,
                        page_number=page.page_number,
                        start_char=start,
                        end_char=end,
                        confidence=0.7,  # Fixed confidence for regex
                    )
                )
        return legal_descriptions
    
    elif extraction_method == "llm":
        # Use LLM-enhanced extraction
        try:
            from deed_ocr.extractors.llm import LLMEnhancedExtractor
            
            extractor = LLMEnhancedExtractor(
                llm_provider=llm_provider or "gemini",
                use_layout=use_layout,
            )
            return extractor.extract_legal_descriptions(page_texts, pdf_path)
        except ImportError:
            logger.error("LLM dependencies not installed. Install with: poetry install --extras llm")
            # Fall back to hybrid method
            extraction_method = "hybrid"
    
    # Default hybrid method (regex + NLP)
    legal_descriptions = []

    # TODO: Implement hybrid extraction logic
    # 1. Apply regex patterns
    # 2. Apply NLP techniques (spaCy)
    # 3. Score and rank potential matches
    # 4. Filter by confidence threshold

    # Stub implementation
    for page in page_texts:
        # This is a placeholder - actual implementation would use sophisticated pattern matching
        if "legal description" in page.text.lower():
            legal_descriptions.append(
                LegalDescription(
                    text="[Legal description would be extracted here]",
                    page_number=page.page_number,
                    start_char=0,
                    end_char=100,
                    confidence=0.95,
                )
            )

    return legal_descriptions


def process_deed_pdf(
    pdf_path: Path,
    ocr_provider: str = "google",
    output_dir: Optional[Path] = None,
    extraction_method: str = "hybrid",
    llm_provider: Optional[str] = None,
    use_layout: bool = True,
) -> DeedOCRResult:
    """
    Main processing function for deed PDF files.

    Args:
        pdf_path: Path to the deed PDF file
        ocr_provider: OCR provider to use ('google' or 'textract')
        output_dir: Optional output directory for results
        extraction_method: Method for extraction ('regex', 'hybrid', 'llm')
        llm_provider: LLM provider if using LLM method
        use_layout: Whether to use layout detection

    Returns:
        DeedOCRResult object containing all extracted information
    """
    logger.info(f"Processing deed PDF: {pdf_path}")

    # Step 1: Extract text using selected OCR provider
    if ocr_provider == "google":
        page_texts = extract_text_google_vision(pdf_path)
    elif ocr_provider == "textract":
        page_texts = extract_text_textract(pdf_path)
    else:
        raise ValueError(f"Unknown OCR provider: {ocr_provider}")

    # Step 2: Find legal descriptions
    legal_descriptions = find_legal_descriptions(
        page_texts,
        extraction_method=extraction_method,
        llm_provider=llm_provider,
        use_layout=use_layout,
        pdf_path=pdf_path,
    )

    # Step 3: Create result object
    result = DeedOCRResult(
        source_pdf=pdf_path.name,
        total_pages=len(page_texts),
        legal_descriptions=legal_descriptions,
        pages_text=page_texts,
    )

    # Step 4: Save outputs if directory specified
    if output_dir:
        save_outputs(result, output_dir)

    return result


def save_outputs(result: DeedOCRResult, output_dir: Path) -> None:
    """
    Save OCR results to output directory.

    Args:
        result: DeedOCRResult object
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual page text files
    for page in result.pages_text:
        page_file = output_dir / f"page_{page.page_number}.txt"
        page_file.write_text(page.text, encoding="utf-8")

    # Save JSON output
    json_file = output_dir / "result.json"
    json_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    logger.info(f"Saved outputs to {output_dir}") 
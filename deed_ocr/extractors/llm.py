"""LLM-enhanced legal description extraction."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from deed_ocr.extractors.regex import extract_with_regex
from deed_ocr.main import LegalDescription, PageText
from deed_ocr.models.layout import (
    HybridLayoutDetector,
    identify_legal_description_regions,
)
from deed_ocr.models.vision_llm import create_llm_extractor, LLMEnsemble

logger = logging.getLogger(__name__)


class LLMEnhancedExtractor:
    """Combines traditional extraction with LLM capabilities."""

    def __init__(
        self,
        llm_provider: str = "gemini",
        use_layout: bool = True,
        use_ensemble: bool = False,
    ):
        """
        Initialize LLM-enhanced extractor.
        
        Args:
            llm_provider: LLM provider to use ('qwen3', 'gemini', 'chatgpt')
            use_layout: Whether to use layout detection
            use_ensemble: Whether to use multiple LLM providers
        """
        self.llm_provider = llm_provider
        self.use_layout = use_layout
        self.use_ensemble = use_ensemble
        
        # Initialize components
        if use_ensemble:
            # Try to initialize multiple providers
            providers = []
            for provider in ["gemini", "chatgpt"]:
                try:
                    providers.append(create_llm_extractor(provider))
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider}: {e}")
            self.llm_extractor = LLMEnsemble(providers) if providers else None
        else:
            self.llm_extractor = create_llm_extractor(llm_provider)
        
        if use_layout:
            self.layout_detector = HybridLayoutDetector()

    def extract_legal_descriptions(
        self, page_texts: List[PageText], pdf_path: Optional[Path] = None
    ) -> List[LegalDescription]:
        """
        Extract legal descriptions using hybrid LLM approach.
        
        Args:
            page_texts: OCR text for each page
            pdf_path: Optional path to original PDF for image processing
            
        Returns:
            List of extracted legal descriptions with confidence scores
        """
        all_descriptions = []
        
        for page in page_texts:
            # Step 1: Traditional extraction
            regex_matches = extract_with_regex(page.text)
            
            # Step 2: Layout detection (if enabled and PDF available)
            layout_info = None
            if self.use_layout and pdf_path:
                try:
                    # TODO: Convert PDF page to image
                    page_image_path = None  # Would be extracted from PDF
                    if page_image_path:
                        layout = self.layout_detector.detect_layout(page_image_path)
                        layout_info = {
                            "boxes": [box.dict() for box in layout.boxes],
                            "potential_regions": [
                                box.dict()
                                for box in identify_legal_description_regions(layout)
                            ],
                        }
                except Exception as e:
                    logger.error(f"Layout detection failed: {e}")
            
            # Step 3: LLM enhancement (if PDF available)
            if self.llm_extractor and pdf_path:
                try:
                    # Prepare candidates from regex
                    candidates = [match[0] for match in regex_matches[:5]]
                    
                    # TODO: Get page image path
                    page_image_path = None  # Would be extracted from PDF
                    
                    if page_image_path:
                        llm_result = self.llm_extractor.extract_legal_description(
                            page_image_path,
                            page.text,
                            layout_info,
                            candidates,
                        )
                        
                        # Create LegalDescription from LLM result
                        if llm_result.confidence > 0.5:
                            all_descriptions.append(
                                LegalDescription(
                                    text=llm_result.text,
                                    page_number=page.page_number,
                                    start_char=0,  # TODO: Find actual position
                                    end_char=len(llm_result.text),
                                    confidence=llm_result.confidence,
                                )
                            )
                            continue  # Skip traditional method if LLM succeeded
                
                except Exception as e:
                    logger.error(f"LLM extraction failed: {e}")
            
            # Step 4: Fallback to traditional extraction
            for match_text, start, end in regex_matches:
                # Calculate confidence based on pattern strength
                confidence = self._calculate_traditional_confidence(match_text, page.text)
                
                all_descriptions.append(
                    LegalDescription(
                        text=match_text,
                        page_number=page.page_number,
                        start_char=start,
                        end_char=end,
                        confidence=confidence,
                    )
                )
        
        # Step 5: Post-process and deduplicate
        return self._post_process_descriptions(all_descriptions)

    def _calculate_traditional_confidence(self, match_text: str, full_text: str) -> float:
        """Calculate confidence score for traditional extraction."""
        confidence = 0.5  # Base confidence for regex match
        
        # Boost confidence if near legal description headers
        headers = ["legal description", "property description", "described as follows"]
        for header in headers:
            if header in full_text.lower():
                header_pos = full_text.lower().find(header)
                match_pos = full_text.find(match_text)
                # If match is within 500 chars of header, boost confidence
                if abs(match_pos - header_pos) < 500:
                    confidence += 0.2
                    break
        
        # Boost confidence based on match length (longer is usually better)
        if len(match_text) > 100:
            confidence += 0.1
        if len(match_text) > 200:
            confidence += 0.1
        
        # Cap at 0.9 for traditional methods
        return min(confidence, 0.9)

    def _post_process_descriptions(
        self, descriptions: List[LegalDescription]
    ) -> List[LegalDescription]:
        """Post-process and deduplicate legal descriptions."""
        if not descriptions:
            return []
        
        # Sort by confidence
        descriptions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicates and overlapping descriptions
        final_descriptions = []
        for desc in descriptions:
            # Check if this description overlaps with any already selected
            is_duplicate = False
            for final_desc in final_descriptions:
                if (
                    desc.page_number == final_desc.page_number
                    and self._text_similarity(desc.text, final_desc.text) > 0.8
                ):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_descriptions.append(desc)
        
        return final_descriptions

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (0-1)."""
        # Simple implementation - could use more sophisticated methods
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        if text1_lower == text2_lower:
            return 1.0
        
        # Check if one contains the other
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return 0.9
        
        # Simple word overlap
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0 
"""Vision + Gemini workflow for deed OCR processing."""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from deed_ocr.ocr.vision_ocr import create_vision_ocr_engine
from deed_ocr.ocr.gemini_ocr import create_gemini_ocr_engine
from deed_ocr.utils.pdf_converter import create_pdf_converter

logger = logging.getLogger(__name__)


class VisionWorkflowError(Exception):
    """Custom exception for vision workflow errors."""
    def __init__(self, message: str, error_type: str = "unknown", original_error: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class VisionGeminiDeedResult(BaseModel):
    """Result model for Vision + Gemini deed OCR processing."""
    
    source_pdf: str = Field(description="Source PDF filename")
    total_pages: int = Field(description="Total number of pages processed")
    vision_results: Dict[str, Any] = Field(description="Text extraction results from Google Vision")
    gemini_structured_result: Dict[str, Any] = Field(description="Structured analysis from Gemini AI")
    combined_full_text: str = Field(description="Combined full text from all pages")
    # Error tracking fields
    has_errors: bool = Field(default=False, description="Whether any errors occurred during processing")
    error_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of errors encountered")
    retry_needed: bool = Field(default=False, description="Whether this document needs retry")


class VisionGeminiDeedOCRWorkflow:
    """Workflow combining Google Vision API for OCR and Gemini AI for structure extraction."""
    
    def __init__(self, 
                 vision_credentials_path: Optional[str] = None,
                 gemini_api_key: Optional[str] = None, 
                 dpi: int = 300,
                 max_retries: int = 3,
                 retry_delay: float = 5.0):
        """
        Initialize the Vision + Gemini workflow.
        
        Args:
            vision_credentials_path: Path to Google Cloud service account JSON file
            gemini_api_key: Google AI API key for Gemini
            dpi: Image resolution for PDF conversion
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        try:
            self.vision_engine = create_vision_ocr_engine(vision_credentials_path)
            self.gemini_engine = create_gemini_ocr_engine(gemini_api_key)
            self.pdf_converter = create_pdf_converter(dpi=dpi, format="PNG")
        except Exception as e:
            raise VisionWorkflowError(f"Failed to initialize workflow components: {str(e)}", "initialization", e)
        
    def _handle_api_error(self, error: Exception, operation: str) -> Dict[str, str]:
        """
        Handle and categorize API errors.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            
        Returns:
            Dictionary with error information
        """
        error_info = {
            "operation": operation,
            "error_message": str(error),
            "error_type": "unknown",
            "retry_recommended": False
        }
        
        error_str = str(error).lower()
        
        # Categorize common errors
        if "timeout" in error_str or "timed out" in error_str:
            error_info["error_type"] = "timeout"
            error_info["retry_recommended"] = True
        elif "rate limit" in error_str or "quota" in error_str:
            error_info["error_type"] = "rate_limit"
            error_info["retry_recommended"] = True
        elif "network" in error_str or "connection" in error_str or "unreachable" in error_str:
            error_info["error_type"] = "network"
            error_info["retry_recommended"] = True
        elif "authentication" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
            error_info["error_type"] = "authentication"
            error_info["retry_recommended"] = False
        elif "not found" in error_str or "404" in error_str:
            error_info["error_type"] = "not_found"
            error_info["retry_recommended"] = False
        elif "internal server error" in error_str or "500" in error_str:
            error_info["error_type"] = "server_error"
            error_info["retry_recommended"] = True
        elif "json" in error_str and "parse" in error_str:
            error_info["error_type"] = "json_parsing"
            error_info["retry_recommended"] = True
        elif "credentials" in error_str:
            error_info["error_type"] = "credentials"
            error_info["retry_recommended"] = False
        else:
            error_info["retry_recommended"] = True  # Default to retry for unknown errors
            
        return error_info
    
    def _retry_operation(self, operation_func, *args, **kwargs):
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation_func: Function to retry
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation
            
        Raises:
            The last exception if all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_info = self._handle_api_error(e, operation_func.__name__)
                
                if attempt < self.max_retries and error_info["retry_recommended"]:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed: {error_info['error_message']}. "
                                 f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    break
        
        raise last_error
        
    def process_pdf(self, pdf_path: Path, output_dir: Optional[Path] = None) -> VisionGeminiDeedResult:
        """
        Process a deed PDF through the Vision + Gemini workflow.
        
        Workflow: PDF → Images → Google Vision API (Text) → Gemini AI (Structure) → Structured JSON
        
        Args:
            pdf_path: Path to the deed PDF file
            output_dir: Optional directory to save results
            
        Returns:
            VisionGeminiDeedResult containing all extracted and structured information
        """
        logger.info(f"Starting Vision + Gemini deed OCR workflow for: {pdf_path}")
        
        error_summary = {
            "pdf_conversion_errors": [],
            "vision_ocr_errors": [],
            "gemini_processing_errors": [],
            "save_errors": []
        }
        has_errors = False
        
        try:
            # Step 1: Convert PDF to images
            logger.info("Step 1: Converting PDF to images...")
            try:
                image_bytes_list = self._retry_operation(self.pdf_converter.convert_to_bytes, pdf_path)
            except Exception as e:
                error_info = self._handle_api_error(e, "PDF conversion")
                error_summary["pdf_conversion_errors"].append(error_info)
                has_errors = True
                raise VisionWorkflowError(f"Failed to convert PDF to images: {str(e)}", "pdf_conversion", e)
            
            # Step 2: Extract text from images using Google Vision API
            logger.info("Step 2: Extracting text with Google Vision API...")
            try:
                vision_results = self._retry_operation(
                    self.vision_engine.extract_text_from_multiple_images, 
                    image_bytes_list
                )
                combined_full_text = vision_results["combined_text"]
            except Exception as e:
                error_info = self._handle_api_error(e, "Vision OCR")
                error_summary["vision_ocr_errors"].append(error_info)
                has_errors = True
                # Create fallback vision results
                vision_results = {
                    "combined_text": "",
                    "total_pages": len(image_bytes_list),
                    "page_results": [
                        {
                            "page_number": i + 1,
                            "error": str(e),
                            "text": "",
                            "character_count": 0
                        }
                        for i in range(len(image_bytes_list))
                    ],
                    "total_characters": 0,
                    "processing_status": "error",
                    "error_info": error_info
                }
                combined_full_text = ""
            
            # Step 3: Process extracted text with Gemini AI for structure
            logger.info("Step 3: Structuring text with Gemini AI...")
            try:
                if combined_full_text.strip():
                    gemini_structured_result = self._retry_operation(
                        self.gemini_engine.process_extracted_text, 
                        combined_full_text
                    )
                else:
                    raise Exception("No text available for Gemini processing")
            except Exception as e:
                error_info = self._handle_api_error(e, "Gemini text processing")
                error_summary["gemini_processing_errors"].append(error_info)
                has_errors = True
                logger.error(f"Error processing text with Gemini AI: {str(e)}")
                # Fallback structure
                gemini_structured_result = {
                    "error": str(e),
                    "error_info": error_info,
                    "processing_status": "error",
                    "legal_description_block": [],
                    "details": {}
                }
            
            # Step 4: Create combined result
            logger.info("Step 4: Creating combined result...")
            result = VisionGeminiDeedResult(
                source_pdf=pdf_path.name,
                total_pages=len(image_bytes_list),
                vision_results=vision_results,
                gemini_structured_result=gemini_structured_result,
                combined_full_text=combined_full_text,
                has_errors=has_errors,
                error_summary=error_summary,
                retry_needed=has_errors and any(
                    error.get("retry_recommended", False) 
                    for error_list in error_summary.values() 
                    if isinstance(error_list, list)
                    for error in error_list
                )
            )
            
            # Step 5: Save results if output directory specified
            if output_dir:
                try:
                    self._save_results(result, output_dir, pdf_path)
                except Exception as e:
                    error_info = self._handle_api_error(e, "Save results")
                    error_summary["save_errors"].append(error_info)
                    result.has_errors = True
                    result.error_summary = error_summary
                    logger.error(f"Error saving results: {str(e)}")
                    # Don't raise here, return result with error info
                    
            logger.info("Vision + Gemini workflow completed")
            return result
            
        except VisionWorkflowError:
            # Re-raise workflow errors as they already have proper context
            raise
        except Exception as e:
            # Handle any unexpected errors
            error_info = self._handle_api_error(e, "Workflow execution")
            logger.error(f"Unexpected workflow error: {str(e)}")
            raise VisionWorkflowError(f"Workflow failed unexpectedly: {str(e)}", "unexpected", e)
        
    def _save_results(self, result: VisionGeminiDeedResult, output_dir: Path, original_pdf_path: Path) -> None:
        """Save workflow results to output directory."""
        # Create a folder named after the PDF file (without extension)
        pdf_name = Path(result.source_pdf).stem
        pdf_output_dir = output_dir / f"{pdf_name}_vision_gemini"
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Apply post-processing to clean up results
            logger.info("Applying post-processing to results")
            cleaned_gemini_result, cleaned_full_text = self._post_process_results(
                result.gemini_structured_result, 
                result.combined_full_text
            )
            
            # Save cleaned combined full text
            text_file = pdf_output_dir / "full_text.txt"
            text_file.write_text(cleaned_full_text, encoding='utf-8')
            logger.info(f"Cleaned full text saved to {text_file}")
            
            # Save Google Vision results
            vision_file = pdf_output_dir / "vision_results.json"
            with open(vision_file, 'w', encoding='utf-8') as f:
                json.dump(result.vision_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Vision results saved to {vision_file}")
            
            # Save individual page text files (also clean these)
            pages_dir = pdf_output_dir / "pages_text"
            pages_dir.mkdir(exist_ok=True)
            
            for page_result in result.vision_results.get("page_results", []):
                page_num = page_result.get("page_number", "unknown")
                page_text_file = pages_dir / f"page_{page_num}.txt"
                page_text = page_result.get("text", "")
                
                # Clean page text as well
                cleaned_page_text = page_text
                for watermark in ["UNOFFICIAL COPY", "UNOFFICIAL COPY UNOFFICIAL COPY", "UNO"]:
                    cleaned_page_text = cleaned_page_text.replace(watermark, "")
                
                page_text_file.write_text(cleaned_page_text, encoding='utf-8')
            
            logger.info(f"Individual cleaned page texts saved to {pages_dir}")
            
            # Save cleaned Gemini structured results
            gemini_file = pdf_output_dir / "gemini_structured.json"
            with open(gemini_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_gemini_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Cleaned Gemini structured results saved to {gemini_file}")
            
            # Save error summary and retry information
            if result.has_errors:
                error_file = pdf_output_dir / "error_summary.json"
                error_data = {
                    "has_errors": result.has_errors,
                    "retry_needed": result.retry_needed,
                    "error_summary": result.error_summary,
                    "vision_processing_status": result.vision_results.get("processing_status", "unknown"),
                    "gemini_processing_status": result.gemini_structured_result.get("processing_status", "unknown")
                }
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Error summary saved to {error_file}")
            
        except Exception as e:
            logger.error(f"Error saving basic results: {str(e)}")
            raise
        
        try:
            # Save complete workflow result with cleaned data
            complete_result_file = pdf_output_dir / "complete_result.json"
            with open(complete_result_file, 'w', encoding='utf-8') as f:
                # Convert Pydantic model to dict for JSON serialization
                result_dict = result.model_dump()
                # Update with cleaned results
                result_dict["gemini_structured_result"] = cleaned_gemini_result
                result_dict["combined_full_text"] = cleaned_full_text
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Complete workflow result with cleaned data saved to {complete_result_file}")
            
            # Copy original PDF to result folder
            original_pdf_copy = pdf_output_dir / original_pdf_path.name
            shutil.copy2(original_pdf_path, original_pdf_copy)
            logger.info(f"Original PDF copied to {original_pdf_copy}")
            
        except Exception as e:
            logger.error(f"Error saving complete results or copying PDF: {str(e)}")
            # Don't raise here as main processing is done
        
        # Log token usage if available (use cleaned result)
        if "token_usage" in cleaned_gemini_result:
            token_usage = cleaned_gemini_result["token_usage"]
            logger.info(f"Gemini token usage: {token_usage}")
            
            try:
                # Save token usage separately
                token_usage_file = pdf_output_dir / "token_usage.json"
                with open(token_usage_file, 'w', encoding='utf-8') as f:
                    json.dump(token_usage, f, indent=2, ensure_ascii=False)
                logger.info(f"Token usage saved to {token_usage_file}")
                
                # Calculate and save estimated cost breakdown
                try:
                    cost_breakdown = self._calculate_estimated_cost(
                        token_usage, 
                        self.gemini_engine.model
                    )
                    
                    cost_file = pdf_output_dir / "estimated_cost_breakdown.json"
                    with open(cost_file, 'w', encoding='utf-8') as f:
                        json.dump(cost_breakdown, f, indent=2, ensure_ascii=False)
                        
                    logger.info(f"Estimated cost breakdown saved to {cost_file}")
                    logger.info(f"Estimated total cost: ${cost_breakdown['estimated_costs_usd']['total_cost']:.6f}")
                    logger.info(f"Model: {cost_breakdown['model']} ({cost_breakdown['accuracy_mode']} mode)")
                    
                except Exception as cost_error:
                    logger.error(f"Error calculating cost breakdown: {str(cost_error)}")
                    
            except Exception as e:
                logger.error(f"Error saving token usage: {str(e)}")
        elif "token_usage" in result.gemini_structured_result:
            # Fallback to original result if cleaned result doesn't have token_usage
            token_usage = result.gemini_structured_result["token_usage"]
            logger.info(f"Gemini token usage: {token_usage}")
            
            try:
                # Save token usage separately
                token_usage_file = pdf_output_dir / "token_usage.json"
                with open(token_usage_file, 'w', encoding='utf-8') as f:
                    json.dump(token_usage, f, indent=2, ensure_ascii=False)
                logger.info(f"Token usage saved to {token_usage_file}")
                
                # Calculate and save estimated cost breakdown
                try:
                    cost_breakdown = self._calculate_estimated_cost(
                        token_usage, 
                        self.gemini_engine.model
                    )
                    
                    cost_file = pdf_output_dir / "estimated_cost_breakdown.json"
                    with open(cost_file, 'w', encoding='utf-8') as f:
                        json.dump(cost_breakdown, f, indent=2, ensure_ascii=False)
                        
                    logger.info(f"Estimated cost breakdown saved to {cost_file}")
                    logger.info(f"Estimated total cost: ${cost_breakdown['estimated_costs_usd']['total_cost']:.6f}")
                    logger.info(f"Model: {cost_breakdown['model']} ({cost_breakdown['accuracy_mode']} mode)")
                    
                except Exception as cost_error:
                    logger.error(f"Error calculating cost breakdown: {str(cost_error)}")
                    
            except Exception as e:
                logger.error(f"Error saving token usage: {str(e)}")
        
        logger.info(f"All results saved to {pdf_output_dir}")


    def _remove_duplicates_from_list(self, items: List[str]) -> List[str]:
        """
        Remove duplicates from a list while preserving order.
        
        Args:
            items: List of strings that may contain duplicates
            
        Returns:
            List with duplicates removed, order preserved
        """
        seen = set()
        result = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _calculate_estimated_cost(self, token_usage: Dict[str, int], model: str) -> Dict[str, Any]:
        """
        Calculate estimated cost based on token usage and model pricing.
        Note: Vision workflow doesn't control high_accuracy parameter, but models other than 
        gemini-2.5-flash-preview-05-20 always use high accuracy mode.
        
        Args:
            token_usage: Dictionary containing token usage statistics
            model: Model name used for processing
            
        Returns:
            Dictionary containing cost breakdown
        """
        input_tokens = token_usage.get("input_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        
        # Calculate output tokens as total - input
        output_tokens = max(0, total_tokens - input_tokens)
        
        # Define pricing per million tokens
        pricing = {
            "gemini-2.5-flash-preview-05-20": {
                "input_cost_per_million": 0.15,
                "output_cost_per_million_normal": 0.60,
                "output_cost_per_million_high_accuracy": 3.50
            },
            "gemini-2.5-pro-preview-06-05": {
                "input_cost_per_million": 1.25,
                "output_cost_per_million_high_accuracy": 10.00  # Pro model always uses high accuracy
            }
        }
        
        # Get model pricing (default to flash if model not found)
        if model in pricing:
            model_pricing = pricing[model]
        else:
            logger.warning(f"Unknown model {model}, using gemini-2.5-flash-preview-05-20 pricing")
            model_pricing = pricing["gemini-2.5-flash-preview-05-20"]
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * model_pricing["input_cost_per_million"]
        
        # Vision workflow: gemini-2.5-flash-preview-05-20 uses normal mode, others use high accuracy
        if model == "gemini-2.5-flash-preview-05-20":
            output_cost_per_million = model_pricing["output_cost_per_million_normal"]
            accuracy_mode = "normal"
        else:
            output_cost_per_million = model_pricing["output_cost_per_million_high_accuracy"]
            accuracy_mode = "high_accuracy"
            
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost
        
        cost_breakdown = {
            "model": model,
            "accuracy_mode": accuracy_mode,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "pricing_per_million_tokens": {
                "input_cost": model_pricing["input_cost_per_million"],
                "output_cost": output_cost_per_million
            },
            "estimated_costs_usd": {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6)
            },
            "cost_breakdown_formatted": {
                "input": f"${input_cost:.6f} ({input_tokens:,} tokens × ${model_pricing['input_cost_per_million']}/M)",
                "output": f"${output_cost:.6f} ({output_tokens:,} tokens × ${output_cost_per_million}/M)",
                "total": f"${total_cost:.6f}"
            }
        }
        
        return cost_breakdown

    def _post_process_results(self, gemini_result: Dict[str, Any], full_text: str) -> tuple[Dict[str, Any], str]:
        """
        Post-process results to clean up data before saving.
        
        Args:
            gemini_result: The Gemini structured result dictionary to clean up
            full_text: The full text to clean up
            
        Returns:
            Tuple of (cleaned_gemini_result, cleaned_full_text)
        """
        # Clean up full text - remove watermarked text
        cleaned_full_text = full_text
        watermarks_to_remove = [
            "UNOFFICIAL COPY",
            "UNOFFICIAL COPY UNOFFICIAL COPY",
            "UNO",  # Sometimes partial watermarks appear
        ]
        
        for watermark in watermarks_to_remove:
            cleaned_full_text = cleaned_full_text.replace(watermark, "")
        
        # Clean up extra whitespace and line breaks caused by watermark removal
        import re
        cleaned_full_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_full_text)  # Remove excessive line breaks
        cleaned_full_text = re.sub(r' +', ' ', cleaned_full_text)  # Remove multiple spaces
        cleaned_full_text = cleaned_full_text.strip()
        
        # Clean up Gemini result
        cleaned_gemini_result = dict(gemini_result)
        
        # Remove details.TRS since it's redundant with top-level TRS
        if "details" in cleaned_gemini_result and isinstance(cleaned_gemini_result["details"], dict):
            if "TRS" in cleaned_gemini_result["details"]:
                del cleaned_gemini_result["details"]["TRS"]
                logger.info("Removed redundant details.TRS field")
        
        # Remove duplicates from all array/list fields (skip special fields)
        special_fields = {'token_usage', 'processing_status', 'error', 'error_info'}
        
        for key, value in cleaned_gemini_result.items():
            if key not in special_fields and isinstance(value, list) and all(isinstance(item, str) for item in value):
                original_count = len(value)
                cleaned_gemini_result[key] = self._remove_duplicates_from_list(value)
                new_count = len(cleaned_gemini_result[key])
                if new_count < original_count:
                    logger.info(f"Removed {original_count - new_count} duplicates from {key}")
        
        # Also clean up nested arrays in details
        if "details" in cleaned_gemini_result and isinstance(cleaned_gemini_result["details"], dict):
            for key, value in cleaned_gemini_result["details"].items():
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    original_count = len(value)
                    cleaned_gemini_result["details"][key] = self._remove_duplicates_from_list(value)
                    new_count = len(cleaned_gemini_result["details"][key])
                    if new_count < original_count:
                        logger.info(f"Removed {original_count - new_count} duplicates from details.{key}")
        
        logger.info("Post-processing completed: cleaned watermarks and removed duplicates")
        return cleaned_gemini_result, cleaned_full_text


def process_deed_pdf_vision_gemini(
    pdf_path: Path, 
    vision_credentials_path: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dpi: int = 300,
    max_retries: int = 3,
    retry_delay: float = 5.0
) -> VisionGeminiDeedResult:
    """
    Simple function to process a deed PDF with Vision + Gemini workflow.
    
    Args:
        pdf_path: Path to the PDF file
        vision_credentials_path: Path to Google Cloud service account JSON file
        gemini_api_key: Google AI API key (optional, can use environment variable)
        output_dir: Optional output directory for results
        dpi: Image resolution for PDF conversion
        max_retries: Maximum number of retries for failed operations
        retry_delay: Delay between retries in seconds
        
    Returns:
        VisionGeminiDeedResult containing extracted and structured information
    """
    workflow = VisionGeminiDeedOCRWorkflow(
        vision_credentials_path=vision_credentials_path,
        gemini_api_key=gemini_api_key,
        dpi=dpi,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    return workflow.process_pdf(pdf_path, output_dir) 
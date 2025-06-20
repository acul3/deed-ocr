"""Simplified workflow for deed OCR processing."""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from deed_ocr.ocr.gemini_ocr import create_gemini_ocr_engine
from deed_ocr.utils.pdf_converter import create_pdf_converter

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Custom exception for workflow errors."""
    def __init__(self, message: str, error_type: str = "unknown", original_error: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class SimplifiedDeedResult(BaseModel):
    """Simplified result model for deed OCR processing."""
    
    source_pdf: str = Field(description="Source PDF filename")
    total_pages: int = Field(description="Total number of pages processed")
    pages_data: List[Dict[str, Any]] = Field(description="OCR results for each page")
    combined_full_text: str = Field(description="Combined full text from all pages")
    all_legal_descriptions: List[str] = Field(description="All legal descriptions found")
    all_details: Dict[str, Any] = Field(description="Combined details from all pages")
    # Error tracking fields
    has_errors: bool = Field(default=False, description="Whether any errors occurred during processing")
    error_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of errors encountered")
    retry_needed: bool = Field(default=False, description="Whether this document needs retry")


class SimplifiedDeedOCRWorkflow:
    """Simplified workflow for deed OCR processing using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-preview-05-20", dpi: int = 300, max_retries: int = 3, retry_delay: float = 5.0, high_accuracy: bool = False):
        """
        Initialize the simplified workflow.
        
        Args:
            api_key: Google AI API key
            model: Gemini model to use (default: gemini-2.5-flash-preview-05-20)
            dpi: Image resolution for PDF conversion
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
            high_accuracy: Enable high-accuracy mode for better results
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.high_accuracy = high_accuracy
        
        try:
            self.ocr_engine = create_gemini_ocr_engine(api_key, model=model)
            self.pdf_converter = create_pdf_converter(dpi=dpi, format="PNG")
        except Exception as e:
            raise WorkflowError(f"Failed to initialize workflow components: {str(e)}", "initialization", e)
        
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
        
    def process_pdf(self, pdf_path: Path, output_dir: Optional[Path] = None) -> SimplifiedDeedResult:
        """
        Process a deed PDF through the complete workflow.
        
        Workflow: PDF → PDF-Image → OCR Engine (Gemini AI) → Structured JSON
        
        Args:
            pdf_path: Path to the deed PDF file
            output_dir: Optional directory to save results
            
        Returns:
            SimplifiedDeedResult containing all extracted information
        """
        logger.info(f"Starting simplified deed OCR workflow for: {pdf_path}")
        
        error_summary = {
            "pdf_conversion_errors": [],
            "page_processing_errors": [],
            "full_pdf_processing_error": None,
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
                raise WorkflowError(f"Failed to convert PDF to images: {str(e)}", "pdf_conversion", e)
            
            # Step 2: Process each image with Gemini AI OCR using simplified prompt
            logger.info("Step 2: Processing images with Gemini AI OCR...")
            pages_data = []
            combined_full_text = ""
            all_legal_descriptions = []
            all_reserve_retain = []
            all_oil_mineral = []
            all_trs = []
            combined_details = {}
            total_token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            
            for page_num, image_bytes in enumerate(image_bytes_list, 1):
                logger.info(f"Processing page {page_num}/{len(image_bytes_list)}")
                
                try:
                    # Extract data using simplified Gemini AI method for all PDFs
                    page_result = self._retry_operation(
                        self.ocr_engine.process_image_bytes_simplified,
                        image_bytes, 
                        "image/png",
                        self.high_accuracy
                    )
                    
                    # Add page number to the result
                    page_result["page_number"] = page_num
                    page_result["processing_status"] = "success"
                    pages_data.append(page_result)
                    
                    # Accumulate token usage
                    if "token_usage" in page_result and isinstance(page_result["token_usage"], dict):
                        token_info = page_result["token_usage"]
                        total_token_usage["input_tokens"] += token_info.get("input_tokens", 0)
                        total_token_usage["output_tokens"] += token_info.get("output_tokens", 0)
                        total_token_usage["total_tokens"] += token_info.get("total_tokens", 0)
                    
                    # Combine data across pages
                    if "full_text" in page_result:
                        combined_full_text += f"\n--- Page {page_num} ---\n{page_result['full_text']}\n"
                    
                    if "legal_description_block" in page_result:
                        legal_descriptions = page_result["legal_description_block"]
                        if isinstance(legal_descriptions, list):
                            all_legal_descriptions.extend(legal_descriptions)
                        elif legal_descriptions:
                            all_legal_descriptions.append(str(legal_descriptions))
                    
                    # Handle simplified format fields
                    if "reserve_retain" in page_result:
                        reserve_retain = page_result["reserve_retain"]
                        if isinstance(reserve_retain, list):
                            all_reserve_retain.extend(reserve_retain)
                        elif reserve_retain:
                            all_reserve_retain.append(str(reserve_retain))
                    
                    if "oil_mineral" in page_result:
                        oil_mineral = page_result["oil_mineral"]
                        if isinstance(oil_mineral, list):
                            all_oil_mineral.extend(oil_mineral)
                        elif oil_mineral:
                            all_oil_mineral.append(str(oil_mineral))
                    
                    if "TRS" in page_result:
                        trs = page_result["TRS"]
                        if isinstance(trs, list):
                            all_trs.extend(trs)
                        elif trs:
                            all_trs.append(str(trs))
                    
                    # Handle traditional details format (for single page or backward compatibility)
                    if "details" in page_result and isinstance(page_result["details"], dict):
                        # Merge details, with page-specific prefixing for conflicts
                        for key, value in page_result["details"].items():
                            if key in combined_details:
                                # Handle conflicts by creating page-specific keys
                                combined_details[f"page_{page_num}_{key}"] = value
                            else:
                                combined_details[key] = value
                                
                except Exception as e:
                    error_info = self._handle_api_error(e, f"Page {page_num} processing")
                    error_summary["page_processing_errors"].append(error_info)
                    has_errors = True
                    
                    logger.error(f"Error processing page {page_num}: {str(e)}")
                    # Continue with other pages
                    error_result = {
                        "page_number": page_num,
                        "processing_status": "error",
                        "error": str(e),
                        "error_type": error_info["error_type"],
                        "retry_recommended": error_info["retry_recommended"],
                        "full_text": "",
                        "legal_description_block": [],
                        "reserve_retain": [],
                        "oil_mineral": [],
                        "TRS": [],
                        "details": {}
                    }
                    pages_data.append(error_result)
            
            # Step 3: Create structured result
            logger.info("Step 3: Creating structured result...")
            result = SimplifiedDeedResult(
                source_pdf=pdf_path.name,
                total_pages=len(image_bytes_list),
                pages_data=pages_data,
                combined_full_text=combined_full_text.strip(),
                all_legal_descriptions=all_legal_descriptions,
                all_details=combined_details,
                has_errors=has_errors,
                error_summary=error_summary,
                retry_needed=has_errors and any(
                    error.get("retry_recommended", False) 
                    for error_list in error_summary.values() 
                    if isinstance(error_list, list)
                    for error in error_list
                )
            )
            
            # Add simplified format fields to result
            result.all_details["reserve_retain"] = all_reserve_retain
            result.all_details["oil_mineral"] = all_oil_mineral  
            result.all_details["TRS"] = all_trs
            
            # Step 4: Save results if output directory specified
            if output_dir:
                try:
                    self._save_results(result, output_dir, pdf_path, total_token_usage)
                except Exception as e:
                    error_info = self._handle_api_error(e, "Save results")
                    error_summary["save_errors"].append(error_info)
                    result.has_errors = True
                    result.error_summary = error_summary
                    logger.error(f"Error saving results: {str(e)}")
                    # Don't raise here, return result with error info
                    
            logger.info("Workflow completed successfully")
            return result
            
        except WorkflowError:
            # Re-raise workflow errors as they already have proper context
            raise
        except Exception as e:
            # Handle any unexpected errors
            error_info = self._handle_api_error(e, "Workflow execution")
            logger.error(f"Unexpected workflow error: {str(e)}")
            raise WorkflowError(f"Workflow failed unexpectedly: {str(e)}", "unexpected", e)
        
    def _save_results(self, result: SimplifiedDeedResult, output_dir: Path, original_pdf_path: Path, token_usage: Dict[str, int]) -> None:
        """Save workflow results to output directory, with each PDF getting its own folder."""
        # Create a folder named after the PDF file (without extension)
        pdf_name = Path(result.source_pdf).stem  # Remove .pdf extension
        pdf_output_dir = output_dir / pdf_name
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Apply post-processing to clean up full text (remove watermarks)
            cleaned_full_text = result.combined_full_text
            watermarks_to_remove = [
                "UNOFFICIAL COPY",
                "UNOFFICIAL COPY UNOFFICIAL COPY",
            ]
            
            for watermark in watermarks_to_remove:
                cleaned_full_text = cleaned_full_text.replace(watermark, "")
            
            # Clean up extra whitespace and line breaks caused by watermark removal
            import re
            cleaned_full_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_full_text)  # Remove excessive line breaks
            cleaned_full_text = re.sub(r' +', ' ', cleaned_full_text)  # Remove multiple spaces
            cleaned_full_text = cleaned_full_text.strip()
            
            # Save cleaned combined full text
            text_file = pdf_output_dir / "full_text.txt"
            text_file.write_text(cleaned_full_text, encoding='utf-8')
            
            # Save individual page results
            pages_dir = pdf_output_dir / "pages"
            pages_dir.mkdir(exist_ok=True)
            
            for page_data in result.pages_data:
                page_num = page_data.get("page_number", "unknown")
                page_file = pages_dir / f"page_{page_num}.json"
                with open(page_file, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, indent=2, ensure_ascii=False)
            
            # Save error summary and retry information
            if result.has_errors:
                error_file = pdf_output_dir / "error_summary.json"
                error_data = {
                    "has_errors": result.has_errors,
                    "retry_needed": result.retry_needed,
                    "error_summary": result.error_summary,
                    "failed_pages": [
                        page_data["page_number"] 
                        for page_data in result.pages_data 
                        if page_data.get("processing_status") == "error"
                    ]
                }
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Error summary saved to {error_file}")
            
        except Exception as e:
            logger.error(f"Error saving basic results: {str(e)}")
            raise
        
        # Process entire PDF for all documents (with error handling)
        try:
            logger.info(f"Processing entire PDF document ({result.total_pages} pages)")
            pdf_result = self._retry_operation(
                self.ocr_engine.process_pdf_from_result_folder, 
                str(pdf_output_dir), 
                str(original_pdf_path)
            )
            
            # Add full PDF token usage to total
            if pdf_result and isinstance(pdf_result, dict) and "token_usage" in pdf_result:
                pdf_token_info = pdf_result["token_usage"]
                if isinstance(pdf_token_info, dict):
                    token_usage["input_tokens"] += pdf_token_info.get("input_tokens", 0)
                    token_usage["output_tokens"] += pdf_token_info.get("output_tokens", 0)
                    token_usage["total_tokens"] += pdf_token_info.get("total_tokens", 0)
            
            # Save full PDF analysis result
            if pdf_result and isinstance(pdf_result, dict) and "error" not in pdf_result:
                full_pdf_file = pdf_output_dir / "full_pdf_analysis.json"
                with open(full_pdf_file, 'w', encoding='utf-8') as f:
                    json.dump(pdf_result, f, indent=2, ensure_ascii=False)
            
                # Create final merged result
                logger.info("Creating final merged result")
                final_result = self._create_final_result(pdf_output_dir, pdf_result, result.pages_data)
                
                # Apply post-processing to final result
                logger.info("Applying post-processing to final result")
                cleaned_final_result, _ = self._post_process_results(final_result, cleaned_full_text)
                
                # Save cleaned final result
                final_file = pdf_output_dir / "final_result.json"
                with open(final_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_final_result, f, indent=2, ensure_ascii=False)
                
                logger.info("Final result saved")
            
            logger.info("Full PDF analysis completed and saved")
            
        except Exception as e:
            error_info = self._handle_api_error(e, "Full PDF processing")
            result.error_summary["full_pdf_processing_error"] = error_info
            result.has_errors = True
            if error_info["retry_recommended"]:
                result.retry_needed = True
            logger.error(f"Error processing entire PDF: {str(e)}")
            
            # Save error info for full PDF processing
            try:
                full_pdf_error_file = pdf_output_dir / "full_pdf_error.json"
                with open(full_pdf_error_file, 'w', encoding='utf-8') as f:
                    json.dump(error_info, f, indent=2, ensure_ascii=False)
            except Exception as save_error:
                logger.error(f"Could not save full PDF error info: {str(save_error)}")
        
        try:
            # Save total token usage
            token_usage_file = pdf_output_dir / "token_usage.json"
            with open(token_usage_file, 'w', encoding='utf-8') as f:
                json.dump(token_usage, f, indent=2, ensure_ascii=False)
            logger.info(f"Token usage saved: {token_usage}")
            
            # Calculate and save estimated cost breakdown
            try:
                cost_breakdown = self._calculate_estimated_cost(
                    token_usage, 
                    self.ocr_engine.model, 
                    self.high_accuracy
                )
                
                cost_file = pdf_output_dir / "estimated_cost_breakdown.json"
                with open(cost_file, 'w', encoding='utf-8') as f:
                    json.dump(cost_breakdown, f, indent=2, ensure_ascii=False)
                    
                logger.info(f"Estimated cost breakdown saved to {cost_file}")
                logger.info(f"Estimated total cost: ${cost_breakdown['estimated_costs_usd']['total_cost']:.6f}")
                logger.info(f"Model: {cost_breakdown['model']} ({cost_breakdown['accuracy_mode']} mode)")
                
            except Exception as cost_error:
                logger.error(f"Error calculating cost breakdown: {str(cost_error)}")
            
            # Copy original PDF to result folder
            original_pdf_copy = pdf_output_dir / original_pdf_path.name
            shutil.copy2(original_pdf_path, original_pdf_copy)
            logger.info(f"Original PDF copied to {original_pdf_copy}")
            
        except Exception as e:
            logger.error(f"Error saving token usage or copying PDF: {str(e)}")
            # Don't raise here as main processing is done
        
        logger.info(f"Results saved to {pdf_output_dir}")

    def _create_final_result(self, pdf_output_dir: Path, full_pdf_result: Dict[str, Any], pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create final result with specific merge logic.
        
        Args:
            pdf_output_dir: Directory containing the results
            full_pdf_result: Results from full PDF analysis
            pages_data: Results from individual page processing
            
        Returns:
            Final merged result dictionary
        """
        # Start with all values from full_pdf_analysis.json
        final_result = dict(full_pdf_result)
        
        # Add full_text from full_text.txt
        full_text_file = pdf_output_dir / "full_text.txt"
        if full_text_file.exists():
            final_result["full_text"] = full_text_file.read_text(encoding='utf-8')
        
        # Collect legal_description_block from pages (remove duplicates)
        pages_legal_descriptions = []
        pages_trs = []
        pages_reserve_retain = []
        pages_oil_mineral = []
        
        for page_data in pages_data:
            # Collect legal descriptions
            if "legal_description_block" in page_data:
                legal_desc = page_data["legal_description_block"]
                if isinstance(legal_desc, list):
                    for desc in legal_desc:
                        if desc and desc not in pages_legal_descriptions:
                            pages_legal_descriptions.append(desc)
                elif legal_desc and legal_desc not in pages_legal_descriptions:
                    pages_legal_descriptions.append(str(legal_desc))
            
            # Collect TRS
            if "TRS" in page_data:
                trs_data = page_data["TRS"]
                if isinstance(trs_data, list):
                    for trs in trs_data:
                        if trs and trs not in pages_trs:
                            pages_trs.append(trs)
                elif trs_data and trs_data not in pages_trs:
                    pages_trs.append(str(trs_data))
            
            # Collect reserve_retain
            if "reserve_retain" in page_data:
                reserve_data = page_data["reserve_retain"]
                if isinstance(reserve_data, list):
                    pages_reserve_retain.extend(reserve_data)
                elif reserve_data:
                    pages_reserve_retain.append(str(reserve_data))
            
            # Collect oil_mineral
            if "oil_mineral" in page_data:
                oil_data = page_data["oil_mineral"]
                if isinstance(oil_data, list):
                    pages_oil_mineral.extend(oil_data)
                elif oil_data:
                    pages_oil_mineral.append(str(oil_data))
        
        # Merge legal_description_block (full_pdf + pages, remove duplicates)
        final_legal_descriptions = []
        
        # Add from full_pdf_analysis first
        if "legal_description_block" in final_result:
            pdf_legal_desc = final_result["legal_description_block"]
            if isinstance(pdf_legal_desc, list):
                final_legal_descriptions.extend(pdf_legal_desc)
            elif pdf_legal_desc:
                final_legal_descriptions.append(str(pdf_legal_desc))
        
        # Add from pages (avoiding duplicates)
        for desc in pages_legal_descriptions:
            if desc not in final_legal_descriptions:
                final_legal_descriptions.append(desc)
        
        final_result["legal_description_block"] = final_legal_descriptions
        
        # Merge TRS (pages + details.TRS from full_pdf)
        final_trs = list(pages_trs)  # Start with pages TRS
        
        # Add from full_pdf details.TRS
        if "details" in final_result and isinstance(final_result["details"], dict):
            details_trs = final_result["details"].get("TRS", [])
            if isinstance(details_trs, list):
                for trs in details_trs:
                    if trs and trs not in final_trs:
                        final_trs.append(trs)
            elif details_trs and details_trs not in final_trs:
                final_trs.append(str(details_trs))
        
        # Also check top-level TRS in full_pdf
        if "TRS" in final_result:
            top_level_trs = final_result["TRS"]
            if isinstance(top_level_trs, list):
                for trs in top_level_trs:
                    if trs and trs not in final_trs:
                        final_trs.append(trs)
            elif top_level_trs and top_level_trs not in final_trs:
                final_trs.append(str(top_level_trs))
        
        final_result["TRS"] = final_trs
        
        # Use reserve_retain and oil_mineral from pages
        final_result["reserve_retain"] = pages_reserve_retain
        final_result["oil_mineral"] = pages_oil_mineral
        
        logger.info(f"Created final result with {len(final_legal_descriptions)} legal descriptions, "
                   f"{len(final_trs)} TRS entries, {len(pages_reserve_retain)} reserve_retain entries, "
                   f"and {len(pages_oil_mineral)} oil_mineral entries")
        
        return final_result

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

    def _calculate_estimated_cost(self, token_usage: Dict[str, int], model: str, high_accuracy: bool = False) -> Dict[str, Any]:
        """
        Calculate estimated cost based on token usage and model pricing.
        
        Args:
            token_usage: Dictionary containing token usage statistics
            model: Model name used for processing
            high_accuracy: Whether high accuracy mode was used (only applies to gemini-2.5-flash-preview-05-20)
            
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
        
        # Determine if high accuracy was actually used
        # For gemini-2.5-flash-preview-05-20: use high_accuracy parameter
        # For all other models: always use high accuracy mode
        actual_high_accuracy = high_accuracy if model == "gemini-2.5-flash-preview-05-20" else True
        
        if actual_high_accuracy and "output_cost_per_million_high_accuracy" in model_pricing:
            output_cost_per_million = model_pricing["output_cost_per_million_high_accuracy"]
            accuracy_mode = "high_accuracy"
        else:
            output_cost_per_million = model_pricing["output_cost_per_million_normal"]
            accuracy_mode = "normal"
            
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

    def _post_process_results(self, final_result: Dict[str, Any], full_text: str) -> tuple[Dict[str, Any], str]:
        """
        Post-process results to clean up data before saving.
        
        Args:
            final_result: The final result dictionary to clean up
            full_text: The full text to clean up
            
        Returns:
            Tuple of (cleaned_final_result, cleaned_full_text)
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
        
        # Clean up final result
        cleaned_final_result = dict(final_result)
        
        # Remove details.TRS since it's redundant with top-level TRS
        if "details" in cleaned_final_result and isinstance(cleaned_final_result["details"], dict):
            if "TRS" in cleaned_final_result["details"]:
                del cleaned_final_result["details"]["TRS"]
                logger.info("Removed redundant details.TRS field")
        
        # Remove duplicates from all array/list fields (skip special fields)
        special_fields = {'token_usage', 'processing_status', 'error', 'error_info', 'full_text'}
        
        for key, value in cleaned_final_result.items():
            if key not in special_fields and isinstance(value, list) and all(isinstance(item, str) for item in value):
                original_count = len(value)
                cleaned_final_result[key] = self._remove_duplicates_from_list(value)
                new_count = len(cleaned_final_result[key])
                if new_count < original_count:
                    logger.info(f"Removed {original_count - new_count} duplicates from {key}")
        
        # Also clean up nested arrays in details
        if "details" in cleaned_final_result and isinstance(cleaned_final_result["details"], dict):
            for key, value in cleaned_final_result["details"].items():
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    original_count = len(value)
                    cleaned_final_result["details"][key] = self._remove_duplicates_from_list(value)
                    new_count = len(cleaned_final_result["details"][key])
                    if new_count < original_count:
                        logger.info(f"Removed {original_count - new_count} duplicates from details.{key}")
        
        logger.info("Post-processing completed: cleaned watermarks and removed duplicates")
        return cleaned_final_result, cleaned_full_text


def process_deed_pdf_simple(
    pdf_path: Path, 
    api_key: Optional[str] = None, 
    model: str = "gemini-2.5-flash-preview-05-20",
    output_dir: Optional[Path] = None,
    dpi: int = 300,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    high_accuracy: bool = False
) -> SimplifiedDeedResult:
    """
    Simple function to process a deed PDF with minimal setup.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Google AI API key (optional, can use environment variable)
        model: Gemini model to use (default: gemini-2.5-flash-preview-05-20)
        output_dir: Optional output directory for results
        dpi: Image resolution for PDF conversion
        max_retries: Maximum number of retries for failed operations
        retry_delay: Delay between retries in seconds
        high_accuracy: Enable high-accuracy mode for better results
        
    Returns:
        SimplifiedDeedResult containing extracted information
    """
    workflow = SimplifiedDeedOCRWorkflow(api_key=api_key, model=model, dpi=dpi, max_retries=max_retries, retry_delay=retry_delay, high_accuracy=high_accuracy)
    return workflow.process_pdf(pdf_path, output_dir) 
"""Google Cloud Vision OCR Engine for deed document processing."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from google.cloud import vision

logger = logging.getLogger(__name__)


class VisionOCRError(Exception):
    """Custom exception for Vision OCR errors."""
    def __init__(self, message: str, error_type: str = "unknown", original_error: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class VisionOCREngine:
    """OCR Engine using Google Cloud Vision API for text extraction."""
    
    def __init__(self, credentials_path: Optional[str] = None, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Initialize Vision OCR Engine.
        
        Args:
            credentials_path: Path to Google Cloud service account JSON file
                            If None, will use GOOGLE_APPLICATION_CREDENTIALS environment variable
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            logger.info(f"Set Google Cloud credentials path: {credentials_path}")
        
        try:
            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision client: {str(e)}")
            raise VisionOCRError(f"Failed to initialize Vision client: {str(e)}", "initialization", e)
        
    def _handle_api_error(self, error: Exception, operation: str) -> Dict[str, str]:
        """
        Handle and categorize Vision API errors.
        
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
        elif "rate limit" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
            error_info["error_type"] = "rate_limit"
            error_info["retry_recommended"] = True
        elif "network" in error_str or "connection" in error_str or "unreachable" in error_str:
            error_info["error_type"] = "network"
            error_info["retry_recommended"] = True
        elif "credentials" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            error_info["error_type"] = "authentication"
            error_info["retry_recommended"] = False
        elif "not found" in error_str or "404" in error_str:
            error_info["error_type"] = "not_found"
            error_info["retry_recommended"] = False
        elif "internal server error" in error_str or "500" in error_str or "internal_error" in error_str:
            error_info["error_type"] = "server_error"
            error_info["retry_recommended"] = True
        elif "invalid_argument" in error_str or "bad request" in error_str:
            error_info["error_type"] = "invalid_argument"
            error_info["retry_recommended"] = False
        elif "permission" in error_str or "forbidden" in error_str:
            error_info["error_type"] = "permission"
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
                    logger.warning(f"Vision API attempt {attempt + 1} failed: {error_info['error_message']}. "
                                 f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    break
        
        raise last_error
    
    def _safe_document_text_detection(self, image):
        """
        Safely perform document text detection with proper error handling.
        
        Args:
            image: Vision API image object
            
        Returns:
            Vision API response
        """
        try:
            response = self.client.document_text_detection(image=image)
            
            # Check for errors in the response
            if response.error.message:
                raise Exception(f"Google Vision API error: {response.error.message}")
            
            return response
        except Exception as e:
            error_info = self._handle_api_error(e, "document_text_detection")
            logger.error(f"Vision API error: {error_info}")
            raise VisionOCRError(f"Vision API call failed: {str(e)}", error_info["error_type"], e)
        
    def extract_text_from_image_bytes(self, image_bytes: bytes) -> str:
        """
        Extract text from image bytes using Google Vision API.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted text from the image
        """
        try:
            image = vision.Image(content=image_bytes)
            response = self._retry_operation(self._safe_document_text_detection, image)
            
            # Extract full text
            full_text = response.full_text_annotation.text if response.full_text_annotation else ""
            
            logger.info(f"Successfully extracted {len(full_text)} characters from image")
            return full_text
            
        except VisionOCRError:
            raise  # Re-raise our custom errors
        except Exception as e:
            error_info = self._handle_api_error(e, "extract_text_from_image_bytes")
            logger.error(f"Unexpected error in extract_text_from_image_bytes: {str(e)}")
            raise VisionOCRError(f"Failed to extract text from image bytes: {str(e)}", error_info["error_type"], e)
            
    def extract_text_from_image_path(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from image file using Google Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            
            return self.extract_text_from_image_bytes(content)
            
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {image_path}")
            raise VisionOCRError(f"Image file not found: {image_path}", "file_not_found", e)
        except PermissionError as e:
            logger.error(f"Permission denied reading image file: {image_path}")
            raise VisionOCRError(f"Permission denied reading image file: {image_path}", "permission", e)
        except Exception as e:
            logger.error(f"Error reading image file {image_path}: {str(e)}")
            raise VisionOCRError(f"Failed to read image file: {str(e)}", "file_read_error", e)
            
    def extract_text_from_multiple_images(self, image_bytes_list: list) -> Dict[str, Any]:
        """
        Extract text from multiple images and combine results.
        
        Args:
            image_bytes_list: List of image bytes
            
        Returns:
            Dictionary containing combined text and individual page results
        """
        combined_text = ""
        page_results = []
        processing_errors = []
        
        try:
            for page_num, image_bytes in enumerate(image_bytes_list, 1):
                logger.info(f"Processing page {page_num}/{len(image_bytes_list)}")
                
                try:
                    page_text = self._retry_operation(self.extract_text_from_image_bytes, image_bytes)
                    page_result = {
                        "page_number": page_num,
                        "text": page_text,
                        "character_count": len(page_text),
                        "processing_status": "success"
                    }
                    page_results.append(page_result)
                    
                    # Add to combined text with page separator
                    combined_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                    
                except Exception as e:
                    error_info = self._handle_api_error(e, f"Page {page_num} processing")
                    processing_errors.append({
                        "page_number": page_num,
                        "error_info": error_info
                    })
                    logger.error(f"Error processing page {page_num}: {str(e)}")
                    
                    error_result = {
                        "page_number": page_num,
                        "error": str(e),
                        "error_type": error_info["error_type"],
                        "retry_recommended": error_info["retry_recommended"],
                        "text": "",
                        "character_count": 0,
                        "processing_status": "error"
                    }
                    page_results.append(error_result)
            
            result = {
                "combined_text": combined_text.strip(),
                "total_pages": len(image_bytes_list),
                "page_results": page_results,
                "total_characters": len(combined_text.strip()),
                "processing_errors": processing_errors,
                "has_errors": len(processing_errors) > 0,
                "successful_pages": len(image_bytes_list) - len(processing_errors)
            }
            
            logger.info(f"Successfully processed {result['successful_pages']}/{len(image_bytes_list)} pages, "
                       f"extracted {result['total_characters']} total characters")
            
            return result
            
        except Exception as e:
            error_info = self._handle_api_error(e, "extract_text_from_multiple_images")
            logger.error(f"Unexpected error processing multiple images: {str(e)}")
            raise VisionOCRError(f"Failed to process multiple images: {str(e)}", error_info["error_type"], e)
            
    def extract_detailed_text_info(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract detailed text information including bounding boxes and confidence.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary containing detailed text information
        """
        try:
            image = vision.Image(content=image_bytes)
            response = self._retry_operation(self._safe_document_text_detection, image)
            
            result = {
                "full_text": "",
                "pages": [],
                "blocks": [],
                "paragraphs": [],
                "words": [],
                "symbols": []
            }
            
            if not response.full_text_annotation:
                logger.warning("No text detected in image")
                return result
            
            # Extract full text
            result["full_text"] = response.full_text_annotation.text
            
            # Extract detailed structure
            for page in response.full_text_annotation.pages:
                page_info = {
                    "width": page.width,
                    "height": page.height,
                    "blocks": len(page.blocks)
                }
                result["pages"].append(page_info)
                
                for block in page.blocks:
                    block_text = ""
                    for paragraph in block.paragraphs:
                        paragraph_text = ""
                        for word in paragraph.words:
                            word_text = ""
                            for symbol in word.symbols:
                                word_text += symbol.text
                                result["symbols"].append({
                                    "text": symbol.text,
                                    "confidence": symbol.confidence if hasattr(symbol, 'confidence') else None
                                })
                            result["words"].append({
                                "text": word_text,
                                "confidence": word.confidence if hasattr(word, 'confidence') else None
                            })
                            paragraph_text += word_text + " "
                        result["paragraphs"].append({
                            "text": paragraph_text.strip(),
                            "confidence": paragraph.confidence if hasattr(paragraph, 'confidence') else None
                        })
                        block_text += paragraph_text
                    result["blocks"].append({
                        "text": block_text.strip(),
                        "confidence": block.confidence if hasattr(block, 'confidence') else None
                    })
            
            logger.info(f"Extracted detailed text info: {len(result['words'])} words, "
                       f"{len(result['paragraphs'])} paragraphs, {len(result['blocks'])} blocks")
            
            return result
            
        except VisionOCRError:
            raise  # Re-raise our custom errors
        except Exception as e:
            error_info = self._handle_api_error(e, "extract_detailed_text_info")
            logger.error(f"Unexpected error extracting detailed text info: {str(e)}")
            raise VisionOCRError(f"Failed to extract detailed text info: {str(e)}", error_info["error_type"], e)


def create_vision_ocr_engine(credentials_path: Optional[str] = None, max_retries: int = 3, retry_delay: float = 2.0) -> VisionOCREngine:
    """
    Factory function to create Vision OCR Engine.
    
    Args:
        credentials_path: Path to Google Cloud service account JSON file
                         If None, will use GOOGLE_APPLICATION_CREDENTIALS environment variable
        max_retries: Maximum number of retries for failed operations
        retry_delay: Delay between retries in seconds
        
    Returns:
        Configured VisionOCREngine instance
    """
    return VisionOCREngine(credentials_path=credentials_path, max_retries=max_retries, retry_delay=retry_delay) 
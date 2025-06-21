"""Gemini AI OCR Engine for deed document processing."""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
import json_repair

logger = logging.getLogger(__name__)


class GeminiOCRError(Exception):
    """Custom exception for Gemini OCR errors."""
    def __init__(self, message: str, error_type: str = "unknown", original_error: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class GeminiOCREngine:
    """OCR Engine using Google Gemini AI for deed document processing."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", max_retries: int = 3, retry_delay: float = 2.0):
        """
        Initialize Gemini OCR Engine.
        
        Args:
            api_key: Google AI API key
            model: Gemini model to use
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise GeminiOCRError(f"Failed to initialize Gemini client: {str(e)}", "initialization", e)
        
    def _handle_api_error(self, error: Exception, operation: str) -> Dict[str, str]:
        """
        Handle and categorize Gemini API errors.
        
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
        elif "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            error_info["error_type"] = "authentication"
            error_info["retry_recommended"] = False
        elif "not found" in error_str or "404" in error_str:
            error_info["error_type"] = "not_found"
            error_info["retry_recommended"] = False
            if "model" in error_str:
                error_info["error_type"] = "model_not_found"
        elif "internal server error" in error_str or "500" in error_str or "internal_error" in error_str:
            error_info["error_type"] = "server_error"
            error_info["retry_recommended"] = True
        elif "json" in error_str and ("parse" in error_str or "decode" in error_str):
            error_info["error_type"] = "json_parsing"
            error_info["retry_recommended"] = True
        elif "content_filter" in error_str or "safety" in error_str:
            error_info["error_type"] = "content_filter"
            error_info["retry_recommended"] = False
        elif "invalid_argument" in error_str:
            error_info["error_type"] = "invalid_argument"
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
                    logger.warning(f"Gemini API attempt {attempt + 1} failed: {error_info['error_message']}. "
                                 f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    break
        
        raise last_error
    
    def _safe_generate_content(self, contents, config):
        """
        Safely generate content with proper error handling.
        
        Args:
            contents: Content to send to Gemini
            config: Generation configuration
            
        Returns:
            Generated response
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response
        except Exception as e:
            error_info = self._handle_api_error(e, "generate_content")
            logger.error(f"Gemini API error: {error_info}")
            raise GeminiOCRError(f"Gemini API call failed: {str(e)}", error_info["error_type"], e)
    
    def _parse_json_response(self, response_text: str, fallback_structure: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Parse JSON response with fallback handling.
        
        Args:
            response_text: Raw response text from API
            fallback_structure: Fallback structure if parsing fails
            
        Returns:
            Parsed JSON or fallback structure
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing error: {json_error}")
            logger.error(f"Attempting to repair JSON...")
            
            try:
                result = json_repair.loads(response_text)
                logger.info("Successfully parsed JSON after repair")
                return result
            except Exception as repair_error:
                logger.error(f"JSON repair failed: {repair_error}")
                
                # Return fallback structure with error info
                fallback = fallback_structure or {
                    "full_text": "",
                    "legal_description_block": [],
                    "details": {}
                }
                fallback.update({
                    "error": "JSON parsing failed",
                    "json_error": str(json_error),
                    "repair_error": str(repair_error),
                    "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
                })
                return fallback
        
    def extract_from_image(self, base64_image: str) -> Dict[str, Any]:
        """
        Extract structured data from a deed image using Gemini AI.
        
        Args:
            base64_image: Base64 encoded image data
            
        Returns:
            Dictionary containing extracted text, legal description, and details
        """
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            mime_type="image/png",
                            data=base64.b64decode(base64_image),
                        ),
                        types.Part.from_text(text="""Extract and structure information from legal property document PDF images.

Task: Analyze the provided legal property document image and extract all relevant information into the specified JSON format.

Output Format:
{
   "full_text":"<Complete verbatim extracted text from the document image>",
   "legal_description_block":[
      "<Full legal property descriptions including metes and bounds, lot/block references, and any survey information>"
   ],
   "details":{
      "document_type":"<Primary document category (e.g., Deed, Decree, Stipulation, OGL(Oil and Gas Lease))>",
      "document_subtype":"<Specific document type (e.g., Warranty Deed, Quitclaim Deed, Mineral Deed, Oil & Gas Lease, Decree of Heirship, Quiet Title Decree)>",
      "parties":{
         "<party_type_1>":[
           {"name":"<Name of the party>", "address":"<Address of the party if any>"},// if there are multiple parties in the same role, include all in the array
         ],
         "<party_type_2>":[
            {"name":"<Name of the party>", "address":"<Address of the party if any>"},// if there are multiple parties in the same role, include all in the array
         ],
         "<additional_party_types>":[
            {"name":"<Name of the additional party>", "address":"<Address of the additional party if any>"},// if there are multiple parties in the same role, include all in the array
         ]
      },
      "TRS":[
         "<Township/Range/Section reference 1, e.g., T2N R3W S14>",
         "<Township/Range/Section reference 2>"
      ],
      "<other relevant fields>":"<values>"
   }
}

Instructions:
1. Extract ALL text exactly as it appears in the document for "full_text"
2. Identify and extract complete legal descriptions, including all survey details
3. Use None or empty String for any fields not present in the document
4. Maintain original spelling and capitalization for names
5. If multiple properties are described, include all in the respective arrays
6. Add any document-specific fields not covered above as "<other relevant fields>"

Party Type Guidelines:
- For DEEDS: Use "grantor" and "grantee"
- For LEASES: Use "lessor" and "lessee"
- For DECREES:  Use "plaintiff" and "defendant"
- For STIPULATION: Use "Stipulating_Party_1", "Stipulating_Party_2" so on
- For OTHER DOCUMENTS: Use the party designations as they appear in the document"""),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )
            
            response = self._retry_operation(self._safe_generate_content, contents, generate_content_config)
            
            # Log token usage
            token_usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
                logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            # Parse JSON response
            result = self._parse_json_response(response.text)
            result["token_usage"] = token_usage
            logger.info("Successfully extracted data from image using Gemini AI")
            return result
            
        except GeminiOCRError:
            raise  # Re-raise our custom errors
        except Exception as e:
            error_info = self._handle_api_error(e, "extract_from_image")
            logger.error(f"Unexpected error in extract_from_image: {str(e)}")
            raise GeminiOCRError(f"Failed to extract data from image: {str(e)}", error_info["error_type"], e)
            
    def process_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> Dict[str, Any]:
        """
        Process image bytes directly without base64 encoding.
        
        Args:
            image_bytes: Raw image bytes
            mime_type: MIME type of the image
            
        Returns:
            Dictionary containing extracted text, legal description, and details
        """
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            mime_type=mime_type,
                            data=image_bytes,
                        ),
                        types.Part.from_text(text="""Extract and structure information from legal property  document PDF images.

Task: Analyze the provided legal property document image and extract all relevant information into the specified JSON format.

Output Format:
{
   "full_text":"<Complete verbatim extracted text from the document image>",
   "legal_description_block":[
      "<Full legal property descriptions including metes and bounds, lot/block references, and any survey information>"
   ],
   "details":{
      "document_type":"<Primary document category (e.g., Deed, Decree, Stipulation, Lease)>",
      "document_subtype":"<Specific document type (e.g., Warranty Deed, Quitclaim Deed, Mineral Deed, Oil & Gas Lease, Decree of Heirship, Quiet Title Decree)>",
      "parties":{
         "<party_type_1>":[
            "<Names of parties in this role>"
         ],
         "<party_type_2>":[
            "<Names of parties in this role>"
         ],
         "<additional_party_types>":[
            "<Names as needed>"
         ]
      },
      "TRS":[
         "<Township/Range/Section reference 1, e.g., T2N R3W S14>",
         "<Township/Range/Section reference 2>"
      ],
      "<other relevant fields>":"<values>"
   }
}

Instructions:
1. Extract ALL text exactly as it appears in the document for "full_text"
2. Identify and extract complete legal descriptions, including all survey details
3. Use None or empty String for any fields not present in the document
4. Maintain original spelling and capitalization for names
5. If multiple properties are described, include all in the respective arrays
6. Add any document-specific fields not covered above as "<other relevant fields>"

Party Type Guidelines:
- For DEEDS: Use "grantor" and "grantee"
- For LEASES: Use "lessor" and "lessee"
- For DECREES:  Use "plaintiff" and "defendant"
- For STIPULATION: Use "Stipulating_Party_1", "Stipulating_Party_2" so on
- For OTHER DOCUMENTS: Use the party designations as they appear in the document"""),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )
            
            response = self._retry_operation(self._safe_generate_content, contents, generate_content_config)
            
            # Log token usage
            token_usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
                logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            # Parse JSON response
            result = self._parse_json_response(response.text)
            result["token_usage"] = token_usage
            logger.info("Successfully extracted data from image bytes using Gemini AI")
            return result
            
        except GeminiOCRError:
            raise  # Re-raise our custom errors
        except Exception as e:
            error_info = self._handle_api_error(e, "process_image_bytes")
            logger.error(f"Unexpected error in process_image_bytes: {str(e)}")
            raise GeminiOCRError(f"Failed to process image bytes: {str(e)}", error_info["error_type"], e)

    def process_image_bytes_simplified(self, image_bytes: bytes, mime_type: str = "image/png", high_accuracy: bool = False) -> Dict[str, Any]:
        """
        Process image bytes with simplified prompt for multi-page documents.
        
        Args:
            image_bytes: Raw image bytes
            mime_type: MIME type of the image
            high_accuracy: Enable high-accuracy mode (removes thinking budget limitation)
            
        Returns:
            Dictionary containing extracted text and specific fields
        """
        fallback_structure = {
            "full_text": "",
            "legal_description_block": [],
            "reserve_retain": [],
            "oil_mineral": [],
            "TRS": []
        }
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            mime_type=mime_type,
                            data=image_bytes,
                        ),
                        types.Part.from_text(text="""Carefully Extract and structure information from legal property document PDF images.

Task: Analyze carefully the provided legal property document image and extract all relevant information into the specified JSON format.

Output Format:
{
   "full_text":"<Complete verbatim extracted text from the document image>",
   "legal_description_block":[
      "<sentence contain contains the legal description of the property >"
   ],
   "judgment_description":["<Decision reached by court as phrased in the document. Starting with the words "IT IS ORDERED, ADJUDGED AND DECREED that...>"],
   "reserve_retain":["<sentence contains any of the terms similar or equal to reservation,exception,or retain.>"],
   "oil_mineral":["sentence where either "oil" or "minerals" are mentioned"],
   "aliquot":["<Aliquot description found in most documents,including metes and bounds, lot/block references all.">"],
   "Interest_fraction":"[<sentence that include if the extent of interest transferred from the grantor to the grantee(s)>]",
   "grantors_interest_mentioned":"<sentence provide an excerpt where there is a declaration of the grantor conveying "all" of their interest, which might refer broadly to conveying all the real property or all the parcel of land described>",
   "TRS":[
         "<Township/Range/Section reference 1, e.g.Township 7 North, Range 58 West , T2N R3W S14>",
         "<Township/Range/Section reference 2>"
      ]
    TRS_details:[
        {
            "Township":"<Township number> e.g 10N",
            "Range":"<Range number> e.g 59W",
            "Section":"<Section number> e.g 10",
            "TRS":"<Township/Range/Section reference, e.g. T2N R3W S14>"
            "County":"<County name ONLY> e.g Weld,Dodge,etc (comma separated if multiple counties)",
        }, // if there are multiple TRS, include all in the array
    ]
}

Instructions:
1. Extract ALL text exactly as it appears in the document for "full_text"
2. Use empty String or None for any fields not present in the document
3. Aliquot should be standardized remove 1/4 and /4 , and replace 1/2 with 2 for example : S½ -> S2, "The East Half of the Northwest Quarter" -> "E2NW",SW1/4NE1/4 -> "SWNE", "E1/2NW1/4" -> "E2NW", "E/2NW/4" -> "E2NW", "E2NW", "E1/2 NW" -> "E2NW"""),
                    ],
                ),
            ]
            
            # Determine if we should use high accuracy mode
            # For gemini-2.5-flash: use high_accuracy parameter
            # For all other models: always use high accuracy mode
            use_high_accuracy = high_accuracy if self.model == "gemini-2.5-flash" else True
            
            if use_high_accuracy:
                # High-accuracy mode: no thinking budget limitation
                generate_content_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            else:
                # Default mode: fast processing with thinking budget limitation (only for flash model)
                generate_content_config = types.GenerateContentConfig(
                    thinking_config = types.ThinkingConfig(
                        thinking_budget=0,
                    ),
                    response_mime_type="application/json",
                )
            
            response = self._retry_operation(self._safe_generate_content, contents, generate_content_config)
            
            # Log and capture token usage
            token_usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
                logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            try:
                result = self._parse_json_response(response.text)
                result["token_usage"] = token_usage
                logger.info("Successfully processed Image with simplified prompt using Gemini AI")
                return result
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing error: {json_error}")
                logger.error(f"try to repair the json")
                try:
                    result = json_repair.loads(response.text)
                    logger.info("Successfully parsed JSON after cleaning")
                    return result
                except Exception as e:
                    logger.error("Failed to parse JSON even after cleaning")
                    # Return a fallback structure
                    return {
                        "error": "JSON parsing failed",
                        "raw_response": response.text,
                        "full_text": "",
                        "legal_description_block": [],
                        "details": {}
                    } 
            
        
        except Exception as e:
            try:
                result = json_repair.loads(response.text)
                logger.info("Successfully parsed JSON after cleaning")
                return result
            except Exception as f:
                logger.error(f"Failed to parse JSON even after cleaning {str(f)}")
            error_info = self._handle_api_error(e, "process_image_bytes_simplified")
            logger.error(f"Unexpected error in process_image_bytes_simplified: {str(e)}")
            
            # Return fallback structure with error info
            fallback_structure.update({
                "error": str(e),
                "error_type": error_info["error_type"],
                "retry_recommended": error_info["retry_recommended"]
            })
            return fallback_structure

    def process_entire_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process entire PDF file directly with Gemini AI for multi-page documents.
        
        Args:
            pdf_path: Path to the original PDF file
            
        Returns:
            Dictionary containing structured legal document information
        """
        try:
            # Read and encode PDF to base64
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            mime_type="application/pdf",
                            data=pdf_bytes,
                        ),
                        types.Part.from_text(text="""Carefully analyze, extract, and structure information from provided legal property document PDF images.

Task: Analyze the provided legal property document image and extract all relevant information into the specified JSON format.

Output Format:
{
   "legal_description_block":[
      "<sentence that include legal property descriptions including metes and bounds, lot/block references,etc>"
   ],
   "details":{
      "document_type":"<Primary document category can be interpreted from the title of the document or first page of the document (Deed, Decree, Stipulation or Oil & Gas Lease)>",
      "document_subtype":"<Specific document type (e.g., Warranty Deed, Quitclaim Deed, Mineral Deed, Decree of Heirship, Quiet Title Decree,Stipulation of Interest)>",
      "document_date":"<Date in which the document was made and/or executed, use Format: YYYY-MM-DD>",
      "recorded_date":"<Date in which the document was recorded, use Format: YYYY-MM-DD>",
      "effective_date":"<Date in which the document became effective if any, use Format: YYYY-MM-DD>",
      "reception_number":"<the reception or recording number of the document if any>",
      "grantor":[{
          "name":"<Name of the which party is conveying rights >",
          "address":"<Address of the grantor if any, change the state name to its abbreviation if any>" 
          "type":"<Type of the grantor, e.g.grantor(for deed), Lessor(for oil and gas lease) ,plaintiff (for decree),Deceased(for decree),Stipulating_Party(for stipulation)>"
      },// if there are multiple grantees, include all in the array],
      "grantee":[{
          "name":"<Name of the which party is conveying rights >",
          "address":"<Address of the grantee if any, change the state name to its abbreviation if any>"
          "type":"<Type of the grantee, e.g.grantee(for deed), lessee(for oil and gas lease), defendant(for decree),Recipients(for decree),Stipulating_Party(for stipulation)>"
      },// if there are multiple grantees, include all in the array],
      "deed_details":{
         "grantors_interest":"<Specifically check if there is a declaration of the grantor conveying all or a portion of their interest return true or false otherwise return null>",
         "Interest_fraction":"<sentence that include if the extent of interest transferred from the grantor to the grantee(s)>",
         "subject_to":"<sentence that includes the phrase "subject to," focusing on documented encumbrances or reservations.>",
      },
      "lease_details":{
         "gross_acreage":"<number of the total acreage of the property>",
         "lease_royalty":"<number of the royalty percentage of the property>",
         "lease_primary_term":"<the term of the lease>",
         "lease_secondary_lease_term":"<the secondary lease term of the lease following the primary lease term if any>",
      },
      "county":"<primary county for the property being conveyed in this legal document if any> e.g Weld ",
      "state":"<primary state for the property being conveyed in this legal document if any> e.g Colorado",
      "book_county":"<identifier for the document in the county record>",
      "page_county":"<identifier number for the page of the document in the county record>",
      "<other relevant fields>":"<values>"
   }
}

Instructions:
1. Use None or empty String for any fields not present in the document
2. Identify and extract complete legal descriptions
3. Maintain original spelling and capitalization for names
4. If multiple properties are described, include all in the respective arrays
5. Add any document-specific fields not covered above as "<other relevant fields>"

IMPORTANT: If no values are found in the document for deed_details or lease_details, set the entire object to None.
Only populate these objects if the document contains the relevant information"""),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )
            
            response = self._retry_operation(self._safe_generate_content, contents, generate_content_config)
            
            # Log and capture token usage
            token_usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
                logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            # Parse JSON response with improved error handling
            try:
                result = self._parse_json_response(response.text)
                result["token_usage"] = token_usage
                logger.info("Successfully processed entire PDF using Gemini AI")
                return result
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing error: {json_error}")
                logger.error(f"try to repair the json")
                try:
                    result = json_repair.loads(response.text)
                    logger.info("Successfully parsed JSON after cleaning")
                    return result
                except Exception as e:
                    logger.error("Failed to parse JSON even after cleaning")
                    # Return a fallback structure
                    return {
                        "error": "JSON parsing failed",
                        "raw_response": response.text,
                        "full_text": "",
                        "legal_description_block": [],
                        "details": {}
                    } 
        except Exception as e:
            logger.error(f"Error processing entire PDF with Gemini AI: {str(e)}")
            raise

    def process_pdf_from_result_folder(self, pdf_result_folder: str, original_pdf_path: str) -> Dict[str, Any]:
        """
        Process entire PDF file and save results.
        Processes all PDFs regardless of page count.
        
        Args:
            pdf_result_folder: Path to the PDF result folder
            original_pdf_path: Path to the original PDF file
            
        Returns:
            Dictionary containing structured legal document information
        """
        try:
            result_folder = Path(pdf_result_folder)
            
            # Check page count for logging purposes
            pages_folder = result_folder / "pages"
            if pages_folder.exists():
                page_files = list(pages_folder.glob("page_*.json"))
            else:
                page_files = []
            
            logger.info(f"PDF has {len(page_files)} page(s). Processing entire PDF: {original_pdf_path}")
            
            # Process the entire PDF with Gemini
            result = self.process_entire_pdf(original_pdf_path)
            
            # Save the result even if there were parsing errors
            output_file = result_folder / "full_pdf_analysis.json"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Full PDF analysis saved to {output_file}")
            except Exception as save_error:
                logger.error(f"Error saving PDF analysis: {save_error}")
                # Try to save raw response as backup
                backup_file = result_folder / "full_pdf_analysis_backup.txt"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(str(result))
                logger.info(f"Backup analysis saved to {backup_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing full PDF from result folder: {str(e)}")
            raise

    def process_extracted_text(self, ocr_text: str) -> Dict[str, Any]:
        """
        Process extracted OCR text to extract structured legal document information.
        
        Args:
            ocr_text: The extracted text from OCR processing
            
        Returns:
            Dictionary containing structured legal document information
        """
        try:
            prompt_text = f"""Extract and structure information from extracted legal property document ocr text.

Task: Analyze the provided extracted legal property document ocr text and extract all relevant information into the specified JSON format.

Output Format:
{{
   "legal_description_block":[
      "<sentence that include legal property descriptions including metes and bounds, lot/block references,etc>"
   ],
   "reserve_retain":["<sentence contains any of the terms similar or equal to reservation,exception,or retain.>"],
   "oil_mineral":["sentence where either "oil" or "minerals" are mentioned"],
   "details":{{
      "document_type":"<Primary document category (e.g., Deed, Decree, Stipulation, Lease)>",
      "document_subtype":"<Specific document type (e.g., Warranty Deed, Quitclaim Deed, Mineral Deed, Oil & Gas Lease, Decree of Heirship, Quiet Title Decree)>",
      "parties":{{
         "<party_type_1>":[
            "<Names of parties in this role>"
         ],
         "<party_type_2>":[
            "<Names of parties in this role>"
         ],
         "<additional_party_types>":[
            "<Names as needed>"
         ]
      }},
      "TRS":[
         "<Township/Range/Section reference 1, e.g., Township 7 North, Range 58 West , T2N R3W S14>",
         "<Township/Range/Section reference 2>"
      ],
      "deed_details":{{
         "grantors_interest":"<grantors interest in the property>",
         "Interest_fraction":"<sentence that include if the extent of interest transferred from the grantor to the grantee(s)>",
         "subject_to":"<sentence that includes the phrase "subject to," focusing on documented encumbrances or reservations.>",
      }},
      "lease_details":{{
         "gross_acreage":"<the total acreage of the property>",
         "lease_royalty":"<the royalty percentage of the property>",
         "lease_term":"<the term of the lease>",
      }},
      "<other relevant fields>":"<values>"
   }}
}}

Instructions:
1. Use None or empty String for any fields not present in the document
2. Identify and extract complete legal descriptions
3. Maintain original spelling and capitalization for names
4. If multiple properties are described, include all in the respective arrays
5. Add any document-specific fields not covered above as "<other relevant fields>"

Party Type Guidelines:
- For DEEDS: Use "grantor" and "grantee"
- For LEASES: Use "lessor" and "lessee"
- For DECREES:  Use "plaintiff" and "defendant"
- For STIPULATION: Use "Stipulating_Party_1", "Stipulating_Party_2" so on
- For OTHER DOCUMENTS: Use the party designations as they appear in the document

IMPORTANT: If no values are found in the document for deed_details or lease_details, set the entire object to None.
If the document is not a Deed,Decree, or Stipulation set "deed_details": None
If the document is not a lease, set "lease_details": None
Only populate these objects if the document contains the relevant information

ocr text: {ocr_text}"""
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt_text),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )
            
            response = self._retry_operation(self._safe_generate_content, contents, generate_content_config)
            
            # Log and capture token usage
            token_usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
                logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            # Parse JSON response with improved error handling
            try:
                result = self._parse_json_response(response.text)
                result["token_usage"] = token_usage
                logger.info("Successfully processed entire PDF using Gemini AI")
                return result
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing error: {json_error}")
                logger.error(f"try to repair the json")
                try:
                    result = json_repair.loads(response.text)
                    logger.info("Successfully parsed JSON after cleaning")
                    return result
                except Exception as e:
                    logger.error("Failed to parse JSON even after cleaning")
                    # Return a fallback structure
                    return {
                        "error": "JSON parsing failed",
                        "raw_response": response.text,
                        "full_text": "",
                        "legal_description_block": [],
                        "details": {}
                    } 
            
        except Exception as e:
            logger.error(f"Error processing extracted text with Gemini AI: {str(e)}")
            raise


def create_gemini_ocr_engine(api_key: Optional[str] = None, model: str = "gemini-2.5-flash", max_retries: int = 3, retry_delay: float = 2.0) -> GeminiOCREngine:
    """
    Factory function to create Gemini OCR Engine.
    
    Args:
        api_key: Google AI API key (if None, will try to get from environment)
        model: Gemini model to use (default: gemini-2.5-flash)
        max_retries: Maximum number of retries for failed operations
        retry_delay: Delay between retries in seconds
        
    Returns:
        Configured GeminiOCREngine instance
    """
    import os
    
    if api_key is None:
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        
    if not api_key:
        raise ValueError("Google AI API key is required. Set GOOGLE_AI_API_KEY environment variable or pass api_key parameter.")
        
    return GeminiOCREngine(api_key=api_key, model=model, max_retries=max_retries, retry_delay=retry_delay) 
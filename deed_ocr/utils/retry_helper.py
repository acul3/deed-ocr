"""Utility functions to help identify and retry failed OCR processes."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class RetryHelper:
    """Helper class to identify and manage failed OCR processes for retry."""
    
    @staticmethod
    def scan_output_directory(output_dir: Path) -> Dict[str, Any]:
        """
        Scan output directory to identify failed processes that need retry.
        
        Args:
            output_dir: Directory containing OCR results
            
        Returns:
            Dictionary with retry information
        """
        retry_info = {
            "failed_documents": [],
            "failed_pages": [],
            "summary": {
                "total_documents_scanned": 0,
                "documents_with_errors": 0,
                "documents_needing_retry": 0,
                "total_failed_pages": 0
            }
        }
        
        if not output_dir.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return retry_info
        
        # Scan all subdirectories for PDF results
        for pdf_dir in output_dir.iterdir():
            if not pdf_dir.is_dir():
                continue
                
            retry_info["summary"]["total_documents_scanned"] += 1
            pdf_name = pdf_dir.name
            
            # Check for error summary file
            error_file = pdf_dir / "error_summary.json"
            if error_file.exists():
                try:
                    with open(error_file, 'r', encoding='utf-8') as f:
                        error_data = json.load(f)
                    
                    if error_data.get("has_errors", False):
                        retry_info["summary"]["documents_with_errors"] += 1
                        
                        doc_info = {
                            "pdf_name": pdf_name,
                            "pdf_directory": str(pdf_dir),
                            "retry_needed": error_data.get("retry_needed", False),
                            "error_summary": error_data.get("error_summary", {}),
                            "failed_pages": error_data.get("failed_pages", [])
                        }
                        
                        retry_info["failed_documents"].append(doc_info)
                        
                        if doc_info["retry_needed"]:
                            retry_info["summary"]["documents_needing_retry"] += 1
                        
                        retry_info["summary"]["total_failed_pages"] += len(doc_info["failed_pages"])
                        
                        # Add individual failed pages to list
                        for page_num in doc_info["failed_pages"]:
                            retry_info["failed_pages"].append({
                                "pdf_name": pdf_name,
                                "page_number": page_num,
                                "pdf_directory": str(pdf_dir)
                            })
                
                except Exception as e:
                    logger.error(f"Error reading error summary for {pdf_name}: {str(e)}")
            
            # Also check for Vision+Gemini workflow errors
            vision_error_file = pdf_dir / "error_summary.json"
            if not error_file.exists() and vision_error_file.exists():
                # This might be a vision+gemini workflow result
                try:
                    with open(vision_error_file, 'r', encoding='utf-8') as f:
                        error_data = json.load(f)
                    
                    if error_data.get("has_errors", False):
                        retry_info["summary"]["documents_with_errors"] += 1
                        
                        doc_info = {
                            "pdf_name": pdf_name,
                            "pdf_directory": str(pdf_dir),
                            "retry_needed": error_data.get("retry_needed", False),
                            "error_summary": error_data.get("error_summary", {}),
                            "workflow_type": "vision_gemini"
                        }
                        
                        retry_info["failed_documents"].append(doc_info)
                        
                        if doc_info["retry_needed"]:
                            retry_info["summary"]["documents_needing_retry"] += 1
                
                except Exception as e:
                    logger.error(f"Error reading vision error summary for {pdf_name}: {str(e)}")
        
        return retry_info
    
    @staticmethod
    def generate_retry_report(retry_info: Dict[str, Any], output_file: Optional[Path] = None) -> str:
        """
        Generate a human-readable retry report.
        
        Args:
            retry_info: Retry information from scan_output_directory
            output_file: Optional file to save the report
            
        Returns:
            Report as string
        """
        summary = retry_info["summary"]
        
        report_lines = [
            "=== OCR PROCESS RETRY REPORT ===",
            "",
            "SUMMARY:",
            f"  Total documents scanned: {summary['total_documents_scanned']}",
            f"  Documents with errors: {summary['documents_with_errors']}",
            f"  Documents needing retry: {summary['documents_needing_retry']}",
            f"  Total failed pages: {summary['total_failed_pages']}",
            ""
        ]
        
        if retry_info["failed_documents"]:
            report_lines.extend([
                "FAILED DOCUMENTS:",
                "=================="
            ])
            
            for doc in retry_info["failed_documents"]:
                report_lines.extend([
                    f"PDF: {doc['pdf_name']}",
                    f"  Directory: {doc['pdf_directory']}",
                    f"  Retry needed: {doc['retry_needed']}",
                    f"  Workflow type: {doc.get('workflow_type', 'standard')}"
                ])
                
                if "failed_pages" in doc and doc["failed_pages"]:
                    report_lines.append(f"  Failed pages: {', '.join(map(str, doc['failed_pages']))}")
                
                # Summarize error types
                error_summary = doc.get("error_summary", {})
                error_types = []
                for error_category, errors in error_summary.items():
                    if isinstance(errors, list) and errors:
                        for error in errors:
                            if isinstance(error, dict):
                                error_types.append(error.get("error_type", "unknown"))
                    elif isinstance(errors, dict) and errors:
                        error_types.append(errors.get("error_type", "unknown"))
                
                if error_types:
                    unique_error_types = list(set(error_types))
                    report_lines.append(f"  Error types: {', '.join(unique_error_types)}")
                
                report_lines.append("")
        else:
            report_lines.append("No failed documents found.")
        
        report = "\n".join(report_lines)
        
        if output_file:
            try:
                output_file.write_text(report, encoding='utf-8')
                logger.info(f"Retry report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving retry report: {str(e)}")
        
        return report
    
    @staticmethod
    def get_retry_candidates(retry_info: Dict[str, Any], error_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get list of documents that are candidates for retry.
        
        Args:
            retry_info: Retry information from scan_output_directory
            error_types: Optional list of error types to filter by (e.g., ['timeout', 'rate_limit'])
            
        Returns:
            List of documents that should be retried
        """
        candidates = []
        
        for doc in retry_info["failed_documents"]:
            if not doc.get("retry_needed", False):
                continue
            
            # If error types filter is specified, check if document has matching errors
            if error_types:
                error_summary = doc.get("error_summary", {})
                doc_error_types = []
                
                for error_category, errors in error_summary.items():
                    if isinstance(errors, list):
                        for error in errors:
                            if isinstance(error, dict):
                                doc_error_types.append(error.get("error_type", "unknown"))
                    elif isinstance(errors, dict) and errors:
                        doc_error_types.append(errors.get("error_type", "unknown"))
                
                # Check if any of the document's error types match the filter
                if not any(et in doc_error_types for et in error_types):
                    continue
            
            candidates.append(doc)
        
        return candidates
    
    @staticmethod
    def create_retry_script(retry_candidates: List[Dict[str, Any]], 
                          script_path: Path, 
                          workflow_type: str = "simplified") -> None:
        """
        Create a Python script to retry failed documents.
        
        Args:
            retry_candidates: List of documents to retry
            script_path: Path where to save the retry script
            workflow_type: Type of workflow to use ('simplified' or 'vision_gemini')
        """
        script_lines = [
            "#!/usr/bin/env python3",
            '"""Auto-generated script to retry failed OCR processes."""',
            "",
            "import logging",
            "from pathlib import Path",
            "import sys",
            "",
            "# Add the deed_ocr package to path if needed",
            "# sys.path.append('/path/to/your/deed_ocr')",
            "",
            "from deed_ocr.workflow import process_deed_pdf_simple",
            ""
        ]
        
        if workflow_type == "vision_gemini":
            script_lines.extend([
                "from deed_ocr.workflow_vision import process_deed_pdf_vision_gemini",
                ""
            ])
        
        script_lines.extend([
            "# Configure logging",
            "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')",
            "logger = logging.getLogger(__name__)",
            "",
            "def retry_failed_documents():",
            '    """Retry failed OCR documents."""',
            "    failed_count = 0",
            "    success_count = 0",
            "",
            f"    # Total documents to retry: {len(retry_candidates)}",
            ""
        ])
        
        for i, doc in enumerate(retry_candidates):
            pdf_dir = Path(doc["pdf_directory"])
            original_pdf = None
            
            # Try to find the original PDF in the result directory
            for pdf_file in pdf_dir.glob("*.pdf"):
                original_pdf = pdf_file
                break
            
            if not original_pdf:
                script_lines.extend([
                    f"    # Document {i+1}: {doc['pdf_name']} - SKIPPED (original PDF not found)",
                    ""
                ])
                continue
            
            script_lines.extend([
                f"    # Document {i+1}: {doc['pdf_name']}",
                f"    logger.info('Retrying {doc['pdf_name']}...')",
                "    try:",
                f"        pdf_path = Path(r'{original_pdf}')",
                f"        output_dir = Path(r'{pdf_dir.parent}')",
                ""
            ])
            
            if workflow_type == "vision_gemini":
                script_lines.extend([
                    "        result = process_deed_pdf_vision_gemini(",
                    "            pdf_path=pdf_path,",
                    "            output_dir=output_dir,",
                    "            max_retries=5,  # Increased retries for problematic documents",
                    "            retry_delay=10.0  # Longer delay between retries",
                    "        )"
                ])
            else:
                script_lines.extend([
                    "        result = process_deed_pdf_simple(",
                    "            pdf_path=pdf_path,",
                    "            output_dir=output_dir,",
                    "            max_retries=5,  # Increased retries for problematic documents",
                    "            retry_delay=10.0  # Longer delay between retries",
                    "        )"
                ])
            
            script_lines.extend([
                "        ",
                "        if result.has_errors:",
                f"            logger.warning('Document {doc['pdf_name']} still has errors after retry')",
                "            failed_count += 1",
                "        else:",
                f"            logger.info('Document {doc['pdf_name']} processed successfully')",
                "            success_count += 1",
                "            ",
                "    except Exception as e:",
                f"        logger.error('Error retrying {doc['pdf_name']}: {{str(e)}}')",
                "        failed_count += 1",
                "",
                ""
            ])
        
        script_lines.extend([
            "    logger.info(f'Retry completed. Success: {success_count}, Failed: {failed_count}')",
            "",
            "if __name__ == '__main__':",
            "    retry_failed_documents()"
        ])
        
        script_content = "\n".join(script_lines)
        
        try:
            script_path.write_text(script_content, encoding='utf-8')
            # Make script executable on Unix systems
            try:
                script_path.chmod(0o755)
            except:
                pass  # Ignore chmod errors on Windows
            logger.info(f"Retry script created: {script_path}")
        except Exception as e:
            logger.error(f"Error creating retry script: {str(e)}")


def scan_and_report(output_dir: Path, report_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to scan directory and generate report.
    
    Args:
        output_dir: Directory containing OCR results
        report_file: Optional file to save the report
        
    Returns:
        Retry information dictionary
    """
    helper = RetryHelper()
    retry_info = helper.scan_output_directory(output_dir)
    report = helper.generate_retry_report(retry_info, report_file)
    
    print(report)
    return retry_info


def create_retry_script_for_timeouts(output_dir: Path, script_path: Path) -> None:
    """
    Convenience function to create retry script for timeout/network errors.
    
    Args:
        output_dir: Directory containing OCR results  
        script_path: Path where to save the retry script
    """
    helper = RetryHelper()
    retry_info = helper.scan_output_directory(output_dir)
    
    # Filter for timeout and network errors
    candidates = helper.get_retry_candidates(
        retry_info, 
        error_types=['timeout', 'network', 'rate_limit', 'server_error']
    )
    
    helper.create_retry_script(candidates, script_path)
    print(f"Created retry script for {len(candidates)} documents with timeout/network errors: {script_path}") 
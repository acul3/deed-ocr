#!/usr/bin/env python3
"""Example script showing how to identify and retry failed OCR processes."""

import logging
from pathlib import Path

from deed_ocr.utils.retry_helper import scan_and_report, create_retry_script_for_timeouts, RetryHelper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Example usage of retry helper utilities."""
    
    # Example: Scan output directory for failed processes
    output_dir = Path("output_results")  # Change to your actual output directory
    
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return
    
    print("=== SCANNING FOR FAILED PROCESSES ===")
    
    # Scan and generate report
    retry_info = scan_and_report(output_dir, report_file=output_dir / "retry_report.txt")
    
    # Create retry script for timeout/network errors only
    script_path = output_dir / "retry_timeouts.py"
    create_retry_script_for_timeouts(output_dir, script_path)
    
    # Example: Create retry script for all failed documents
    helper = RetryHelper()
    all_candidates = helper.get_retry_candidates(retry_info)
    if all_candidates:
        all_retry_script = output_dir / "retry_all_failed.py"
        helper.create_retry_script(all_candidates, all_retry_script)
        print(f"Created retry script for all {len(all_candidates)} failed documents: {all_retry_script}")
    
    # Example: Create retry script for specific error types
    auth_candidates = helper.get_retry_candidates(retry_info, error_types=['authentication', 'credentials'])
    if auth_candidates:
        auth_retry_script = output_dir / "retry_auth_errors.py"
        helper.create_retry_script(auth_candidates, auth_retry_script)
        print(f"Created retry script for {len(auth_candidates)} authentication errors: {auth_retry_script}")
    
    print("\n=== SUMMARY ===")
    summary = retry_info["summary"]
    print(f"Total documents scanned: {summary['total_documents_scanned']}")
    print(f"Documents with errors: {summary['documents_with_errors']}")
    print(f"Documents needing retry: {summary['documents_needing_retry']}")
    
    if summary['documents_needing_retry'] > 0:
        print(f"\nYou can now run the generated retry scripts:")
        if script_path.exists():
            print(f"  python {script_path}")
        if 'all_retry_script' in locals() and all_retry_script.exists():
            print(f"  python {all_retry_script}")


if __name__ == "__main__":
    main() 
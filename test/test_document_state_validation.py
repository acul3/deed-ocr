#!/usr/bin/env python3
"""
Test script to demonstrate the new document state functionality in TRS validation.
"""

import json
import logging
from pathlib import Path
from deed_ocr.utils.trs_validator import TRSValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def test_document_state_validation():
    """Test TRS validation with and without document state."""
    
    logger.info("=== Testing Document State in TRS Validation ===")
    
    # Create a test TRS entry that could be in multiple states
    test_trs = {
        "Township": "7N",
        "Range": "60W", 
        "Section": "32",
        "TRS": "T 7 N, R 60 W S32",
        "County": "Weld"  # This could be in Colorado or potentially other states
    }
    
    # Initialize validator (without geodatabase for this demo)
    validator = TRSValidator()
    
    # Test 1: Without document state (uses county-based lookup)
    logger.info("\n--- Test 1: Without Document State ---")
    logger.info(f"Testing TRS: {test_trs}")
    
    # Simulate getting state from county
    state_from_county = validator.get_state_from_county("Weld")
    logger.info(f"State from county lookup: Weld -> {state_from_county}")
    
    # Test 2: With document state (prioritizes document state)
    logger.info("\n--- Test 2: With Document State ---")
    document_state = "Colorado"
    state_from_document = validator.get_state_abbreviation_from_name(document_state)
    logger.info(f"State from document: {document_state} -> {state_from_document}")
    
    # Test 3: Demonstrate the full workflow
    logger.info("\n--- Test 3: Full Workflow with Real Data ---")
    
    # Load a real final result to test
    final_result_path = Path("results_final24/1650511-1/final_result.json")
    
    if final_result_path.exists():
        with open(final_result_path, 'r', encoding='utf-8') as f:
            final_result = json.load(f)
        
        # Show the extracted document state
        document_state = final_result.get("details", {}).get("state")
        logger.info(f"Document state extracted: {document_state}")
        
        # Show TRS details
        trs_details = final_result.get("TRS_details", [])
        logger.info(f"Found {len(trs_details)} TRS entries:")
        for i, trs in enumerate(trs_details, 1):
            county = trs.get("County", "Unknown")
            logger.info(f"  {i}. County: {county}")
            
            # Show what would happen with vs without document state
            if document_state:
                doc_state_abbr = validator.get_state_abbreviation_from_name(document_state)
                logger.info(f"     With document state: {document_state} -> {doc_state_abbr}")
            
            county_state_abbr = validator.get_state_from_county(county)
            logger.info(f"     With county lookup: {county} -> {county_state_abbr}")
    else:
        logger.warning(f"Final result file not found: {final_result_path}")
    
    logger.info("\n=== Test Complete ===")
    logger.info("The TRS validator now:")
    logger.info("1. First checks for document state in details['state']")
    logger.info("2. Gets state abbreviation from the full state name")
    logger.info("3. Falls back to county-based state lookup if no document state")
    logger.info("4. This ensures more accurate validation when state is known")

if __name__ == "__main__":
    test_document_state_validation() 
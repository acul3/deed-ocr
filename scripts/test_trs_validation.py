#!/usr/bin/env python3
"""
Test script for TRS validation functionality.

This script demonstrates how to validate TRS details from existing final result files
and create validation Excel reports.
"""

import json
import logging
from pathlib import Path
import sys

# Add the parent directory to the path so we can import deed_ocr modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deed_ocr.utils.trs_validator import (
    TRSValidator, 
    create_trs_validation_excel, 
    validate_trs_from_final_result
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def test_trs_validation_from_final_result():
    """Test TRS validation using an existing final result file."""
    
    # Configuration - Uses environment variables or defaults
    import os
    FINAL_RESULT_PATH = Path("../results_oil_deed/1170220-1/final_result.json")
    GEODATABASE_PATH = os.getenv("GEODATABASE_PATH", "/Users/samsulrahmadani/Downloads/geojson/ilmocplss.gdb")
    COUNTIES_JSON_PATH = os.getenv("COUNTIES_JSON_PATH", "../counties_list.json")  # Optional
    OUTPUT_EXCEL_PATH = Path("../test_trs_validation_results.xlsx")
    
    logger.info("=== TRS Validation Test ===")
    
    # Check if final result file exists
    if not FINAL_RESULT_PATH.exists():
        logger.error(f"Final result file not found: {FINAL_RESULT_PATH}")
        logger.info("Available final result files:")
        
        # Find available final result files
        results_dirs = [
            Path("../results_oil_deed"),
            Path("../results_stage_2"),
            Path("../results_stage_24"),
            Path("../results_oil2")
        ]
        
        for results_dir in results_dirs:
            if results_dir.exists():
                for final_result_file in results_dir.rglob("final_result.json"):
                    logger.info(f"  {final_result_file}")
        
        return False
    
    # Check if geodatabase exists
    if not Path(GEODATABASE_PATH).exists():
        logger.error(f"Geodatabase not found: {GEODATABASE_PATH}")
        logger.info("TRS validation requires a PLSS geodatabase file (.gdb)")
        logger.info("Please update the GEODATABASE_PATH variable with the correct path")
        return False
    
    logger.info(f"Using final result: {FINAL_RESULT_PATH}")
    logger.info(f"Using geodatabase: {GEODATABASE_PATH}")
    
    try:
        # Load and display the TRS details from the final result
        with open(FINAL_RESULT_PATH, 'r', encoding='utf-8') as f:
            final_result = json.load(f)
        
        trs_details_list = final_result.get("TRS_details", [])
        logger.info(f"Found {len(trs_details_list)} TRS details entries:")
        
        for i, trs_details in enumerate(trs_details_list, 1):
            logger.info(f"  {i}. {trs_details}")
        
        # Perform TRS validation
        logger.info("\nStarting TRS validation...")
        validation_results = validate_trs_from_final_result(
            FINAL_RESULT_PATH,
            GEODATABASE_PATH,
            COUNTIES_JSON_PATH if Path(COUNTIES_JSON_PATH).exists() else None,
            OUTPUT_EXCEL_PATH
        )
        
        # Display validation results
        logger.info("\n=== Validation Results ===")
        valid_count = 0
        for i, (trs_details, validation_result) in enumerate(zip(trs_details_list, validation_results), 1):
            status = "✅ VALID" if validation_result.is_valid else "❌ INVALID"
            logger.info(f"{i}. {status}: {trs_details.get('TRS', 'N/A')}")
            
            if validation_result.log_messages:
                for msg in validation_result.log_messages:
                    logger.info(f"   Log: {msg}")
            
            if validation_result.error_message:
                logger.info(f"   Error: {validation_result.error_message}")
            
            if validation_result.is_valid:
                valid_count += 1
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Total TRS entries: {len(validation_results)}")
        logger.info(f"Valid entries: {valid_count}")
        logger.info(f"Invalid entries: {len(validation_results) - valid_count}")
        logger.info(f"Validation Excel saved to: {OUTPUT_EXCEL_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during TRS validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_trs_validation():
    """Test TRS validation with manually created TRS data."""
    
    logger.info("\n=== Manual TRS Validation Test ===")
    
    # Sample TRS data for testing
    test_trs_data = [
        {
            "Township": "5 North",
            "Range": "66 West",
            "Section": "29",
            "TRS": "T5N R66W S29",
            "County": "Weld"
        },
        {
            "Township": "5 North", 
            "Range": "66 West",
            "Section": "7",
            "TRS": "T5N R66W S7",
            "County": "Weld"
        },
        {
            "Township": "7N",
            "Range": "60W", 
            "Section": "32",
            "TRS": "T 7 N, R 60 W S32",
            "County": "Weld, Morgan"  # Multiple counties - should validate for both
        },
        {
            "Township": "6N",
            "Range": "61W",
            "Section": "12", 
            "TRS": "T 6 N, R 61 W S12",
            "County": "Weld, Morgan County"  # Multiple counties with "County" suffix
        },
        {
            "Township": "99 North",  # Invalid - should not exist
            "Range": "99 West",
            "Section": "99",
            "TRS": "T99N R99W S99",
            "County": "InvalidCounty"
        }
    ]
    
    import os
    GEODATABASE_PATH = os.getenv("GEODATABASE_PATH", "/Users/samsulrahmadani/Downloads/geojson/ilmocplss.gdb")
    OUTPUT_EXCEL_PATH = Path("../test_manual_trs_validation.xlsx")
    
    if not Path(GEODATABASE_PATH).exists():
        logger.error(f"Geodatabase not found: {GEODATABASE_PATH}")
        return False
    
    try:
        # Initialize validator
        validator = TRSValidator(GEODATABASE_PATH)
        
        # Validate each TRS entry
        validation_results = []
        logger.info("Validating test TRS data...")
        
        for i, trs_data in enumerate(test_trs_data, 1):
            logger.info(f"\n{i}. Testing: {trs_data['TRS']} in {trs_data['County']} County")
            result = validator.validate_trs(trs_data)
            validation_results.append(result)
            
            status = "✅ VALID" if result.is_valid else "❌ INVALID"
            logger.info(f"   Result: {status}")
            
            for msg in result.log_messages:
                logger.info(f"   Log: {msg}")
        
        # Create Excel report
        create_trs_validation_excel(
            test_trs_data,
            validation_results,
            OUTPUT_EXCEL_PATH,
            "Manual_Test"
        )
        
        logger.info(f"\nManual validation Excel saved to: {OUTPUT_EXCEL_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"Error during manual TRS validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run TRS validation tests."""
    
    logger.info("Starting TRS validation tests...")
    
    # Test 1: Validation from existing final result
    success1 = test_trs_validation_from_final_result()
    
    # Test 2: Manual validation test
    success2 = test_manual_trs_validation()
    
    if success1 or success2:
        logger.info("\n✅ TRS validation tests completed successfully!")
        if success1:
            logger.info("   - Final result validation: ✅")
        if success2:
            logger.info("   - Manual validation: ✅")
    else:
        logger.error("\n❌ All TRS validation tests failed!")
        logger.info("Make sure you have:")
        logger.info("  1. A PLSS geodatabase file (.gdb)")
        logger.info("  2. GeoPandas installed (pip install geopandas)")
        logger.info("  3. Existing final result files to test")


if __name__ == "__main__":
    main() 
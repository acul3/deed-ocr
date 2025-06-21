#!/usr/bin/env python3
"""
Demo script showing multi-county TRS validation functionality.

This demonstrates how the TRS validator handles cases where TRS entries
contain multiple counties separated by commas (e.g., "Weld, Morgan").
"""

import logging
from pathlib import Path
import sys

# Add the parent directory to the path so we can import deed_ocr modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deed_ocr.utils.trs_validator import TRSValidator, create_trs_validation_excel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def demo_multi_county_validation():
    """Demonstrate multi-county TRS validation."""
    
    logger.info("=== Multi-County TRS Validation Demo ===")
    
    # Configuration
    import os
    geodatabase_path = os.getenv("GEODATABASE_PATH")
    counties_json_path = os.getenv("COUNTIES_JSON_PATH")
    
    if not geodatabase_path:
        logger.error("GEODATABASE_PATH environment variable not set")
        logger.info("Please set GEODATABASE_PATH in your .env file")
        return False
    
    if not Path(geodatabase_path).exists():
        logger.error(f"Geodatabase not found: {geodatabase_path}")
        return False
    
    logger.info(f"Using geodatabase: {geodatabase_path}")
    
    try:
        # Initialize validator
        validator = TRSValidator(geodatabase_path, counties_json_path)
        
        # Sample TRS data with multi-county scenarios
        test_cases = [
            {
                "name": "Single County - Valid",
                "data": {
                    "Township": "5 North",
                    "Range": "66 West",
                    "Section": "29",
                    "TRS": "T5N R66W S29", 
                    "County": "Weld"
                }
            },
            {
                "name": "Two Counties - Both Valid",
                "data": {
                    "Township": "7N",
                    "Range": "60W",
                    "Section": "32",
                    "TRS": "T 7 N, R 60 W S32",
                    "County": "Weld, Morgan"
                }
            },
            {
                "name": "Two Counties with 'County' Suffix",
                "data": {
                    "Township": "6N",
                    "Range": "61W", 
                    "Section": "12",
                    "TRS": "T 6 N, R 61 W S12",
                    "County": "Weld County, Morgan County"
                }
            },
            {
                "name": "Mixed Valid/Invalid Counties",
                "data": {
                    "Township": "5 North",
                    "Range": "66 West",
                    "Section": "29", 
                    "TRS": "T5N R66W S29",
                    "County": "Weld, InvalidCounty"
                }
            },
            {
                "name": "All Invalid Counties",
                "data": {
                    "Township": "99 North",
                    "Range": "99 West",
                    "Section": "99",
                    "TRS": "T99N R99W S99",
                    "County": "InvalidCounty1, InvalidCounty2"
                }
            },
            {
                "name": "Three Counties - Mixed Results",
                "data": {
                    "Township": "1 North",
                    "Range": "68 West",
                    "Section": "1",
                    "TRS": "T1N R68W S1",
                    "County": "Weld, Morgan, InvalidCounty"
                }
            }
        ]
        
        logger.info("\n🧪 Testing Multi-County Validation Cases:")
        logger.info("=" * 60)
        
        validation_results = []
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n{i}. {test_case['name']}")
            logger.info(f"   County: {test_case['data']['County']}")
            logger.info(f"   TRS: {test_case['data']['TRS']}")
            
            # Validate
            result = validator.validate_trs(test_case['data'])
            validation_results.append(result)
            
            # Show result
            status = "✅ VALID" if result.is_valid else "❌ INVALID"
            logger.info(f"   Result: {status}")
            
            # Show detailed logs
            for log_msg in result.log_messages:
                logger.info(f"     📝 {log_msg}")
        
        # Create Excel report to demonstrate output format
        output_excel = Path("../demo_multi_county_validation.xlsx")
        create_trs_validation_excel(
            [case['data'] for case in test_cases],
            validation_results,
            output_excel,
            "Multi_County_Demo"
        )
        
        logger.info(f"\n📊 Excel report saved to: {output_excel}")
        
        # Summary
        valid_count = sum(1 for result in validation_results if result.is_valid)
        total_count = len(validation_results)
        
        logger.info(f"\n📈 Summary:")
        logger.info(f"   Total test cases: {total_count}")
        logger.info(f"   Valid cases: {valid_count}")
        logger.info(f"   Invalid cases: {total_count - valid_count}")
        
        logger.info(f"\n💡 Key Points:")
        logger.info(f"   • Multi-county entries are validated against ALL counties")
        logger.info(f"   • TRS is VALID if ANY county validates successfully")
        logger.info(f"   • Handles 'County' suffix automatically")
        logger.info(f"   • Detailed logs show per-county validation results")
        logger.info(f"   • Excel output includes county count and detailed status")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the multi-county validation demo."""
    
    logger.info("🚀 Starting multi-county TRS validation demo...")
    
    success = demo_multi_county_validation()
    
    if success:
        logger.info("\n✅ Demo completed successfully!")
        logger.info("Multi-county TRS validation is working correctly.")
    else:
        logger.error("\n❌ Demo failed!")
        logger.info("Make sure you have:")
        logger.info("  1. GEODATABASE_PATH set in .env file")
        logger.info("  2. GeoPandas installed (pip install geopandas)")
        logger.info("  3. Valid PLSS geodatabase file")


if __name__ == "__main__":
    main() 
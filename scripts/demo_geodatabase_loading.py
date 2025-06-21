#!/usr/bin/env python3
"""
Demo script showing that the geodatabase is loaded only once at startup.

This demonstrates the performance improvement where the PLSS geodatabase
is loaded once during workflow initialization, not per document.
"""

import logging
import time
from pathlib import Path
import sys

# Add the parent directory to the path so we can import deed_ocr modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deed_ocr.workflow import SimplifiedDeedOCRWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def demo_geodatabase_loading():
    """Demonstrate that geodatabase is loaded once at initialization."""
    
    logger.info("=== Geodatabase Loading Demo ===")
    
    # Configuration
    import os
    geodatabase_path = os.getenv("GEODATABASE_PATH")
    counties_json_path = os.getenv("COUNTIES_JSON_PATH")
    
    if not geodatabase_path:
        logger.error("GEODATABASE_PATH environment variable not set")
        logger.info("Please set GEODATABASE_PATH in your .env file or environment")
        return False
    
    if not Path(geodatabase_path).exists():
        logger.error(f"Geodatabase not found: {geodatabase_path}")
        return False
    
    logger.info(f"Using geodatabase: {geodatabase_path}")
    if counties_json_path:
        logger.info(f"Using counties file: {counties_json_path}")
    
    # Time the workflow initialization
    logger.info("\n⏱️  Timing workflow initialization (geodatabase loading)...")
    start_time = time.time()
    
    try:
        workflow = SimplifiedDeedOCRWorkflow(
            geodatabase_path=geodatabase_path,
            counties_json_path=counties_json_path,
            enable_trs_validation=True
        )
        
        initialization_time = time.time() - start_time
        logger.info(f"✅ Workflow initialized in {initialization_time:.2f} seconds")
        
        if workflow.trs_validator:
            logger.info("🗺️  TRS validator ready for use")
            logger.info("📊 Geodatabase is now loaded in memory and ready for validation")
            
            # Demo: Multiple validation calls should be fast now
            logger.info("\n⚡ Testing multiple TRS validations (should be fast)...")
            
            test_trs_data = [
                {
                    "Township": "5 North",
                    "Range": "66 West", 
                    "Section": "29",
                    "County": "Weld"
                },
                {
                    "Township": "5 North",
                    "Range": "66 West",
                    "Section": "7", 
                    "County": "Weld"
                },
                {
                    "Township": "1 North",
                    "Range": "68 West",
                    "Section": "1",
                    "County": "Weld"
                }
            ]
            
            total_validation_time = 0
            for i, trs_data in enumerate(test_trs_data, 1):
                start_validation = time.time()
                result = workflow.trs_validator.validate_trs(trs_data)
                validation_time = time.time() - start_validation
                total_validation_time += validation_time
                
                status = "✅ Valid" if result.is_valid else "❌ Invalid"
                trs_str = f"T{trs_data['Township'].replace(' North', 'N').replace(' South', 'S')} R{trs_data['Range'].replace(' West', 'W').replace(' East', 'E')} S{trs_data['Section']}"
                logger.info(f"  {i}. {trs_str} - {status} ({validation_time:.3f}s)")
            
            avg_validation_time = total_validation_time / len(test_trs_data)
            logger.info(f"\n📈 Average validation time per TRS: {avg_validation_time:.3f} seconds")
            logger.info("💡 Subsequent validations are fast because geodatabase is already loaded!")
            
        else:
            logger.warning("❌ TRS validator not initialized")
            return False
        
        logger.info(f"\n🎯 Performance Benefits:")
        logger.info(f"  • Geodatabase loaded once at startup: {initialization_time:.2f}s")
        logger.info(f"  • Fast validation per TRS entry: ~{avg_validation_time:.3f}s")
        logger.info(f"  • No reload time for batch processing")
        logger.info(f"  • Memory efficient with single geodatabase instance")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the geodatabase loading demo."""
    
    logger.info("🚀 Starting geodatabase loading performance demo...")
    
    success = demo_geodatabase_loading()
    
    if success:
        logger.info("\n✅ Demo completed successfully!")
        logger.info("The geodatabase is loaded once at startup for optimal performance.")
    else:
        logger.error("\n❌ Demo failed!")
        logger.info("Make sure you have:")
        logger.info("  1. GEODATABASE_PATH set in .env file")
        logger.info("  2. GeoPandas installed (pip install geopandas)")
        logger.info("  3. Valid PLSS geodatabase file")


if __name__ == "__main__":
    main() 
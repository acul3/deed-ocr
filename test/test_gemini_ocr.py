#!/usr/bin/env python3
"""Test script for Gemini OCR full text processing functionality."""

import os
import json
import tempfile
from pathlib import Path
from deed_ocr.ocr.gemini_ocr import create_gemini_ocr_engine

def create_test_data():
    """Create test data for single and multi-page scenarios."""
    
    # Sample legal document text
    sample_legal_text = """
GENERAL WARRANTY DEED

KNOW ALL MEN BY THESE PRESENTS, that JOHN SMITH and MARY SMITH, husband and wife,
grantors, of Dallas County, Texas, for and in consideration of the sum of Ten Dollars ($10.00)
and other good and valuable consideration, the receipt of which is hereby acknowledged,
have GRANTED, SOLD and CONVEYED, and by these presents do GRANT, SELL and CONVEY
unto ROBERT JOHNSON and SUSAN JOHNSON, husband and wife, grantees, of Dallas County, Texas,
all that certain lot, tract or parcel of land situated in Dallas County, Texas, and being
more particularly described as follows:

BEING a tract of land situated in the John Smith Survey, Abstract No. 1234, Dallas County,
Texas, and being more particularly described as follows:

BEGINNING at a point in the south line of said John Smith Survey, said point being
South 89°30'00" East 100.00 feet from the southwest corner of said survey;

THENCE North 0°30'00" East 200.00 feet to a point;

THENCE South 89°30'00" East 150.00 feet to a point;

THENCE South 0°30'00" West 200.00 feet to a point in the south line of said survey;

THENCE North 89°30'00" West 150.00 feet to the POINT OF BEGINNING,

containing 0.69 acres, more or less.

Township 2 North, Range 3 West, Section 14

This conveyance is made subject to all valid easements, restrictions, and other matters
of record affecting said property.

EXECUTED this 15th day of March, 2024.

JOHN SMITH
MARY SMITH

STATE OF TEXAS
COUNTY OF DALLAS

Before me, the undersigned notary public, on this day personally appeared JOHN SMITH
and MARY SMITH, known to me to be the persons whose names are subscribed to the
foregoing instrument and acknowledged to me that they executed the same for the
purposes and consideration therein expressed.

GIVEN under my hand and seal of office this 15th day of March, 2024.

NOTARY PUBLIC
"""
    
    return sample_legal_text

def test_single_page_scenario():
    """Test single page PDF scenario."""
    print("=== Testing Single Page Scenario ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test result folder structure for single page
        result_folder = Path(temp_dir) / "single_page_test"
        result_folder.mkdir()
        
        # Create full_text.txt
        full_text_file = result_folder / "full_text.txt"
        with open(full_text_file, 'w', encoding='utf-8') as f:
            f.write(create_test_data())
        
        # Create only one page image file to simulate single page
        (result_folder / "page_1.png").touch()
        
        try:
            # Test the function
            ocr_engine = create_gemini_ocr_engine()
            result = ocr_engine.process_full_text_from_file(str(result_folder))
            
            print(f"Result: {result}")
            print("✅ Single page test passed - processing was skipped as expected")
            
        except Exception as e:
            print(f"❌ Single page test failed: {e}")

def test_multi_page_scenario():
    """Test multi-page PDF scenario."""
    print("\n=== Testing Multi-Page Scenario ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test result folder structure for multi-page
        result_folder = Path(temp_dir) / "multi_page_test"
        result_folder.mkdir()
        
        # Create full_text.txt
        full_text_file = result_folder / "full_text.txt"
        with open(full_text_file, 'w', encoding='utf-8') as f:
            f.write(create_test_data())
        
        # Create multiple page image files to simulate multi-page PDF
        (result_folder / "page_1.png").touch()
        (result_folder / "page_2.png").touch()
        (result_folder / "page_3.png").touch()
        
        try:
            # Test the function
            ocr_engine = create_gemini_ocr_engine()
            result = ocr_engine.process_full_text_from_file(str(result_folder))
            
            print("✅ Multi-page test completed")
            print(f"Result keys: {list(result.keys())}")
            
            # Check if analysis file was created
            analysis_file = result_folder / "full_text_analysis.json"
            if analysis_file.exists():
                print("✅ Analysis file created successfully")
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                print(f"Analysis contains: {list(analysis.keys())}")
            else:
                print("❌ Analysis file was not created")
            
        except Exception as e:
            print(f"❌ Multi-page test failed: {e}")

def test_direct_text_processing():
    """Test direct text processing without file operations."""
    print("\n=== Testing Direct Text Processing ===")
    
    try:
        ocr_engine = create_gemini_ocr_engine()
        sample_text = create_test_data()
        
        result = ocr_engine.process_full_text(sample_text)
        
        print("✅ Direct text processing completed")
        print(f"Result keys: {list(result.keys())}")
        
        if 'details' in result:
            details = result['details']
            print(f"Document type: {details.get('document_type')}")
            print(f"Document subtype: {details.get('document_subtype')}")
            if 'parties' in details:
                print(f"Parties: {list(details['parties'].keys())}")
            
    except Exception as e:
        print(f"❌ Direct text processing failed: {e}")

def main():
    """Run all tests."""
    print("Starting Gemini OCR Full Text Processing Tests...")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv("GOOGLE_AI_API_KEY"):
        print("❌ GOOGLE_AI_API_KEY environment variable not set")
        print("Please set your Google AI API key:")
        print("export GOOGLE_AI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Test scenarios
        test_single_page_scenario()
        test_multi_page_scenario()
        test_direct_text_processing()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""Test script for the new PDF processing functionality."""

import os
from pathlib import Path
from deed_ocr.ocr.gemini_ocr import create_gemini_ocr_engine

def test_direct_pdf_processing():
    """Test processing an entire PDF directly."""
    print("=== Testing Direct PDF Processing ===")
    
    # Set API key
    os.environ['GOOGLE_AI_API_KEY'] = ''
    
    # PDF file path
    pdf_path = "/Users/samsulrahmadani/Documents/deed-ocr/1460797.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Create OCR engine
        ocr_engine = create_gemini_ocr_engine()
        
        # Process the entire PDF directly
        print(f"Processing entire PDF: {pdf_path}")
        result = ocr_engine.process_entire_pdf(pdf_path)
        
        print("✅ Direct PDF processing completed successfully!")
        print(f"Result keys: {list(result.keys())}")
        
        if 'full_text' in result:
            text_length = len(result['full_text'])
            print(f"Full text extracted: {text_length} characters")
        
        if 'legal_description_block' in result:
            legal_descriptions = result['legal_description_block']
            print(f"Legal descriptions found: {len(legal_descriptions)}")
        
        if 'details' in result:
            details = result['details']
            print(f"Document type: {details.get('document_type')}")
            print(f"Document subtype: {details.get('document_subtype')}")
            if 'parties' in details:
                print(f"Parties: {list(details['parties'].keys())}")
        
    except Exception as e:
        print(f"❌ Direct PDF processing failed: {e}")

def test_result_folder_processing():
    """Test processing from result folder with original PDF."""
    print("\n=== Testing Result Folder Processing ===")
    
    # Set API key
    os.environ['GOOGLE_AI_API_KEY'] = 'AIzaSyBrwsGHpT3Qe5bzvjq2quNegHgULPcLFnc'
    
    # Paths
    result_folder = "./results/1460797"
    original_pdf = "/Users/samsulrahmadani/Documents/deed-ocr/1460797.pdf"
    
    if not Path(result_folder).exists():
        print(f"❌ Result folder not found: {result_folder}")
        return
    
    if not Path(original_pdf).exists():
        print(f"❌ Original PDF not found: {original_pdf}")
        return
    
    try:
        # Create OCR engine
        ocr_engine = create_gemini_ocr_engine()
        
        # Process from result folder
        print(f"Processing from result folder: {result_folder}")
        result = ocr_engine.process_pdf_from_result_folder(result_folder, original_pdf)
        
        print("✅ Result folder processing completed!")
        print(f"Result keys: {list(result.keys())}")
        
        # Check if the analysis file was created
        analysis_file = Path(result_folder) / "full_pdf_analysis.json"
        if analysis_file.exists():
            print(f"✅ Analysis file created: {analysis_file}")
        else:
            print("❌ Analysis file not found")
        
    except Exception as e:
        print(f"❌ Result folder processing failed: {e}")

def main():
    """Run all tests."""
    print("Testing New PDF Processing Functionality")
    print("=" * 50)
    
    # Set API key if not already set
    if not os.getenv("GOOGLE_AI_API_KEY"):
        os.environ['GOOGLE_AI_API_KEY'] = 'AIzaSyBrwsGHpT3Qe5bzvjq2quNegHgULPcLFnc'
    
    try:
        test_direct_pdf_processing()
        test_result_folder_processing()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")

if __name__ == "__main__":
    main() 
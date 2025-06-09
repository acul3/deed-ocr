#!/usr/bin/env python3
"""Test script for the simplified deed OCR workflow."""

import logging
import os
from pathlib import Path

from deed_ocr.workflow import process_deed_pdf_simple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test the simplified workflow with the sample PDF."""
    
    # Set your API key here or via environment variable
    api_key = "AIzaSyBrwsGHpT3Qe5bzvjq2quNegHgULPcLFnc"  # Your provided API key
    # Alternatively, you can set the GOOGLE_AI_API_KEY environment variable
    
    # Input PDF file
    pdf_path = Path("3917312-1.pdf")
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    # Output directory
    output_dir = Path("output")
    
    print("Starting simplified deed OCR workflow...")
    print(f"Processing: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    try:
        # Run the workflow
        result = process_deed_pdf_simple(
            pdf_path=pdf_path,
            api_key=api_key,
            output_dir=output_dir,
            dpi=300  # High quality images
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print(f"Source PDF: {result.source_pdf}")
        print(f"Total pages processed: {result.total_pages}")
        print(f"Legal descriptions found: {len(result.all_legal_descriptions)}")
        
        if result.all_legal_descriptions:
            print("\nLegal Descriptions:")
            for i, desc in enumerate(result.all_legal_descriptions, 1):
                print(f"  {i}. {desc[:100]}{'...' if len(desc) > 100 else ''}")
        
        print(f"\nCombined text length: {len(result.combined_full_text)} characters")
        print(f"Details extracted: {len(result.all_details)} items")
        
        if result.all_details:
            print("\nExtracted Details:")
            for key, value in result.all_details.items():
                print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        
        print(f"\nResults saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error running workflow: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
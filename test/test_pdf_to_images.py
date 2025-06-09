#!/usr/bin/env python3
"""Test script for PDF to images conversion."""

import os
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import json
from datetime import datetime


def convert_pdf_to_images(pdf_path, output_folder, dpi=200, format='PNG'):
    """
    Convert a PDF file to individual page images.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Directory to save images and PDF
        dpi (int): Resolution for image conversion (default: 200)
        format (str): Image format (PNG, JPEG, etc.)
    
    Returns:
        dict: Results including paths and metadata
    """
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output directory structure
    output_folder.mkdir(parents=True, exist_ok=True)
    images_folder = output_folder / "images"
    images_folder.mkdir(exist_ok=True)
    
    print(f"Converting PDF: {pdf_path.name}")
    print(f"Output folder: {output_folder}")
    print(f"DPI: {dpi}, Format: {format}")
    
    try:
        # Convert PDF to images
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        image_paths = []
        
        for i, page in enumerate(pages, 1):
            # Save each page as an image
            image_filename = f"page_{i:03d}.{format.lower()}"
            image_path = images_folder / image_filename
            
            page.save(str(image_path), format)
            image_paths.append(str(image_path))
            print(f"  ✅ Saved page {i}: {image_filename}")
        
        # Copy original PDF to output folder
        pdf_copy_path = output_folder / pdf_path.name
        shutil.copy2(str(pdf_path), str(pdf_copy_path))
        print(f"  ✅ Copied original PDF: {pdf_path.name}")
        
        # Create metadata file
        metadata = {
            "original_pdf": str(pdf_path),
            "pdf_name": pdf_path.name,
            "conversion_date": datetime.now().isoformat(),
            "total_pages": len(pages),
            "dpi": dpi,
            "image_format": format,
            "image_paths": image_paths,
            "pdf_copy_path": str(pdf_copy_path),
            "output_folder": str(output_folder)
        }
        
        metadata_path = output_folder / "conversion_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Created metadata file: conversion_metadata.json")
        
        return {
            "success": True,
            "total_pages": len(pages),
            "image_paths": image_paths,
            "pdf_copy_path": str(pdf_copy_path),
            "metadata_path": str(metadata_path),
            "output_folder": str(output_folder)
        }
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def test_pdf_conversion_single():
    """Test converting a single PDF file."""
    print("=== Testing Single PDF Conversion ===")
    
    # Use one of the existing PDF files
    pdf_path = "1460797.pdf"
    output_folder = f"pdf_images_output/{Path(pdf_path).stem}"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    result = convert_pdf_to_images(pdf_path, output_folder)
    
    if result["success"]:
        print(f"✅ Conversion successful!")
        print(f"   Total pages: {result['total_pages']}")
        print(f"   Output folder: {result['output_folder']}")
        print(f"   Images saved to: {Path(result['output_folder']) / 'images'}")
    else:
        print(f"❌ Conversion failed: {result['error']}")


def test_pdf_conversion_multiple():
    """Test converting multiple PDF files."""
    print("\n=== Testing Multiple PDF Conversion ===")
    
    # List of PDF files to convert
    pdf_files = ["1460797.pdf", "3917312-1.pdf"]
    base_output_folder = "pdf_images_output"
    
    results = []
    
    for pdf_file in pdf_files:
        if not Path(pdf_file).exists():
            print(f"❌ PDF file not found: {pdf_file}")
            continue
        
        output_folder = f"{base_output_folder}/{Path(pdf_file).stem}"
        print(f"\nProcessing: {pdf_file}")
        
        result = convert_pdf_to_images(pdf_file, output_folder)
        results.append({"pdf": pdf_file, "result": result})
    
    # Summary
    print(f"\n=== Conversion Summary ===")
    successful = sum(1 for r in results if r["result"]["success"])
    total = len(results)
    print(f"Successful conversions: {successful}/{total}")
    
    for r in results:
        status = "✅" if r["result"]["success"] else "❌"
        print(f"  {status} {r['pdf']}")


def test_high_resolution_conversion():
    """Test converting PDF with high resolution."""
    print("\n=== Testing High Resolution Conversion ===")
    
    pdf_path = "1460797.pdf"
    output_folder = f"pdf_images_output/{Path(pdf_path).stem}_high_res"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Convert with higher DPI
    result = convert_pdf_to_images(pdf_path, output_folder, dpi=300, format='PNG')
    
    if result["success"]:
        print(f"✅ High resolution conversion successful!")
        print(f"   Total pages: {result['total_pages']}")
        print(f"   Output folder: {result['output_folder']}")
        
        # Check file sizes
        images_folder = Path(result['output_folder']) / 'images'
        total_size = sum(f.stat().st_size for f in images_folder.glob('*.png'))
        print(f"   Total image size: {total_size / (1024*1024):.2f} MB")
    else:
        print(f"❌ High resolution conversion failed: {result['error']}")


def cleanup_test_outputs():
    """Clean up test output folders."""
    print("\n=== Cleaning Up Test Outputs ===")
    
    output_folder = Path("pdf_images_output")
    if output_folder.exists():
        try:
            shutil.rmtree(output_folder)
            print("✅ Cleaned up test output folder")
        except Exception as e:
            print(f"❌ Error cleaning up: {e}")
    else:
        print("ℹ️  No test output folder to clean up")


def main():
    """Run all PDF to images tests."""
    print("PDF to Images Conversion Test Suite")
    print("=" * 50)
    
    try:
        # Run tests
        test_pdf_conversion_single()
        test_pdf_conversion_multiple()
        test_high_resolution_conversion()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
        # Ask user about cleanup
        print("\nTest output folder 'pdf_images_output' has been created.")
        print("To clean up, run: python test_pdf_to_images.py --cleanup")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup_test_outputs()
    else:
        main() 
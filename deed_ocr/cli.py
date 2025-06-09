"""Command-line interface for deed-ocr."""

import logging
from pathlib import Path
from typing import List

import click
from dotenv import load_dotenv

from deed_ocr.workflow import process_deed_pdf_simple
from deed_ocr.workflow_vision import process_deed_pdf_vision_gemini

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def find_pdf_files(input_path: Path, recursive: bool = False) -> List[Path]:
    """Find PDF files in the given path."""
    pdf_files = []
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        pdf_files.append(input_path)
    elif input_path.is_dir():
        if recursive:
            pdf_files.extend(input_path.rglob('*.pdf'))
            pdf_files.extend(input_path.rglob('*.PDF'))
        else:
            pdf_files.extend(input_path.glob('*.pdf'))
            pdf_files.extend(input_path.glob('*.PDF'))
    
    return sorted(pdf_files)


@click.command()
@click.option(
    "--input", "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input PDF file or folder containing PDFs",
)
@click.option(
    "--output", "-o",
    "out_dir", 
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for results",
)
@click.option(
    "--recursive", "-r",
    is_flag=True,
    help="Process PDFs in subdirectories recursively (only when input is a folder)",
)
@click.option(
    "--ocr-engine",
    type=click.Choice(["gemini", "vision-gemini"]),  # TODO: Add support for ["textract", "local"]
    default="gemini",
    help="OCR engine to use: 'gemini' (direct image-to-structure) or 'vision-gemini' (Google Vision OCR + Gemini structure)",
)
@click.option(
    "--api-key",
    type=str,
    help="API key for Gemini AI (optional, can use GOOGLE_AI_API_KEY environment variable)",
)
@click.option(
    "--model",
    type=str,
    default="gemini-2.5-flash-preview-05-20",
    help="Gemini model to use (default: gemini-2.5-flash-preview-05-20)",
)
@click.option(
    "--vision-credentials",
    type=click.Path(exists=True, path_type=Path),
    help="Path to Google Cloud Vision service account JSON file (for vision-gemini engine, optional if GOOGLE_APPLICATION_CREDENTIALS is set)",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="Image resolution for PDF conversion (default: 300)",
)
@click.option(
    "--high-accuracy",
    is_flag=True,
    help="Enable high-accuracy mode for better OCR results (slower, uses more tokens)",
)
@click.option(
    "--post-process",
    type=click.Choice(["none", "regex", "nlp"]),  # TODO: Implement regex and NLP post-processing
    default="none",
    help="Post-processing method for extracted data (default: none)",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "txt", "csv"]),  # TODO: Implement additional output formats
    default="json",
    help="Output format (default: json)",
)
@click.option(
    "--separate-folders",
    is_flag=True,
    help="Create separate output folders for each PDF (default: single folder with prefixed files)",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    help="Continue processing other PDFs if one fails",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose logging"
)
def main(
    input_path: Path,
    out_dir: Path,
    recursive: bool,
    ocr_engine: str,
    api_key: str,
    model: str,
    vision_credentials: Path,
    dpi: int,
    high_accuracy: bool,
    post_process: str,
    output_format: str,
    separate_folders: bool,
    continue_on_error: bool,
    verbose: bool,
) -> None:
    """
    Extract legal descriptions from deed PDFs using AI-powered OCR.

    This tool processes scanned deed PDFs to:
    1. Convert PDF pages to high-resolution images
    2. Extract text and structured data using AI OCR engines
    3. Identify and extract legal descriptions
    4. Output structured JSON with page-level results
    
    INPUT can be either:
    - A single PDF file: --input deed.pdf
    - A folder of PDFs: --input ./deed_folder
    
    Supports two OCR engines:
    • 'gemini': Direct image-to-structure using Gemini AI (faster, less control)
    • 'vision-gemini': Google Vision API + Gemini AI (more accurate text extraction)
    
    Set GOOGLE_AI_API_KEY environment variable or use --api-key option.
    For vision-gemini, also set GOOGLE_APPLICATION_CREDENTIALS or use --vision-credentials.
    
    Examples:
        # Process single PDF with Gemini (direct image-to-structure)
        deed-ocr -i deed.pdf -o ./results
        
        # Process single PDF with specific model
        deed-ocr -i deed.pdf -o ./results --model gemini-2.5-pro-preview-06-05
        
        # Process single PDF with high accuracy (better results, slower)
        deed-ocr -i deed.pdf -o ./results --high-accuracy
        
        # Process single PDF with Vision + Gemini workflow
        deed-ocr -i deed.pdf -o ./results --ocr-engine vision-gemini --vision-credentials /path/to/service-account.json
        
        # Process folder of PDFs with custom model and high accuracy
        deed-ocr -i ./pdf_folder -o ./results --model gemini-2.5-pro-preview-06-05 --high-accuracy
        
        # Process folder recursively with separate output folders and high accuracy
        deed-ocr -i ./pdf_folder -o ./results --recursive --separate-folders --high-accuracy
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Find PDF files to process
        pdf_files = find_pdf_files(input_path, recursive)
        
        if not pdf_files:
            click.echo("❌ No PDF files found in the specified input path!", err=True)
            raise click.ClickException("No PDF files found")
        
        # Validate OCR engine requirements
        if ocr_engine == "vision-gemini":
            # Check for vision credentials (CLI option or environment variable)
            vision_creds_path = None
            if vision_credentials:
                vision_creds_path = str(vision_credentials)
            else:
                # Try to get from environment variable
                import os
                vision_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            if not vision_creds_path:
                click.echo("❌ Error: Google Cloud Vision credentials required for vision-gemini engine!", err=True)
                click.echo("   Either use --vision-credentials /path/to/service-account.json", err=True)
                click.echo("   Or set GOOGLE_APPLICATION_CREDENTIALS environment variable", err=True)
                raise click.ClickException("Missing required Vision credentials")
        else:
            vision_creds_path = str(vision_credentials) if vision_credentials else None
        
        # Display processing summary
        click.echo(f"🏠 Found {len(pdf_files)} PDF file(s) to process")
        click.echo(f"📊 OCR Engine: {ocr_engine}")
        if ocr_engine == "gemini" or ocr_engine == "vision-gemini":
            click.echo(f"🤖 Gemini Model: {model}")
        if ocr_engine == "vision-gemini":
            click.echo(f"🔧 Vision Credentials: {vision_creds_path}")
        click.echo(f"📁 Output Directory: {out_dir}")
        click.echo(f"🖼️  Image Resolution: {dpi} DPI")
        click.echo(f"🎯 High Accuracy: {'enabled' if high_accuracy else 'disabled (default)'}")
        if input_path.is_dir():
            click.echo(f"🔍 Recursive search: {'enabled' if recursive else 'disabled'}")
            click.echo(f"📂 Separate folders: {'enabled' if separate_folders else 'disabled'}")
        
        if post_process != "none":
            click.echo(f"⚙️  Post-processing: {post_process}")
        
        click.echo("=" * 70)
        
        successful_count = 0
        failed_count = 0
        
        # Process each PDF file
        for i, pdf_file in enumerate(pdf_files, 1):
            click.echo(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                # Determine output directory for this PDF
                if separate_folders:
                    pdf_output_dir = out_dir / pdf_file.stem
                else:
                    pdf_output_dir = out_dir
                
                # Main processing
                if ocr_engine == "gemini":
                    result = process_deed_pdf_simple(
                        pdf_path=pdf_file,
                        api_key=api_key,
                        model=model,
                        output_dir=pdf_output_dir,
                        dpi=dpi,
                        high_accuracy=high_accuracy
                    )
                    # Display results for Gemini workflow
                    click.echo(f"    ✅ Success: {result.total_pages} pages, {len(result.all_legal_descriptions)} legal descriptions")
                    
                elif ocr_engine == "vision-gemini":
                    result = process_deed_pdf_vision_gemini(
                        pdf_path=pdf_file,
                        vision_credentials_path=vision_creds_path,
                        gemini_api_key=api_key,
                        output_dir=pdf_output_dir,
                        dpi=dpi
                    )
                    # Display results for Vision + Gemini workflow
                    legal_desc_count = len(result.gemini_structured_result.get("legal_description_block", []))
                    click.echo(f"    ✅ Success: {result.total_pages} pages, {legal_desc_count} legal descriptions")
                    click.echo(f"    📝 Text extracted: {len(result.combined_full_text)} characters")
                    
                else:
                    # TODO: Implement other OCR engines
                    raise click.ClickException(f"OCR engine '{ocr_engine}' not yet implemented")
                
                # TODO: Apply post-processing if requested
                if post_process == "regex":
                    click.echo("    ⚠️  Regex post-processing not yet implemented")
                elif post_process == "nlp":
                    click.echo("    ⚠️  NLP post-processing not yet implemented")
                
                click.echo(f"    💾 Saved to: {pdf_output_dir}")
                
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                error_str = str(e).lower()
                
                # Provide helpful error messages for common issues
                if "404" in error_str and "model" in error_str:
                    click.echo(f"    ❌ Model Error: The model '{model}' was not found or is not supported.", err=True)
                    click.echo(f"    💡 Try using a different model like 'gemini-2.5-pro-preview-06-05' with --model option", err=True)
                elif "authentication" in error_str or "unauthorized" in error_str or "api key" in error_str:
                    click.echo(f"    ❌ Authentication Error: Invalid API key or authentication failed.", err=True)
                    click.echo(f"    💡 Check your GOOGLE_AI_API_KEY or use --api-key option", err=True)
                else:
                    click.echo(f"    ❌ Failed: {str(e)}", err=True)
                
                if verbose:
                    import traceback
                    traceback.print_exc()
                
                if not continue_on_error:
                    if "404" in error_str and "model" in error_str:
                        click.echo(f"\n💥 Model '{model}' not found. Try a different model with --model option.", err=True)
                    else:
                        click.echo(f"\n💥 Stopping due to error. Use --continue-on-error to process remaining files.", err=True)
                    raise click.ClickException(f"Processing failed for {pdf_file.name}")
                else:
                    click.echo(f"    ⏭️  Continuing with remaining files...")
        
        # Final summary
        click.echo("\n" + "=" * 70)
        click.echo("🎯 BATCH PROCESSING COMPLETED!")
        click.echo("=" * 70)
        
        click.echo(f"📊 Total files found: {len(pdf_files)}")
        click.echo(f"✅ Successfully processed: {successful_count}")
        if failed_count > 0:
            click.echo(f"❌ Failed: {failed_count}")
        
        click.echo(f"\n💾 All results saved to: {out_dir.absolute()}")
        
        if successful_count > 0:
            click.echo("\nFiles created per PDF:")
            if ocr_engine == "gemini":
                click.echo("  • {pdf_name}/ - PDF result folder")
                click.echo("    ├── full_text.txt - Combined text")
                click.echo("    ├── full_pdf_analysis.json - Complete structured data")
                click.echo("    ├── final_result.json - Final merged result")
                click.echo("    └── pages/ - Individual page results")
            elif ocr_engine == "vision-gemini":
                click.echo("  • {pdf_name}_vision_gemini/ - PDF result folder")
                click.echo("    ├── full_text.txt - Combined text from Vision API")
                click.echo("    ├── vision_results.json - Raw Vision API results")
                click.echo("    ├── gemini_structured.json - Structured data from Gemini")
                click.echo("    ├── complete_result.json - Complete workflow result")
                click.echo("    ├── token_usage.json - Gemini token usage")
                click.echo("    └── pages_text/ - Individual page text files")
        
        # TODO: Support additional output formats
        if output_format != "json":
            click.echo(f"⚠️  Output format '{output_format}' not yet implemented (defaulting to JSON)")
        
        if failed_count > 0:
            raise click.ClickException(f"Some files failed to process ({failed_count}/{len(pdf_files)})")
        
    except Exception as e:
        if "Some files failed" not in str(e):
            click.echo(f"\n❌ Error: {str(e)}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 
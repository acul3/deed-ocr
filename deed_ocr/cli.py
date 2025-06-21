"""Command-line interface for deed-ocr."""

import logging
import os
from pathlib import Path
from typing import List

import click
from dotenv import load_dotenv

from deed_ocr.workflow import process_deed_pdf_simple, SimplifiedDeedOCRWorkflow
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
    default="gemini-2.5-flash",
    help="Gemini model to use (default: gemini-2.5-flash)",
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
@click.option(
    "--geodatabase-path",
    type=click.Path(path_type=Path),
    default=lambda: Path(os.getenv("GEODATABASE_PATH", "")) if os.getenv("GEODATABASE_PATH") else None,
    help="Path to PLSS geodatabase file (.gdb) for TRS validation (default: GEODATABASE_PATH env var)",
)
@click.option(
    "--counties-json-path",
    type=click.Path(path_type=Path),
    default=lambda: Path(os.getenv("COUNTIES_JSON_PATH", "")) if os.getenv("COUNTIES_JSON_PATH") else None,
    help="Path to counties JSON file for state mapping in TRS validation (default: COUNTIES_JSON_PATH env var)",
)
@click.option(
    "--disable-trs-validation",
    is_flag=True,
    help="Disable TRS validation (enabled by default if geodatabase provided)",
)
@click.option(
    "--stage-2",
    is_flag=True,
    help="Stage-2 processing: Process all PDFs in folder with high-accuracy, create Index_Output.xlsx and doc_texts subfolder",
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
    geodatabase_path: Path,
    counties_json_path: Path,
    disable_trs_validation: bool,
    stage_2: bool,
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
        deed-ocr -i deed.pdf -o ./results --model gemini-2.5-pro
        
        # Process single PDF with high accuracy (better results, slower)
        deed-ocr -i deed.pdf -o ./results --high-accuracy
        
        # Process single PDF with Vision + Gemini workflow
        deed-ocr -i deed.pdf -o ./results --ocr-engine vision-gemini --vision-credentials /path/to/service-account.json
        
        # Process folder of PDFs with custom model and high accuracy
        deed-ocr -i ./pdf_folder -o ./results --model gemini-2.5-pro --high-accuracy
        
        # Process folder recursively with separate output folders and high accuracy
        deed-ocr -i ./pdf_folder -o ./results --recursive --separate-folders --high-accuracy
        
        # Stage-2 processing: Process all PDFs in folder with high-accuracy, create combined outputs
        deed-ocr -i ./pdf_folder -o ./results --stage-2
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Stage-2 validation and setup
        if stage_2:
            # Stage-2 only works with folders, not single files
            if not input_path.is_dir():
                click.echo("❌ Error: --stage-2 requires a folder input, not a single PDF file!", err=True)
                raise click.ClickException("Stage-2 mode requires folder input")
            
            # Force high-accuracy mode for stage-2
            if not high_accuracy:
                click.echo("🎯 Stage-2 mode: Automatically enabling high-accuracy mode")
                high_accuracy = True
            
            # Force separate folders for stage-2
            if not separate_folders:
                click.echo("📂 Stage-2 mode: Automatically enabling separate folders")
                separate_folders = True
            
            # Force Gemini engine for stage-2 (most reliable)
            if ocr_engine != "gemini":
                click.echo(f"🤖 Stage-2 mode: Switching from {ocr_engine} to gemini engine")
                ocr_engine = "gemini"
            
            click.echo("🚀 STAGE-2 PROCESSING MODE ENABLED")
            click.echo("   • High-accuracy mode: ON")
            click.echo("   • Separate folders: ON")
            click.echo("   • Engine: Gemini")
            click.echo("   • Will create Index_Output.xlsx and doc_texts/ folder")
            click.echo("=" * 50)

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
        
        # TRS validation info
        # Check for environment variables if not provided via CLI and show source
        geodatabase_source = ""
        counties_source = ""
        
        if geodatabase_path is None and os.getenv("GEODATABASE_PATH"):
            geodatabase_path = Path(os.getenv("GEODATABASE_PATH"))
            geodatabase_source = " (from env)"
        elif geodatabase_path:
            geodatabase_source = " (from CLI)"
            
        if counties_json_path is None and os.getenv("COUNTIES_JSON_PATH"):
            counties_json_path = Path(os.getenv("COUNTIES_JSON_PATH"))
            counties_source = " (from env)"
        elif counties_json_path:
            counties_source = " (from CLI)"
        
        # Validate paths and show warnings for missing files
        if geodatabase_path and not geodatabase_path.exists():
            click.echo(f"⚠️  Warning: Geodatabase file not found: {geodatabase_path}", err=True)
            geodatabase_path = None
            
        if counties_json_path and not counties_json_path.exists():
            click.echo(f"⚠️  Warning: Counties JSON file not found: {counties_json_path}", err=True)
            click.echo(f"   TRS validation will use built-in county mapping", err=True)
            counties_json_path = None
            
        enable_trs_validation = not disable_trs_validation and (geodatabase_path or counties_json_path)
        if enable_trs_validation:
            click.echo(f"🗺️  TRS Validation: enabled")
            if geodatabase_path:
                click.echo(f"    📊 Geodatabase: {geodatabase_path}{geodatabase_source}")
            if counties_json_path:
                click.echo(f"    🏛️  Counties file: {counties_json_path}{counties_source}")
        else:
            click.echo(f"🗺️  TRS Validation: disabled")
        
        if input_path.is_dir():
            click.echo(f"🔍 Recursive search: {'enabled' if recursive else 'disabled'}")
            click.echo(f"📂 Separate folders: {'enabled' if separate_folders else 'disabled'}")
        
        if post_process != "none":
            click.echo(f"⚙️  Post-processing: {post_process}")
        
        click.echo("=" * 70)
        
        successful_count = 0
        failed_count = 0
        
        # Create workflow instance once for Gemini engine (to avoid reloading geodatabase)
        gemini_workflow = None
        if ocr_engine == "gemini" and len(pdf_files) > 1:
            # Only create shared workflow for multiple files to avoid reloading geodatabase
            try:
                click.echo(f"🔧 Initializing workflow (loading geodatabase once for {len(pdf_files)} files)...")
                gemini_workflow = SimplifiedDeedOCRWorkflow(
                    api_key=api_key,
                    model=model,
                    dpi=dpi,
                    max_retries=3,
                    retry_delay=5.0,
                    high_accuracy=high_accuracy,
                    geodatabase_path=str(geodatabase_path) if geodatabase_path else None,
                    counties_json_path=str(counties_json_path) if counties_json_path else None,
                    enable_trs_validation=enable_trs_validation
                )
                if enable_trs_validation and gemini_workflow.trs_validator:
                    click.echo(f"    ✅ TRS validator loaded successfully")
                click.echo(f"    ✅ Workflow initialized - geodatabase loaded once for batch processing")
            except Exception as e:
                click.echo(f"    ⚠️  Failed to initialize shared workflow: {str(e)}", err=True)
                click.echo(f"    📝 Will create individual workflows for each PDF", err=True)
                gemini_workflow = None
        
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
                    if gemini_workflow is not None:
                        # Use shared workflow instance (geodatabase already loaded)
                        result = gemini_workflow.process_pdf(
                            pdf_path=pdf_file,
                            output_dir=pdf_output_dir,
                            force_reprocess=False
                        )
                    else:
                        # Fallback to individual workflow (single file or shared workflow failed)
                        result = process_deed_pdf_simple(
                            pdf_path=pdf_file,
                            api_key=api_key,
                            model=model,
                            output_dir=pdf_output_dir,
                            dpi=dpi,
                            high_accuracy=high_accuracy,
                            geodatabase_path=str(geodatabase_path) if geodatabase_path else None,
                            counties_json_path=str(counties_json_path) if counties_json_path else None,
                            enable_trs_validation=enable_trs_validation
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
                    click.echo(f"    💡 Try using a different model like 'gemini-2.5-pro' with --model option", err=True)
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
        
        # Stage-2 post-processing
        if stage_2 and successful_count > 0:
            click.echo("\n" + "=" * 70)
            click.echo("🚀 STAGE-2 POST-PROCESSING")
            click.echo("=" * 70)
            
            try:
                # Import the stage-2 functions
                from deed_ocr.utils.excel_converter import combine_excel_files_to_index, collect_and_rename_text_files
                
                # Create Index_Output.xlsx
                click.echo("📊 Creating Index_Output.xlsx...")
                index_output_path = out_dir / "Index_Output.xlsx"
                instruction_csv_path = Path(__file__).resolve().parent.parent / "instruction.csv"
                
                combine_excel_files_to_index(
                    results_dir=out_dir,
                    output_index_path=index_output_path,
                    instruction_csv_path=instruction_csv_path if instruction_csv_path.exists() else None
                )
                click.echo(f"    ✅ Index_Output.xlsx created: {index_output_path}")
                
                # Create doc_texts subfolder
                click.echo("📝 Creating doc_texts/ folder...")
                doc_texts_dir = out_dir / "doc_texts"
                
                collect_and_rename_text_files(
                    results_dir=out_dir,
                    doc_texts_dir=doc_texts_dir
                )
                click.echo(f"    ✅ doc_texts/ folder created: {doc_texts_dir}")
                
                click.echo("\n🎉 Stage-2 post-processing completed successfully!")
                
            except Exception as e:
                click.echo(f"    ❌ Error during stage-2 post-processing: {str(e)}", err=True)
                if verbose:
                    import traceback
                    traceback.print_exc()

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
                click.echo("    ├── final_result.xlsx - Excel export")
                if enable_trs_validation:
                    click.echo("    ├── trs_validation.xlsx - TRS validation results")
                click.echo("    └── pages/ - Individual page results")
            elif ocr_engine == "vision-gemini":
                click.echo("  • {pdf_name}_vision_gemini/ - PDF result folder")
                click.echo("    ├── full_text.txt - Combined text from Vision API")
                click.echo("    ├── vision_results.json - Raw Vision API results")
                click.echo("    ├── gemini_structured.json - Structured data from Gemini")
                click.echo("    ├── complete_result.json - Complete workflow result")
                click.echo("    ├── token_usage.json - Gemini token usage")
                click.echo("    └── pages_text/ - Individual page text files")
            
            if stage_2:
                click.echo("\nStage-2 additional outputs:")
                click.echo("  • Index_Output.xlsx - Combined data from all PDFs")
                click.echo("  • doc_texts/ - All full_text.txt files renamed as:")
                click.echo("    └── {pdf_name}_{reception_number}.txt")
        
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
"""Command-line interface for deed-ocr."""

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from deed_ocr.main import process_deed_pdf

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@click.command()
@click.option(
    "--pdf",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input PDF file",
)
@click.option(
    "--out_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for results",
)
@click.option(
    "--ocr-provider",
    type=click.Choice(["google", "textract"]),
    default="google",
    help="OCR provider to use (default: google)",
)
@click.option(
    "--extraction-method",
    type=click.Choice(["regex", "hybrid", "llm"]),
    default="hybrid",
    help="Extraction method: regex (traditional), hybrid (regex+NLP), or llm (LLM-enhanced)",
)
@click.option(
    "--llm-provider",
    type=click.Choice(["qwen3", "gemini", "chatgpt"]),
    default="gemini",
    help="LLM provider for enhanced extraction (default: gemini)",
)
@click.option(
    "--use-layout/--no-layout",
    default=True,
    help="Use layout detection for better extraction (default: enabled)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose logging"
)
def main(
    pdf: Path,
    out_dir: Path,
    ocr_provider: str,
    extraction_method: str,
    llm_provider: str,
    use_layout: bool,
    verbose: bool,
) -> None:
    """
    Extract legal descriptions from deed PDFs using OCR.

    This tool processes scanned deed PDFs to:
    1. Extract text using Google Cloud Vision or Amazon Textract
    2. Identify and extract legal descriptions
    3. Output page-level text files and structured JSON
    
    For LLM-enhanced extraction, ensure appropriate API keys are set:
    - GEMINI_API_KEY for Gemini Pro
    - OPENAI_API_KEY for ChatGPT
    - QWEN_MODEL_PATH for local Qwen3
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        click.echo(f"Processing {pdf.name}...")
        click.echo(f"OCR Provider: {ocr_provider}")
        click.echo(f"Extraction Method: {extraction_method}")
        
        if extraction_method == "llm":
            click.echo(f"LLM Provider: {llm_provider}")
            click.echo(f"Layout Detection: {'enabled' if use_layout else 'disabled'}")
        
        result = process_deed_pdf(
            pdf,
            ocr_provider=ocr_provider,
            output_dir=out_dir,
            extraction_method=extraction_method,
            llm_provider=llm_provider,
            use_layout=use_layout,
        )
        
        click.echo(f"✓ Processed {result.total_pages} pages")
        click.echo(f"✓ Found {len(result.legal_descriptions)} legal descriptions")
        
        # Show confidence scores for found descriptions
        for i, desc in enumerate(result.legal_descriptions[:3]):
            click.echo(
                f"  {i+1}. Page {desc.page_number}: "
                f"{desc.text[:50]}... (confidence: {desc.confidence:.2f})"
            )
        
        click.echo(f"✓ Results saved to {out_dir}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main() 
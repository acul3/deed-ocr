"""Deed OCR: Legal description extraction from deed PDFs."""

__version__ = "0.1.0"

# Import simplified workflow for easy access
from .workflow import process_deed_pdf_simple, SimplifiedDeedOCRWorkflow, SimplifiedDeedResult

__all__ = [
    "process_deed_pdf_simple", 
    "SimplifiedDeedOCRWorkflow", 
    "SimplifiedDeedResult"
] 
"""PDF to Image conversion utilities."""

import logging
from pathlib import Path
from typing import List, Optional
from PIL import Image
import fitz  # pymupdf
import io
import base64

logger = logging.getLogger(__name__)


class PDFToImageConverter:
    """Convert PDF pages to images for OCR processing."""
    
    def __init__(self, dpi: int = 300, format: str = "PNG"):
        """
        Initialize PDF to Image converter.
        
        Args:
            dpi: Resolution for image conversion (higher = better quality)
            format: Output image format (PNG, JPEG, etc.)
        """
        self.dpi = dpi
        self.format = format
        
    def convert_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PIL Image objects, one per page
        """
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            
            # Open PDF with pymupdf
            pdf_document = fitz.open(str(pdf_path))
            images = []
            
            # Convert each page to image
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Create a matrix for the desired DPI
                # Default DPI in fitz is 72, so we scale accordingly
                zoom = self.dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert pixmap to PIL Image
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
                
            pdf_document.close()
            logger.info(f"Successfully converted {len(images)} pages to images")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
            
    def convert_to_base64(self, pdf_path: Path) -> List[str]:
        """
        Convert PDF pages to base64 encoded images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of base64 encoded image strings
        """
        try:
            images = self.convert_to_images(pdf_path)
            base64_images = []
            
            for i, image in enumerate(images):
                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format=self.format)
                img_bytes = img_buffer.getvalue()
                
                # Encode to base64
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                base64_images.append(base64_image)
                
            logger.info(f"Successfully converted {len(base64_images)} pages to base64")
            return base64_images
            
        except Exception as e:
            logger.error(f"Error converting PDF to base64 images: {str(e)}")
            raise
            
    def convert_to_bytes(self, pdf_path: Path) -> List[bytes]:
        """
        Convert PDF pages to image bytes.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of image bytes
        """
        try:
            images = self.convert_to_images(pdf_path)
            image_bytes_list = []
            
            for image in images:
                img_buffer = io.BytesIO()
                image.save(img_buffer, format=self.format)
                img_bytes = img_buffer.getvalue()
                image_bytes_list.append(img_bytes)
                
            logger.info(f"Successfully converted {len(image_bytes_list)} pages to bytes")
            return image_bytes_list
            
        except Exception as e:
            logger.error(f"Error converting PDF to image bytes: {str(e)}")
            raise
            
    def save_images(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """
        Convert PDF to images and save them to disk.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            
        Returns:
            List of saved image file paths
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            images = self.convert_to_images(pdf_path)
            saved_paths = []
            
            pdf_name = pdf_path.stem
            for i, image in enumerate(images, 1):
                image_path = output_dir / f"{pdf_name}_page_{i:03d}.{self.format.lower()}"
                image.save(image_path, format=self.format)
                saved_paths.append(image_path)
                
            logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving PDF images: {str(e)}")
            raise


def create_pdf_converter(dpi: int = 300, format: str = "PNG") -> PDFToImageConverter:
    """
    Factory function to create PDF to Image converter.
    
    Args:
        dpi: Resolution for image conversion
        format: Output image format
        
    Returns:
        Configured PDFToImageConverter instance
    """
    return PDFToImageConverter(dpi=dpi, format=format) 
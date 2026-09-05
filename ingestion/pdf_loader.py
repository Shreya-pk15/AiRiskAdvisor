"""
PDF document loader using PyMuPDF (fitz).
Extracts text page-by-page from PDF files.
"""

import pymupdf as fitz  # PyMuPDF
from typing import Dict, Any, Union
import io
import os


def load_pdf(file_input: Union[str, bytes, io.BytesIO], filename: str = "document.pdf") -> Dict[str, Any]:
    """
    Extract text content from a PDF document using PyMuPDF.

    Args:
        file_input: File path (str) or file byte stream/BytesIO.
        filename: Name of the PDF document.

    Returns:
        Dict[str, Any]: {
            "text": Extracted document text,
            "source": filename,
            "file_type": "pdf"
        }
    """
    try:
        if isinstance(file_input, str):
            doc = fitz.open(file_input)
        elif isinstance(file_input, bytes):
            doc = fitz.open(stream=file_input, filetype="pdf")
        elif hasattr(file_input, "read"):
            content = file_input.read()
            # Reset stream position if possible
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            doc = fitz.open(stream=content, filetype="pdf")
        else:
            raise ValueError("Unsupported input type for PDF loader.")

        text_pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text.strip():
                text_pages.append(f"--- Page {page_num + 1} ---\n{page_text.strip()}")

        doc.close()

        full_text = "\n\n".join(text_pages)
        if not full_text.strip():
            raise ValueError(f"PDF file '{filename}' contains no readable text.")

        return {
            "text": full_text,
            "source": os.path.basename(filename),
            "file_type": "pdf"
        }
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF '{filename}': {str(e)}")

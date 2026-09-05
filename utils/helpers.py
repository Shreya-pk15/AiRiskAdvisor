"""
Helper utility functions for string formatting, file validation, and metadata display.
"""

import re
import os
from typing import List, Dict, Any

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt"}


def is_allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a supported extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing unsafe path characters."""
    return os.path.basename(filename).strip()


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text content.
    - Replaces multi-whitespace with single spaces
    - Normalizes multi-newlines to double newlines (preserving paragraph boundaries)
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    
    # Replace non-breaking spaces and Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    
    # Compress multiple horizontal spaces
    text = re.sub(r"[ \t]+", " ", text)
    
    # Compress 3 or more vertical newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def format_source_attribution(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of retrieved chunk metadata into a clean list of source document names.
    """
    sources = set()
    for item in retrieved_chunks:
        meta = item.get("metadata", {})
        source_name = meta.get("source", "Unknown Document")
        sources.add(source_name)
    
    if not sources:
        return "No sources available"
    
    return "\n".join([f"- {src}" for src in sorted(sources)])

"""
Plain TXT document loader using Python built-in file handling.
Supports multiple encoding fallbacks (UTF-8, Latin-1, CP1252).
"""

from typing import Dict, Any, Union
import io
import os


def load_txt(file_input: Union[str, bytes, io.BytesIO], filename: str = "document.txt") -> Dict[str, Any]:
    """
    Extract text content from a plain text file.

    Args:
        file_input: File path (str), raw bytes, or BytesIO object.
        filename: Name of the TXT document.

    Returns:
        Dict[str, Any]: {
            "text": Extracted text,
            "source": filename,
            "file_type": "txt"
        }
    """
    try:
        raw_bytes = None
        if isinstance(file_input, str):
            with open(file_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file_input, bytes):
            raw_bytes = file_input
        elif hasattr(file_input, "read"):
            raw_bytes = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)
        else:
            raise ValueError("Unsupported input type for TXT loader.")

        if not raw_bytes:
            raise ValueError(f"TXT file '{filename}' is empty.")

        # Decode with fallback options
        text = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = raw_bytes.decode(encoding)
                break
            except (UnicodeDecodeError, AttributeError):
                continue

        if text is None:
            text = raw_bytes.decode("utf-8", errors="ignore")

        if not text.strip():
            raise ValueError(f"TXT file '{filename}' contains no readable content.")

        return {
            "text": text,
            "source": os.path.basename(filename),
            "file_type": "txt"
        }
    except Exception as e:
        raise RuntimeError(f"Failed to read TXT file '{filename}': {str(e)}")

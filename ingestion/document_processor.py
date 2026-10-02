"""
Unified Document Processor & Normalizer.
Detects file extension, dispatches to the correct loader,
normalizes extracted text, and attaches standard document metadata.
"""

import os
from typing import Dict, Any, Union, List
from ingestion.pdf_loader import load_pdf
from ingestion.docx_loader import load_docx
from ingestion.csv_loader import load_csv
from ingestion.txt_loader import load_txt
from ingestion.xlsx_loader import load_xlsx
from utils.helpers import clean_text, sanitize_filename


class DocumentProcessor:
    """Processor class responsible for loading and normalizing project documents."""

    LOADERS = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".csv": load_csv,
        ".txt": load_txt,
        ".xlsx": load_xlsx,
    }

    def process_file(
        self,
        file_input: Union[str, bytes, Any],
        filename: str,
        project_name: str = "Default Project"
    ) -> Dict[str, Any]:
        """
        Ingest, extract, clean, and normalize a project document.

        Args:
            file_input: Path, bytes, or file stream object.
            filename: Original file name.
            project_name: Associated project identifier.

        Returns:
            Dict[str, Any]: {
                "text": Cleaned document text,
                "source": Sanitized filename,
                "file_type": Extension without dot (e.g. "pdf"),
                "project_name": project_name,
                "character_count": int,
                "word_count": int
            }
        """
        clean_name = sanitize_filename(filename)
        ext = os.path.splitext(clean_name)[1].lower()

        if ext not in self.LOADERS:
            raise ValueError(
                f"Unsupported file type '{ext}' for file '{clean_name}'. "
                f"Allowed types: {list(self.LOADERS.keys())}"
            )

        loader_fn = self.LOADERS[ext]
        raw_doc = loader_fn(file_input, clean_name)

        # Normalize text
        normalized_text = clean_text(raw_doc["text"])

        return {
            "text": normalized_text,
            "source": clean_name,
            "file_type": ext.lstrip("."),
            "project_name": project_name,
            "character_count": len(normalized_text),
            "word_count": len(normalized_text.split())
        }


def process_uploaded_file(
    file_input: Union[str, bytes, Any],
    filename: str,
    project_name: str = "Default Project"
) -> Dict[str, Any]:
    """Helper function to process a single document."""
    processor = DocumentProcessor()
    return processor.process_file(file_input, filename, project_name)

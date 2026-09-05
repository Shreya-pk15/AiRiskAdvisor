"""
Ingestion package for extracting text from PDF, DOCX, CSV, and TXT files.
"""
from ingestion.document_processor import process_uploaded_file, DocumentProcessor

__all__ = ["process_uploaded_file", "DocumentProcessor"]

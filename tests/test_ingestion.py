"""
Unit tests for document ingestion and text normalization modules.
Tests PDF, DOCX, CSV, and TXT loaders using sample data files.
"""

import os
import pytest
from ingestion.document_processor import process_uploaded_file
from generate_sample_data import generate_all_sample_files


@pytest.fixture(scope="module", autouse=True)
def setup_sample_data():
    """Ensure sample data files are generated before running tests."""
    sample_dir = "./sample_data"
    generate_all_sample_files(sample_dir)
    return sample_dir


def test_pdf_ingestion():
    pdf_path = "./sample_data/sample_proposal.pdf"
    doc_info = process_uploaded_file(pdf_path, "sample_proposal.pdf", "SPMS Project")

    assert doc_info["file_type"] == "pdf"
    assert doc_info["source"] == "sample_proposal.pdf"
    assert "Student Project Management System" in doc_info["text"]
    assert "Executive Summary" in doc_info["text"]
    assert doc_info["word_count"] > 20


def test_docx_ingestion():
    docx_path = "./sample_data/sample_srs.docx"
    doc_info = process_uploaded_file(docx_path, "sample_srs.docx", "SPMS Project")

    assert doc_info["file_type"] == "docx"
    assert doc_info["source"] == "sample_srs.docx"
    assert "Software Requirements Specification" in doc_info["text"]
    assert "FR-1" in doc_info["text"]


def test_csv_ingestion():
    csv_path = "./sample_data/sample_tasks.csv"
    doc_info = process_uploaded_file(csv_path, "sample_tasks.csv", "SPMS Project")

    assert doc_info["file_type"] == "csv"
    assert doc_info["source"] == "sample_tasks.csv"
    # Verify semantic row conversion (Record 1: Task_ID: TASK-101 | Task_Name: ...)
    assert "Record 1:" in doc_info["text"]
    assert "Task_Name: Payment Gateway Integration" in doc_info["text"]
    assert "Status: Blocked" in doc_info["text"]


def test_txt_ingestion():
    txt_path = "./sample_data/sample_meeting_notes.txt"
    doc_info = process_uploaded_file(txt_path, "sample_meeting_notes.txt", "SPMS Project")

    assert doc_info["file_type"] == "txt"
    assert doc_info["source"] == "sample_meeting_notes.txt"
    assert "SPRINT REVIEW MEETING NOTES" in doc_info["text"]
    assert "Payment Integration Task is currently BLOCKED" in doc_info["text"]

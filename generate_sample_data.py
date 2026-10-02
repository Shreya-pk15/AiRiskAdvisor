"""
Sample Data Generator Script.

Creates a cohesive sample dataset for the fictional project:
"Student Project Management System (SPMS)"

Generated files in sample_data/:
- sample_proposal.pdf: Project scope, objectives, team roles, and architecture.
- sample_srs.docx: Detailed software functional requirements and deliverables.
- sample_meeting_notes.txt: Sprint review meeting notes documenting blockers and decisions.
- sample_tasks.csv: Sprint task breakdown with assignees, deadlines, and statuses.
- sample_defects.xlsx: Defect tracker spreadsheet (XLSX ingestion validation).
"""

import os
import pymupdf as fitz  # PyMuPDF
import docx
import pandas as pd


def generate_all_sample_files(output_dir: str = "./sample_data"):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Generate sample_proposal.pdf
    # -------------------------------------------------------------------------
    pdf_path = os.path.join(output_dir, "sample_proposal.pdf")
    doc_pdf = fitz.open()
    
    page1 = doc_pdf.new_page()
    proposal_text = (
        "PROJECT PROPOSAL\n\n"
        "Project Title: Student Project Management System (SPMS)\n"
        "Author: Computer Science Department\n"
        "Date: September 2026\n\n"
        "1. Executive Summary & Objective:\n"
        "The primary objective of the Student Project Management System (SPMS) is to provide "
        "a unified platform for undergraduate software development teams to collaborate, track "
        "sprint deliverables, submit progress reports, and assess project risks in real time.\n\n"
        "2. Project Scope:\n"
        "- Document Ingestion and Centralized Repository\n"
        "- Interactive Sprint Planning and Task Board\n"
        "- AI-Assisted Project Health Assessment\n"
        "- Automated Blocker and Risk Detection\n\n"
        "3. Technology Stack:\n"
        "- Frontend: Streamlit and Python\n"
        "- Database: ChromaDB Vector Store\n"
        "- Machine Learning & Embeddings: Sentence Transformers (all-MiniLM-L6-v2)\n"
        "- Language Model: Google Gemini API\n"
        "- Core Language: Python 3.11\n\n"
        "4. Key Project Deliverables:\n"
        "- Deliverable 1: Document Ingestion Pipeline (PDF, DOCX, CSV, TXT)\n"
        "- Deliverable 2: Grounded Vector Retrieval Architecture\n"
        "- Deliverable 3: Interactive Project Health Dashboard\n"
    )
    page1.insert_text((50, 50), proposal_text, fontsize=11)
    doc_pdf.save(pdf_path)
    doc_pdf.close()
    print(f"[OK] Created {pdf_path}")

    # -------------------------------------------------------------------------
    # 2. Generate sample_srs.docx
    # -------------------------------------------------------------------------
    docx_path = os.path.join(output_dir, "sample_srs.docx")
    doc_word = docx.Document()
    
    doc_word.add_heading("Software Requirements Specification (SRS)", level=0)
    doc_word.add_heading("Project: Student Project Management System (SPMS)", level=1)
    
    doc_word.add_heading("1. Functional Requirements", level=2)
    doc_word.add_paragraph(
        "FR-1: The system shall support uploading project artifacts in PDF, DOCX, CSV, and TXT formats."
    )
    doc_word.add_paragraph(
        "FR-2: The system shall convert CSV records into semantic text representations before embedding."
    )
    doc_word.add_paragraph(
        "FR-3: The system shall split documents into overlapping chunks of 400 words with 50 words overlap."
    )
    doc_word.add_paragraph(
        "FR-4: The AI assistant shall answer user questions using only grounded context retrieved from uploaded project files."
    )

    doc_word.add_heading("2. System Architecture & Constraints", level=2)
    doc_word.add_paragraph(
        "The architecture relies on persistent local ChromaDB storage partitioned by project name. "
        "No global state is shared across separate project workspaces."
    )

    table = doc_word.add_table(rows=1, cols=3)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Module"
    hdr_cells[1].text = "Target Completion"
    hdr_cells[2].text = "Priority"

    row_data = [
        ("Ingestion Layer", "Week 1", "High"),
        ("Vector Store & Retrieval", "Week 2", "High"),
        ("Grounded Q&A Flow", "Week 2", "High"),
        ("Risk Forecasting Agent", "Week 4", "Medium")
    ]
    for mod, comp, prio in row_data:
        row_cells = table.add_row().cells
        row_cells[0].text = mod
        row_cells[1].text = comp
        row_cells[2].text = prio

    doc_word.save(docx_path)
    print(f"[OK] Created {docx_path}")

    # -------------------------------------------------------------------------
    # 3. Generate sample_meeting_notes.txt
    # -------------------------------------------------------------------------
    txt_path = os.path.join(output_dir, "sample_meeting_notes.txt")
    meeting_content = (
        "SPRINT REVIEW MEETING NOTES\n"
        "Project: Student Project Management System (SPMS)\n"
        "Date: September 02, 2026\n"
        "Attendees: Shreya (Lead), John (Backend), Alice (UI/UX)\n\n"
        "KEY DISCUSSION POINTS:\n"
        "1. Ingestion module completed ahead of schedule by John.\n"
        "2. Payment Integration Task is currently BLOCKED. The payment gateway provider "
        "has not issued the production API credentials. John cannot proceed until credentials arrive.\n"
        "3. Streamlit UI initial mockup designed by Alice. Search input and document uploaders are verified.\n"
        "4. Risk Identified: Memory usage spike during batch embedding of large PDF files. Solution is to "
        "use sentence-transformers in mini batches.\n"
        "5. Decision: Next sprint will focus on testing ChromaDB retrieval accuracy for multi-document projects.\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(meeting_content)
    print(f"[OK] Created {txt_path}")

    # -------------------------------------------------------------------------
    # 4. Generate sample_tasks.csv
    # -------------------------------------------------------------------------
    csv_path = os.path.join(output_dir, "sample_tasks.csv")
    tasks_data = {
        "Task_ID": ["TASK-101", "TASK-102", "TASK-103", "TASK-104", "TASK-105"],
        "Task_Name": [
            "PDF and DOCX Ingestion",
            "Payment Gateway Integration",
            "Sentence Transformers Setup",
            "Streamlit Dashboard UI",
            "User Authentication Module"
        ],
        "Assignee": ["John", "John", "Shreya", "Alice", "Shreya"],
        "Status": ["Completed", "Blocked", "Completed", "In Progress", "Incomplete"],
        "Deadline": ["2026-09-01", "2026-09-10", "2026-09-03", "2026-09-08", "2026-09-15"],
        "Notes": [
            "Passed all ingestion unit tests.",
            "Blocked: Waiting for API credentials from payment gateway.",
            "Model all-MiniLM-L6-v2 loaded successfully.",
            "Sidebar and file uploader layout finished.",
            "Scheduled for next sprint."
        ]
    }
    df_tasks = pd.DataFrame(tasks_data)
    df_tasks.to_csv(csv_path, index=False)
    print(f"[OK] Created {csv_path}")

    # -------------------------------------------------------------------------
    # 5. Generate sample_defects.xlsx
    # -------------------------------------------------------------------------
    xlsx_path = os.path.join(output_dir, "sample_defects.xlsx")
    defects_data = {
        "Defect_ID": ["DEF-201", "DEF-202"],
        "Summary": [
            "XLSX parser memory spike on large sheets",
            "CSV row semantic conversion missing optional columns",
        ],
        "Severity": ["High", "Medium"],
        "Status": ["Open", "In Progress"],
        "Owner": ["John", "Shreya"],
    }
    pd.DataFrame(defects_data).to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"[OK] Created {xlsx_path}")


if __name__ == "__main__":
    generate_all_sample_files()

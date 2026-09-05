# 🛡️ AI Project Intelligence & Risk Advisor — Milestone 1

A modular, document-grounded Retrieval-Augmented Generation (RAG) application built with **Python**, **Streamlit**, **Sentence Transformers**, **ChromaDB**, and **Google Gemini API**.

Designed specifically for undergraduate B.Tech computer science software project teams, this application ingests heterogeneous project artifacts (PDF proposals, DOCX specifications, TXT meeting notes, CSV task lists), indexes them into a project-isolated vector database, and provides grounded, non-hallucinating AI answers.

---

## 📌 Project Overview & Problem Statement

### Problem Statement
Software development teams and student project groups generate large volumes of fragmented documentation across sprint cycles:
- Project proposals (.pdf)
- Software Requirement Specifications (.docx)
- Sprint review notes (.txt)
- Task boards & issue trackers (.csv)

When project mentors or team members ask questions like *"What tasks are currently blocking our release?"* or *"What technologies are required?"*, finding answers requires searching through multiple separate documents manually.

### Milestone 1 Solution
Milestone 1 implements the foundational **Document-Grounded RAG Pipeline**:
1. Ingests PDF, DOCX, CSV, and TXT files.
2. Normalizes text and transforms tabular CSV rows into semantic key-value strings.
3. Splits documents into overlapping text chunks (~400 words, 50-word overlap).
4. Generates 384-dimensional dense vector embeddings using `all-MiniLM-L6-v2`.
5. Indexes chunks into persistent, project-isolated ChromaDB collections.
6. Retrieves top-K relevant context chunks based on vector similarity search.
7. Executes a strictly grounded prompt against the LLM to generate verifiable answers with source document attribution.

---

## 🏗️ System Architecture

### Milestone 1 RAG Flow

```text
User Uploads Documents (.pdf, .docx, .csv, .txt)
                    │
                    ▼
          [Document Processor]
  (Extracts text, cleans formatting, normalizes)
                    │
                    ▼
            [Text Chunker]
  (Sliding window: 400 words, 50 word overlap)
                    │
                    ▼
          [Embedding Manager]
    (SentenceTransformers: all-MiniLM-L6-v2)
                    │
                    ▼
          [Vector Store Manager]
   (Persistent ChromaDB Collection per Project)
                    │
                    ▼
          User Asks Project Question
                    │
                    ▼
              [Retriever]
 (Generates query embedding & fetches Top-K chunks)
                    │
                    ▼
           [Grounded QA Generator]
 (System Prompt + Retrieved Context + Question -> LLM)
                    │
                    ▼
         Streamlit UI Display
   (Grounded Answer + Source List + Context Chunks)
```

---

## 🤖 Future Multi-Agent Architecture (Target Architecture)

In future milestones, specialized AI agents will consume this RAG Knowledge Base to automate project management tasks:

```text
                             RAG Knowledge Base
                                     │
                                     ▼
                            Agent Orchestrator
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
  Scope Agent                   Risk Agent                   Blocker Agent
 (Extracts deliverables        (Detects timeline             (Identifies pending
  & requirements)               & resource risks)             API keys & blockers)
       │                             │                             │
       └─────────────────────────────┼─────────────────────────────┘
                                     ▼
                        Documentation Generator Agent
                                     │
                                     ▼
                             Forecasting Agent
                                     │
                                     ▼
                          Project Health Scoring
```

---

## 🛠️ Technology Stack & Rationale

| Technology | Purpose | Selection Rationale |
| :--- | :--- | :--- |
| **Python 3.11** | Core Language | Simple, clean syntax, standard for AI & data engineering. |
| **Streamlit** | UI Framework | Rapid, interactive web dashboard in pure Python. |
| **PyMuPDF (`pymupdf`)** | PDF Extraction | Fast, high-accuracy page-by-page text extraction. |
| **python-docx** | DOCX Extraction | Native extraction of paragraphs and table data from Word files. |
| **pandas** | CSV Processing | Converts tabular rows into semantic key-value strings for RAG. |
| **Sentence Transformers** | Embeddings | Local, lightweight `all-MiniLM-L6-v2` embedding model (384-d). |
| **ChromaDB** | Vector Database | Persistent, serverless vector store supporting project isolation. |
| **Google Gemini API** | Grounded QA | Powerful LLM API for context-based answer generation. |
| **python-dotenv** | Config | Securely manages API keys via `.env` files. |

---

## 🧠 Educational RAG Explanation (For Mentor Q&A)

### 1. Why do we chunk text?
- **Embedding Model Limits**: Models like `all-MiniLM-L6-v2` have maximum token input limits (256/512 tokens).
- **Retrieval Precision**: Searching across 400-word paragraphs allows the vector store to pinpoint exact answers (e.g., a specific meeting blocker) rather than returning a 30-page document.

### 2. Why is overlap needed?
- **Sentence Boundary Protection**: A hard split at word 400 might cut a critical requirement sentence in half.
- **Context Continuity**: Overlapping chunks by 50 words ensures boundary sentences appear intact in consecutive chunks.

### 3. Why not embed an entire 20-page PDF as a single vector?
- **Information Dilution**: A single 384-dimensional vector cannot encode 20 pages of diverse technical details. Specific facts get lost in the average vector representation.

---

## 🚀 Installation & Running

### 1. Clone & Set Up Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```

Inside `.env`:
```text
GEMINI_API_KEY=your_actual_gemini_api_key_here
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma
```

### 4. Generate Sample Dataset
Run the sample data generator to create test documents (`sample_proposal.pdf`, `sample_srs.docx`, `sample_meeting_notes.txt`, `sample_tasks.csv`):
```bash
python generate_sample_data.py
```

### 5. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 🧪 Testing & Verification

Run automated unit and integration tests using `pytest`:
```bash
pytest tests/
```

Expected output:
```text
tests/test_chunking.py .                                                 [ 12%]
tests/test_ingestion.py ....                                             [ 62%]
tests/test_retrieval.py ...                                              [100%]
======================= 8 passed in 18.19s =======================
```

---

## 📁 Project Structure

```text
ai-project-intelligence/
├── app.py                      # Streamlit UI Dashboard
├── generate_sample_data.py     # Test dataset generator
├── requirements.txt            # Python dependencies
├── .env.example                # Sample configuration template
├── .gitignore                  # Git ignore rules
├── README.md                   # System documentation
├── conftest.py                 # Pytest path resolution
│
├── ingestion/                  # Document loading & text cleaning
│   ├── __init__.py
│   ├── pdf_loader.py           # PyMuPDF text loader
│   ├── docx_loader.py          # python-docx loader
│   ├── csv_loader.py           # pandas semantic loader
│   ├── txt_loader.py           # TXT file loader with encoding fallbacks
│   └── document_processor.py   # Text normalizer & unified entry point
│
├── rag/                        # Core RAG engine
│   ├── __init__.py
│   ├── chunker.py              # Text chunker with educational comments
│   ├── embeddings.py           # SentenceTransformers wrapper
│   ├── vector_store.py         # Persistent ChromaDB collection manager
│   ├── retriever.py            # Similarity search & Top-K retriever
│   └── qa.py                   # Grounded LLM prompt builder & QA
│
├── models/                     # Data schemas
│   ├── __init__.py
│   └── schemas.py              # Project, Document, Chunk dataclasses
│
├── utils/                      # Helper utilities
│   ├── __init__.py
│   └── helpers.py              # Sanitization & source formatting helpers
│
├── sample_data/                # Generated sample test documents
│   ├── sample_proposal.pdf
│   ├── sample_srs.docx
│   ├── sample_meeting_notes.txt
│   └── sample_tasks.csv
│
└── tests/                      # Automated test suite
    ├── test_ingestion.py
    ├── test_chunking.py
    └── test_retrieval.py
```

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'rag'` | Python path not set | Run `conftest.py` or use `python -m pytest tests/` |
| `UnicodeEncodeError` in Windows console | Windows CP1252 character mapping | Use ASCII symbols in print statements |
| `LLM API Key Not Configured` | Missing `.env` key | Set `GEMINI_API_KEY` in `.env` |
| `PDF file contains no readable text` | Scanned image PDF | Use digital text PDFs or add OCR in future milestones |

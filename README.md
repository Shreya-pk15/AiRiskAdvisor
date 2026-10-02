# 🛡️ AI Project Intelligence & Risk Advisor

A modular, document-grounded Retrieval-Augmented Generation (RAG) application built with **Python**, **Streamlit**, **Sentence Transformers**, **ChromaDB**, **Google Gemini API**, and **Groq API**.

Designed for software project teams, this application ingests heterogeneous project artifacts (PDF proposals, DOCX specifications, TXT meeting notes, CSV task lists, XLSX defect trackers), indexes them into a project-isolated vector database, and provides grounded AI answers and structured project intelligence agents.

---

## 📌 Project Overview

### Milestone 1 — Document-Grounded RAG Pipeline
1. Ingests PDF, DOCX, CSV, TXT, and XLSX files.
2. Normalizes text and transforms tabular CSV/XLSX rows into semantic key-value strings.
3. Segments documents into overlapping text chunks (~400 words, 50-word overlap).
4. Generates 384-dimensional dense vector embeddings using `all-MiniLM-L6-v2`.
5. Indexes chunks into persistent, project-isolated ChromaDB collections.
6. Retrieves top-K relevant context chunks based on vector similarity search.
7. Executes a strictly grounded prompt against the LLM to generate verifiable answers with source document attribution.

---

## 🤖 Milestone 2 — Project Intelligence Agents

### 1. Scope & Deliverable Extraction Agent (Gemini)
Automatically extracts structured project goals, deliverables, milestones, timeline references, and responsibilities using the Google Gemini API (`GEMINI_SCOPE_MODEL=gemini-3.6-flash`).

### 2. Risk Detection & Delivery Forecasting Agent (Groq)
Analyzes project documents to detect evidence-based project risks (Schedule, Dependency, Resource, Technical, Quality, Planning, Delivery) and generate forward-looking delivery forecasts using the Groq API (`GROQ_RISK_MODEL=openai/gpt-oss-120b`).

### 3. Blocker & Action Item Identification Agent (Gemini)
Identifies technical/dependency/approval blockers, pending decisions, unresolved issues, and assigned action items using the Google Gemini API (`GEMINI_BLOCKER_MODEL=gemini-3.6-flash`).

---

## 📋 Milestone 3 — Documentation, Health & Conversational Intelligence

### 1. Documentation Generation Agent (Gemini)
Reuses Scope, Risk, and Blocker agent outputs (when available in the session) to generate:
* **User Stories** — MoSCoW priorities, acceptance criteria, source/evidence per story
* **Risk Register** — structured register from identified risks (no invented owners/deadlines)
* **Action Items** — formal list from blockers, decisions, and documented actions

Model: `GEMINI_DOCGEN_MODEL` (defaults to `gemini-3.6-flash`).

### 2. Project Health Scoring Module (deterministic, no LLM)
Computes an overall **0–100** score from Milestone 2 outputs:
* Scope Clarity (25%)
* Timeline Risk (30%)
* Blocker Status (25%)
* Delivery Risk (20%)

Includes classification (Healthy / Moderate / At Risk / Critical), confidence/data-sufficiency notes, dimension breakdown, and recommendations.

### 3. Conversational Project Intelligence Assistant (Gemini)
Multi-turn chat grounded in ChromaDB retrieval plus cached Scope/Risk/Blocker/Health outputs. Supports follow-up questions, source badges, and safe refusal when documents do not contain the answer.

Model: `GEMINI_CHAT_MODEL` (defaults to `gemini-3.6-flash`).

### Streamlit dashboard flow
1. **Ingest Project Artifacts** → **Index & Process**
2. **Grounded Project Q&A** (Milestone 1 RAG)
3. **Project Intelligence** — Scope & Deliverables | Risks & Delivery | Blockers & Actions
4. **Documentation** — User Stories | Risk Register | Action Items (+ CSV export)
5. **Project Health** — Evaluate & dimension breakdown
6. **Conversational Project Assistant** — chat with suggested prompts

Workspace name in the sidebar acts as **project_id** for ChromaDB isolation.

---

### ⚙️ Environment Configuration

Configure in `.env`:

```env
# Gemini API Configuration (Scope & Blocker Agents)
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_SCOPE_MODEL=gemini-3.6-flash
GEMINI_BLOCKER_MODEL=gemini-3.6-flash
GEMINI_DOCGEN_MODEL=gemini-3.6-flash
GEMINI_CHAT_MODEL=gemini-3.6-flash

# Groq API Configuration (Risk Agent)
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_RISK_MODEL=openai/gpt-oss-120b

# Common Timeout & Vector Settings
AGENT_TIMEOUT_SECONDS=30
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma
```

---

### 🔄 Blocker Agent RAG Retrieval & Gemini Architecture

```text
Selected Workspace (project_id)
             │
             ▼
Targeted Semantic RAG Queries:
  • "meeting blockers unresolved issues"
  • "pending decisions technology approval clarification"
  • "action items assigned tasks owner deadline"
  • "sprint progress issues defect tracker updates"
  • "tasks waiting for people teams approval dependency"
             │
             ▼
      [Retriever Engine]
 (Filtered by project_id in ChromaDB)
             │
             ▼
   [Deduplicated Context]
             │
             ▼
    [GeminiProvider]
 (1 Structured API Generation Call)
             │
             ▼
  [BlockerActionOutput]
   (Validated Pydantic Schema)
```

---

### 🛡️ Grounding Rules & Grounded Output Spec

* **No Hallucination**: The agent extracts **ONLY** items directly supported by retrieved chunks.
* **Missing Field Fallback**: If an assignee, deadline, or attribute is omitted in documents, it sets:
  ```text
  Not specified in the available project documents.
  ```
* **Empty Category Fallback**: If no items are found for a category, it displays an informative notice (e.g. `"No blockers identified from the available project documents."`).
* **Structured Output Schema (`BlockerActionOutput` / `BlockerActionResult`)**:
  ```json
  {
    "blockers": [
      {
        "description": "Payment gateway deployment blocked waiting for security approval",
        "impact": "Delays production release",
        "status": "Active",
        "owner": "Security Team",
        "source": "Sprint_Notes.txt",
        "evidence": "Payment gateway deployment is blocked waiting for security team approval."
      }
    ],
    "pending_decisions": [
      {
        "decision": "Selection of database migration tool",
        "owner": "Architecture Team",
        "status": "Pending",
        "source": "Sprint_Notes.txt"
      }
    ],
    "unresolved_issues": [
      {
        "issue": "Unresolved DB migration pipeline compatibility",
        "status": "Open",
        "source": "Sprint_Notes.txt"
      }
    ],
    "action_items": [
      {
        "action": "Update unit test suite",
        "assignee": "Rahul",
        "deadline": "Friday",
        "status": "Open",
        "priority": "High",
        "source": "Sprint_Notes.txt",
        "evidence": "Rahul to update unit test suite by Friday."
      }
    ],
    "sources": ["Sprint_Notes.txt"]
  }
  ```

---

### 🖥️ Streamlit UI Integration

In Section **3. Project Intelligence**, click **`[🛑 Blockers & Action Items]`**.

The interface renders:
* **Blockers Table**: Blocker description, status, evidence, source.
* **Pending Decisions Table**: Decision, owner, status, source.
* **Unresolved Issues Table**: Issue description, status, evidence, source.
* **Action Items Table**: Action, assignee, deadline, status, priority, evidence, source.
* **Agent Metadata Card**:
  ```text
  Agent: Blocker & Action Agent
  Provider: Gemini
  Execution Time: X seconds
  Status: Success/Failed
  ```

---

## 🛠️ Technology Stack & Rationale

| Technology | Purpose | Selection Rationale |
| :--- | :--- | :--- |
| **Python 3.11** | Core Language | Standard for AI & data engineering. |
| **Streamlit** | UI Framework | Interactive web dashboard in pure Python. |
| **PyMuPDF / python-docx / pandas** | Ingestion | Extraction from PDF, DOCX, CSV, TXT, and XLSX. |
| **Sentence Transformers** | Embeddings | Local `all-MiniLM-L6-v2` embedding model (384-d). |
| **ChromaDB** | Vector Database | Persistent vector store with project workspace isolation. |
| **google-genai** | Scope & Blocker LLM | Google SDK for Gemini models. |
| **groq** | Risk Agent LLM | High-speed Groq SDK for Llama 3 models. |
| **python-dotenv** | Config | Environment variable management. |

---

## 🚀 Installation & Running

### 1. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env` file:
```bash
cp .env.example .env
```
Ensure `GEMINI_API_KEY`, `GEMINI_SCOPE_MODEL`, `GEMINI_BLOCKER_MODEL`, `GROQ_API_KEY`, `GROQ_RISK_MODEL`, and `AGENT_TIMEOUT_SECONDS` are set.

### 3. Generate Sample Data & Run UI
```bash
python generate_sample_data.py
streamlit run app.py
```

---

## 🧪 Testing

Run full automated test suite:
```bash
pytest
```

Milestone 3 focused suites:
```bash
pytest tests/test_documentation_agent.py tests/test_health_scorer.py tests/test_conversational_agent.py
```

To run only the Blocker Agent test suite:
```bash
pytest tests/test_blocker_agent.py
```

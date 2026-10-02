"""
AI Project Intelligence & Risk Advisor — Professional Dashboard UI

A complete, modern project management and risk intelligence platform
for software engineering teams and student project groups.

Architecture & User Flow:
Upload Documents → Index Documents → AI Analysis → Project Dashboard → Detailed Insights → AI Assistant
"""

import io
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from utils.helpers import is_allowed_file
from ingestion.document_processor import DocumentProcessor
from rag.chunker import TextChunker
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.retriever import Retriever
from rag.qa import AnswerGenerator
from agents.scope_agent import ScopeExtractionAgent
from agents.risk_agent import RiskDetectionAgent
from agents.blocker_agent import BlockerActionAgent
from agents.documentation_agent import DocumentationAgent
from agents.health_scorer import ProjectHealthScorer
from agents.conversational_agent import ConversationalProjectAssistant
from llm.gemini_provider import GeminiProvider
from llm.groq_provider import GroqProvider

# ── Page Configuration ──
st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional SaaS Dashboard Dark CSS ──
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #0B1120;
        color: #E2E8F0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Brand Header */
    .sidebar-brand {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
    }
    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #6366F1, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.3;
    }
    .sidebar-tagline {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Sidebar Navigation Pills */
    .stRadio > div {
        gap: 6px;
    }
    .stRadio > div > label {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 14px;
        color: #94A3B8;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    .stRadio > div > label:hover {
        background: rgba(99, 102, 241, 0.15);
        color: #E2E8F0;
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Primary & Secondary Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
    }

    /* Inputs, Selects & Sliders */
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] select {
        border-radius: 8px !important;
        background-color: #1E293B !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #F8FAFC !important;
    }
    .stSlider [data-baseweb="slider"] { color: #6366F1; }
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
        border-radius: 6px !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1E293B;
        border-radius: 10px;
        padding: 5px;
        gap: 6px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #6366F1 !important;
        color: #FFFFFF !important;
    }

    /* Page Header Banner */
    .page-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .page-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
    }
    .page-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-top: 0.3rem;
    }

    /* Top KPI Metric Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(12px);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748B;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.3rem;
    }

    /* Glass Dashboard Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.4rem;
        backdrop-filter: blur(12px);
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Status Badges */
    .status-badge-ontrack {
        background: #052e16; color: #4ade80;
        padding: 4px 12px; border-radius: 20px;
        font-weight: 700; font-size: 0.82rem;
        border: 1px solid #166534;
    }
    .status-badge-atrisk {
        background: #1c1408; color: #fbbf24;
        padding: 4px 12px; border-radius: 20px;
        font-weight: 700; font-size: 0.82rem;
        border: 1px solid #92400e;
    }
    .status-badge-delayed {
        background: #1f0e0e; color: #f87171;
        padding: 4px 12px; border-radius: 20px;
        font-weight: 700; font-size: 0.82rem;
        border: 1px solid #991b1b;
    }
    .status-badge-nodata {
        background: #1a2333; color: #94A3B8;
        padding: 4px 12px; border-radius: 20px;
        font-weight: 700; font-size: 0.82rem;
        border: 1px solid #334155;
    }

    /* Severity Badges */
    .sev-high {
        background: #1f0e0e; color: #f87171;
        padding: 3px 9px; border-radius: 6px;
        font-size: 0.75rem; font-weight: 700;
        border: 1px solid #991b1b;
    }
    .sev-medium {
        background: #1c1408; color: #fbbf24;
        padding: 3px 9px; border-radius: 6px;
        font-size: 0.75rem; font-weight: 700;
        border: 1px solid #92400e;
    }
    .sev-low {
        background: #0c1a2e; color: #60a5fa;
        padding: 3px 9px; border-radius: 6px;
        font-size: 0.75rem; font-weight: 700;
        border: 1px solid #1d4ed8;
    }

    /* MoSCoW Priority Badges */
    .moscow-must   { background: #052e16; color: #4ade80; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 700; border: 1px solid #166534; }
    .moscow-should { background: #0c1a2e; color: #60a5fa; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 700; border: 1px solid #1d4ed8; }
    .moscow-could  { background: #1c1408; color: #fbbf24; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 700; border: 1px solid #92400e; }
    .moscow-wont   { background: #1a2333; color: #64748B; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 700; border: 1px solid #334155; }

    /* Source Citation Tag */
    .source-tag {
        display: inline-flex; align-items: center; gap: 4px;
        background: rgba(99, 102, 241, 0.12);
        color: #A5B4FC;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px; margin-bottom: 5px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* Risk Card */
    .risk-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .risk-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
    }

    /* Evidence Quote */
    .evidence-quote {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #6366F1;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #CBD5E1;
        font-style: italic;
        margin: 0.6rem 0;
    }

    /* Recommendation Box */
    .rec-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        color: #C7D2FE;
        margin-top: 0.6rem;
    }

    /* Forecast Box */
    .forecast-box {
        background: rgba(251, 191, 36, 0.07);
        border-left: 4px solid #F59E0B;
        border-top: 1px solid rgba(251, 191, 36, 0.2);
        border-right: 1px solid rgba(251, 191, 36, 0.2);
        border-bottom: 1px solid rgba(251, 191, 36, 0.2);
        padding: 1.2rem 1.4rem;
        border-radius: 8px;
        font-size: 0.95rem;
        color: #FCD34D;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    /* Milestone Timeline Card */
    .milestone-item {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* File Badge */
    .file-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.06);
        color: #CBD5E1;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Status Indicators */
    .api-status-ok   { display: inline-flex; align-items: center; gap: 5px; background: #052e16; color: #4ade80; padding: 3px 9px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid #166534; }
    .api-status-warn { display: inline-flex; align-items: center; gap: 5px; background: #1f0e0e; color: #f87171; padding: 3px 9px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; border: 1px solid #991b1b; }

    /* Enforced light surfaces and readable text, independent of OS theme. */
    :root { color-scheme: light; }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #FFFFFF !important;
        color: #1F2937 !important;
    }
    [data-testid="stHeader"] { background: rgba(255, 255, 255, 0.96) !important; }
    section[data-testid="stSidebar"] {
        background: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] * { color: #1F2937; }
    .page-header, .kpi-card, .glass-card, .risk-card, .milestone-item,
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF !important;
        color: #1F2937 !important;
        border-color: #E2E8F0 !important;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
    }
    .page-title, .kpi-value { color: #1F2937 !important; }
    .page-subtitle, .kpi-sub, .kpi-label { color: #64748B !important; }
    .stRadio > div > label {
        background: #F1F5F9 !important;
        color: #334155 !important;
        border-color: #E2E8F0 !important;
    }
    .stRadio > div > label:hover { background: #E2E8F0 !important; color: #0F172A !important; }
    .stTabs [data-baseweb="tab"] { color: #475569 !important; }
    .stTabs [aria-selected="true"] { background: #0F766E !important; color: #FFFFFF !important; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
    [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #1F2937 !important;
        border-color: #CBD5E1 !important;
    }
    [style*="background: linear-gradient(135deg, #0F172A"],
    [style*="background: rgba(30, 41, 59"],
    [style*="background: rgba(15, 23, 42"] {
        background: #FFFFFF !important;
        border-color: #E2E8F0 !important;
    }
    [style*="color: #F8FAFC"], [style*="color: #E2E8F0"],
    [style*="color: #CBD5E1"], [style*="color: #C7D2FE"],
    [style*="color: #A5B4FC"], [style*="color: #94A3B8"] {
        color: #334155 !important;
    }
    [style*="color: #38BDF8"] { color: #0F766E !important; }
    [style*="color: #60A5FA"] { color: #1D4ED8 !important; }
    .status-badge-ontrack, .moscow-must { background: #DCFCE7 !important; color: #166534 !important; border-color: #86EFAC !important; }
    .status-badge-atrisk, .moscow-could, .sev-medium { background: #FEF3C7 !important; color: #92400E !important; border-color: #FCD34D !important; }
    .status-badge-delayed, .sev-high { background: #FEE2E2 !important; color: #991B1B !important; border-color: #FCA5A5 !important; }
    .status-badge-nodata, .moscow-wont { background: #F1F5F9 !important; color: #475569 !important; border-color: #CBD5E1 !important; }
    .sev-low, .moscow-should { background: #DBEAFE !important; color: #1E40AF !important; border-color: #93C5FD !important; }
    .source-tag { background: #CCFBF1 !important; color: #115E59 !important; border-color: #99F6E4 !important; }
    .evidence-quote { background: #F8FAFC !important; color: #334155 !important; border-left-color: #0F766E !important; }
    .forecast-box { background: #FFFBEB !important; color: #78350F !important; border-color: #FDE68A !important; }
    </style>
""", unsafe_allow_html=True)


# ── RAG Pipeline Components Initialization ──
@st.cache_resource
def get_rag_components():
    embedder = EmbeddingManager()
    vector_store = VectorStoreManager()
    retriever = Retriever(embedding_manager=embedder, vector_store_manager=vector_store)
    processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=400, overlap=50)
    qa_generator = AnswerGenerator()
    return embedder, vector_store, retriever, processor, chunker, qa_generator


embedder, vector_store, retriever, processor, chunker, qa_generator = get_rag_components()


# ── Read Credentials Securely from .env (Never Expose in UI) ──
_gemini_key     = os.getenv("GEMINI_API_KEY", "")
_groq_key       = os.getenv("GROQ_API_KEY", "")
scope_model     = os.getenv("GEMINI_SCOPE_MODEL", "gemini-1.5-flash")
blocker_model   = os.getenv("GEMINI_BLOCKER_MODEL", "gemini-1.5-flash")
docgen_model    = os.getenv("GEMINI_DOCGEN_MODEL", "gemini-1.5-flash")
chat_model      = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
groq_risk_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-20b")

_gemini_valid   = bool(_gemini_key) and not _gemini_key.startswith("AQ.") and _gemini_key != "your_gemini_api_key_here"
_groq_valid     = bool(_groq_key) and _groq_key != "your_groq_api_key_here"


# ── Sidebar Navigation & Configuration ──
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-title">⚡ AI Project Intelligence &amp; Risk Advisor</div>
            <div class="sidebar-tagline">Project Management Intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    # Workspace Selector
    st.caption("PROJECT WORKSPACE")
    project_name = st.text_input(
        "Workspace",
        value=st.session_state.get("project_name", "Main Workspace"),
        label_visibility="collapsed",
        placeholder="e.g. Smart Campus Application"
    )
    if not project_name or not project_name.strip():
        project_name = "Main Workspace"
    st.session_state["project_name"] = project_name

    # Knowledge Base Status Chip in Sidebar
    collection = vector_store.get_or_create_collection(project_name)
    kb_count = collection.count()
    if kb_count > 0:
        st.markdown(f"<div style='font-size: 0.78rem; color: #4ade80; margin-bottom: 1rem;'>🟢 Knowledge Base: Ready ({kb_count} Chunks)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 0.78rem; color: #94A3B8; margin-bottom: 1rem;'>⚪ Knowledge Base: Not Indexed</div>", unsafe_allow_html=True)

    # Primary Navigation
    st.caption("NAVIGATION")
    NAV_PAGES = [
        "📊 Dashboard",
        "📁 Documents",
        "🎯 Scope & Deliverables",
        "⚠️ Risks",
        "🛑 Blockers & Actions",
        "🩺 Project Health",
        "📄 Documentation",
        "💬 AI Assistant"
    ]

    # Initialize active nav page
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "📊 Dashboard"

    # Ensure index exists in list
    current_index = NAV_PAGES.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in NAV_PAGES else 0

    selected_page = st.radio(
        "Navigation",
        options=NAV_PAGES,
        index=current_index,
        label_visibility="collapsed"
    )
    st.session_state["nav_page"] = selected_page

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.2rem 0;'>", unsafe_allow_html=True)

    # Automatically scale retrieval depth with the size of the indexed workspace.
    top_k = retriever.automatic_top_k(kb_count)
    st.caption(f"RETRIEVAL DEPTH (AUTO): {top_k} chunks")

    # AI Engine Provider
    st.caption("ACTIVE AI ENGINE")
    llm_engine = st.selectbox(
        "AI Provider",
        options=["Groq — Fast & Free (Recommended)", "Google Gemini"],
        index=0,
        label_visibility="collapsed"
    )

    # Secure API Status (No keys shown)
    groq_badge = '<span class="api-status-ok">✅ Groq Active</span>' if _groq_valid else '<span class="api-status-warn">❌ Groq Missing</span>'
    gem_badge  = '<span class="api-status-ok">✅ Gemini Ready</span>' if _gemini_valid else '<span class="api-status-warn">⚠️ Gemini Inactive</span>'
    st.markdown(f"<div style='margin-top: 6px;'>{groq_badge}&nbsp;&nbsp;{gem_badge}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.2rem 0;'>", unsafe_allow_html=True)
    st.caption("SUPPORTED FORMATS")
    st.markdown("""
        <div>
            <span class="file-badge">PDF</span>
            <span class="file-badge">DOCX</span>
            <span class="file-badge">CSV</span>
            <span class="file-badge">TXT</span>
            <span class="file-badge">XLSX</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.7rem; color: #475569; margin-top: 1rem; text-align: center;'>Credentials secured in .env · Zero UI leaks</div>", unsafe_allow_html=True)


# ── Intelligent LLM Provider Resolver ──
def resolve_agent_provider(gemini_model_id: str):
    """Resolve provider from .env config without exposing any secrets in the UI."""
    if llm_engine.startswith("Groq"):
        if _groq_valid:
            try:
                return GroqProvider(model_name=groq_risk_model, api_key=_groq_key)
            except Exception:
                pass
    else:
        if _gemini_valid:
            try:
                return GeminiProvider(model_name=gemini_model_id, api_key=_gemini_key)
            except Exception:
                pass
    # Automatic Failover to Groq
    if _groq_valid:
        try:
            return GroqProvider(model_name=groq_risk_model, api_key=_groq_key)
        except Exception:
            pass
    return None


# ── Helper: Run All Intelligence Agents in Sequence ──
def store_agent_result(result: Dict[str, Any], session_key: str, agent_name: str) -> bool:
    """Store successful extraction results and report agent failures accurately."""
    metadata = result.get("metadata", {})
    if metadata.get("status") != "Success":
        error = metadata.get("error") or "The agent returned an unsuccessful status."
        st.error(f"{agent_name} failed: {error}")
        return False

    st.session_state[session_key] = result.get("data", {})
    note = metadata.get("note")
    if note:
        st.warning(f"{agent_name}: {note}")
    return True


def run_all_project_agents():
    """Execute Scope, Risk, Blocker, and Health evaluation in sequence."""
    collection = vector_store.get_or_create_collection(project_name)
    if collection.count() == 0:
        st.warning("No documents indexed in this workspace yet. Please upload and index documents first.")
        return False

    with st.status("⚡ Running Project Intelligence Agents...", expanded=True) as status:
        # Step 1: Scope & Deliverables
        st.write("🔍 Running Scope & Deliverables Extraction Agent...")
        scope_provider = resolve_agent_provider(scope_model)
        scope_agent = ScopeExtractionAgent(
            provider=scope_provider,
            retriever=retriever,
            model_name=groq_risk_model if llm_engine.startswith("Groq") else scope_model
        )
        s_res = scope_agent.run(project_id=project_name, top_k=top_k)
        scope_ok = store_agent_result(s_res, f"scope_data_{project_name}", "Scope & Deliverables extraction")
        scope_data = st.session_state.get(f"scope_data_{project_name}", {})

        # Step 2: Risks & Delivery Forecasting
        st.write("⚠️ Running Risk Detection & Delivery Forecasting Agent...")
        risk_provider = resolve_agent_provider(groq_risk_model)
        risk_agent = RiskDetectionAgent(
            provider=risk_provider,
            retriever=retriever,
            model_name=groq_risk_model
        )
        r_res = risk_agent.run(project_id=project_name, top_k=top_k)
        risk_ok = store_agent_result(r_res, f"risk_data_{project_name}", "Risk analysis")
        risk_data = st.session_state.get(f"risk_data_{project_name}", {})

        # Step 3: Blockers & Action Items
        st.write("🛑 Running Blocker & Action Item Identification Agent...")
        blocker_provider = resolve_agent_provider(blocker_model)
        blocker_agent = BlockerActionAgent(
            provider=blocker_provider,
            retriever=retriever,
            model_name=groq_risk_model if llm_engine.startswith("Groq") else blocker_model
        )
        b_res = blocker_agent.run(project_id=project_name, top_k=top_k)
        blocker_ok = store_agent_result(b_res, f"blocker_data_{project_name}", "Blocker & Action extraction")
        blocker_data = st.session_state.get(f"blocker_data_{project_name}", {})

        # Step 4: Health Scoring
        st.write("🩺 Evaluating Deterministic Project Health Score...")
        health_scorer = ProjectHealthScorer()
        h_res = health_scorer.evaluate_health(
            scope_data=scope_data,
            risk_data=risk_data,
            blocker_data=blocker_data,
            project_id=project_name,
            retriever=retriever,
            top_k=top_k
        )
        st.session_state[f"health_data_{project_name}"] = h_res.model_dump()

        analysis_ok = scope_ok and risk_ok and blocker_ok
        status.update(
            label="✅ Project Analysis Complete!" if analysis_ok else "⚠️ Analysis completed with errors",
            state="complete" if analysis_ok else "error",
            expanded=not analysis_ok,
        )
    return analysis_ok


# =============================================================================
# PAGE 1: 📊 DASHBOARD (MAIN LANDING PAGE)
# =============================================================================
if selected_page == "📊 Dashboard":
    # Retrieve current state
    scope_data = st.session_state.get(f"scope_data_{project_name}")
    risk_data = st.session_state.get(f"risk_data_{project_name}")
    blocker_data = st.session_state.get(f"blocker_data_{project_name}")
    health_data = st.session_state.get(f"health_data_{project_name}")
    doc_registry = st.session_state.get(f"uploaded_files_{project_name}", [])
    last_indexed = st.session_state.get(f"last_indexed_{project_name}", "Not yet indexed")

    # Project Header
    kb_ready = kb_count > 0
    kb_status_badge = '<span class="status-badge-ontrack">✓ Ready</span>' if kb_ready else '<span class="status-badge-nodata">⚪ Not Indexed</span>'
    num_docs = len(doc_registry) if doc_registry else (1 if kb_ready else 0)

    st.markdown(f"""
        <div class="page-header">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 2px; text-transform: uppercase;">PROJECT OVERVIEW</div>
                <div class="page-title">{project_name}</div>
                <div class="page-subtitle">AI-grounded delivery forecasting, risk tracking, and real-time blocker intelligence.</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.82rem; color: #94A3B8;">Knowledge Base: {kb_status_badge}</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Last Updated: <strong style="color: #CBD5E1;">{last_indexed}</strong></div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">Indexed Artifacts: <strong style="color: #CBD5E1;">{num_docs} Documents</strong> ({kb_count} Chunks)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Prompt user to index if knowledge base is empty
    if kb_count == 0:
        st.info("👋 Welcome! To get started, upload your project documents (PDF, DOCX, TXT, CSV, XLSX) on the **📁 Documents** page and click **[ Index & Analyze Project ]**.", icon="💡")
        if st.button("📁 Go to Documents Page", type="primary"):
            st.session_state["nav_page"] = "📁 Documents"
            st.rerun()
    else:
        # Action Bar
        act_col1, act_col2, act_col3 = st.columns([1.5, 1, 1])
        with act_col1:
            run_all_btn = st.button("⚡ Run Full Project Analysis", type="primary", use_container_width=True)
            if run_all_btn:
                if run_all_project_agents():
                    st.success("Project intelligence analysis refreshed successfully.")
                    st.rerun()
        with act_col2:
            if st.button("📁 Manage Documents", use_container_width=True):
                st.session_state["nav_page"] = "📁 Documents"
                st.rerun()
        with act_col3:
            if st.button("💬 Ask AI Assistant", use_container_width=True):
                st.session_state["nav_page"] = "💬 AI Assistant"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # 4-Card KPI Row
        score_val = health_data.get("overall_score", 0.0) if health_data else 0.0
        health_class = health_data.get("classification", "Not Evaluated") if health_data else "Not Evaluated"
        risks_list = risk_data.get("risks", []) if risk_data else []
        high_risks = len([r for r in risks_list if r.get("severity", "").upper() == "HIGH"])
        blockers_list = blocker_data.get("blockers", []) if blocker_data else []
        open_blockers = len([b for b in blockers_list if b.get("status", "Open").lower() != "resolved"])
        deliverables_list = scope_data.get("deliverables", []) if scope_data else []
        pending_deliv = len([d for d in deliverables_list if d.get("status", "").lower() != "completed"])

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Project Health</div>
                    <div class="kpi-value" style="color: {'#4ade80' if score_val >= 80 else ('#fbbf24' if score_val >= 60 else '#f87171')};">
                        {int(score_val)}<span style="font-size: 1.1rem; color: #64748B;">/100</span>
                    </div>
                    <div class="kpi-sub"><span class="{'status-badge-ontrack' if health_class == 'Healthy' else ('status-badge-atrisk' if health_class == 'Moderate' else 'status-badge-delayed')}">{health_class.upper()}</span></div>
                </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">High Severity Risks</div>
                    <div class="kpi-value" style="color: {'#f87171' if high_risks > 0 else '#4ade80'};">{high_risks}</div>
                    <div class="kpi-sub">{len(risks_list)} total risks identified</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Open Blockers</div>
                    <div class="kpi-value" style="color: {'#f87171' if open_blockers > 0 else '#4ade80'};">{open_blockers}</div>
                    <div class="kpi-sub">{len(blocker_data.get('action_items', [])) if blocker_data else 0} action items tracked</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Pending Deliverables</div>
                    <div class="kpi-value" style="color: #60a5fa;">{pending_deliv}</div>
                    <div class="kpi-sub">{len(deliverables_list)} deliverables identified</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Dashboard Content Split: Health Card & Delivery Forecast
        dash_col1, dash_col2 = st.columns([1.1, 1], gap="large")

        with dash_col1:
            st.markdown("### 🩺 Project Health Status")
            if not health_data:
                st.info("Health score not yet evaluated. Click '⚡ Run Full Project Analysis' or visit Project Health.")
            else:
                dimensions = health_data.get("dimensions", {})
                key_factors = health_data.get("key_factors", [])
                badge_cls = "status-badge-ontrack" if health_class == "Healthy" else ("status-badge-atrisk" if health_class == "Moderate" else "status-badge-delayed")

                st.markdown(f"""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem;">
                            <div>
                                <span style="font-size: 2.6rem; font-weight: 800; color: #38BDF8;">{int(score_val)}</span>
                                <span style="font-size: 1.2rem; color: #94A3B8;"> / 100</span>
                            </div>
                            <span class="{badge_cls}" style="font-size: 1rem; padding: 6px 18px;">{health_class.upper()}</span>
                        </div>
                """, unsafe_allow_html=True)

                # 4 Dimension Progress Bars
                dim_names = [
                    ("Scope Clarity", "scope_clarity"),
                    ("Timeline", "timeline_risk"),
                    ("Blockers", "blocker_status"),
                    ("Delivery Risk", "delivery_risk"),
                ]
                for label, key in dim_names:
                    d_obj = dimensions.get(label, {})
                    d_score = d_obj.get("score", 70.0)
                    st.write(f"**{label}** — {int(d_score)}%")
                    st.progress(int(d_score))

                # Why is the project currently at risk?
                st.markdown("<div style='margin-top: 1.2rem; font-weight: 700; color: #F8FAFC;'>Why is the project currently at risk?</div>", unsafe_allow_html=True)
                if key_factors:
                    for kf in key_factors[:4]:
                        st.markdown(f"• <span style='color: #CBD5E1; font-size: 0.9rem;'>{kf}</span>", unsafe_allow_html=True)
                else:
                    st.caption("No negative factors recorded.")

                st.markdown("</div>", unsafe_allow_html=True)

        with dash_col2:
            st.markdown("### 📅 Delivery Forecast & Schedule")
            if not risk_data:
                st.info("Delivery forecast not yet calculated. Run full project analysis to generate.")
            else:
                del_status = risk_data.get("delivery_status", "Insufficient Data")
                forecast_obj = risk_data.get("delivery_forecast", {})
                forecast_narrative = forecast_obj.get("forecast_analysis") or forecast_obj.get("reason") or "No forward-looking analysis available."
                planned_date = forecast_obj.get("planned_completion", "Not specified in documents")
                est_date = forecast_obj.get("estimated_completion", "Pending analysis")

                del_badge = "status-badge-ontrack" if del_status == "On Track" else ("status-badge-atrisk" if del_status == "At Risk" else "status-badge-delayed")

                st.markdown(f"""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <span style="font-weight: 700; color: #E2E8F0;">Forecast Delivery Status:</span>
                            <span class="{del_badge}">{del_status}</span>
                        </div>
                        <div style="display: flex; gap: 20px; margin-bottom: 1.2rem;">
                            <div>
                                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">PLANNED COMPLETION</div>
                                <div style="font-size: 1.05rem; font-weight: 700; color: #94A3B8;">{planned_date}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">FORECAST COMPLETION</div>
                                <div style="font-size: 1.05rem; font-weight: 700; color: #FCD34D;">{est_date}</div>
                            </div>
                        </div>
                        <div class="forecast-box" style="margin-bottom: 0;">
                            <strong>Delivery Analysis:</strong><br>
                            {forecast_narrative}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Overview Grid: Top Risks & Top Blockers Preview
        st.markdown("<br>", unsafe_allow_html=True)
        sum_col1, sum_col2 = st.columns([1, 1], gap="large")

        with sum_col1:
            st.markdown("### ⚠️ Top Project Risks")
            if not risks_list:
                st.info("No risks detected yet. Click '⚠️ Risks' in the sidebar to run risk analysis.")
            else:
                for r in risks_list[:3]:
                    sev = r.get("severity", "Medium")
                    sev_class = "sev-high" if sev.upper() == "HIGH" else ("sev-medium" if sev.upper() == "MEDIUM" else "sev-low")
                    st.markdown(f"""
                        <div class="risk-card" style="padding: 1rem 1.2rem; margin-bottom: 0.8rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                                <strong style="color: #F8FAFC; font-size: 0.95rem;">{r.get('title', 'Risk Item')}</strong>
                                <span class="{sev_class}">{sev.upper()}</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.4rem;">{r.get('description', '')}</div>
                            <div style="font-size: 0.8rem; color: #64748B;">Source: <code style="color: #A5B4FC;">{r.get('source', 'N/A')}</code></div>
                        </div>
                    """, unsafe_allow_html=True)
                if st.button("View All Risks →", key="dash_view_risks"):
                    st.session_state["nav_page"] = "⚠️ Risks"
                    st.rerun()

        with sum_col2:
            st.markdown("### 🛑 Critical Blockers & Action Items")
            if not blockers_list and not (blocker_data and blocker_data.get("action_items")):
                st.info("No open blockers recorded. Click '🛑 Blockers & Actions' in the sidebar.")
            else:
                for b in blockers_list[:2]:
                    st.markdown(f"""
                        <div class="risk-card" style="border-left: 3px solid #EF4444; padding: 1rem 1.2rem; margin-bottom: 0.8rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                                <strong style="color: #F8FAFC; font-size: 0.95rem;">{b.get('title', 'Blocker')}</strong>
                                <span class="sev-high">{b.get('category', 'Technical').upper()}</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #94A3B8;">{b.get('description', '')}</div>
                            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Owner: <strong style="color: #CBD5E1;">{b.get('owner', 'Unassigned')}</strong></div>
                        </div>
                    """, unsafe_allow_html=True)

                act_items = blocker_data.get("action_items", []) if blocker_data else []
                if act_items:
                    st.markdown("**Pending Action Items:**")
                    for a in act_items[:2]:
                        st.markdown(f"• **{a.get('action')}** — Owner: `{a.get('owner', 'Team')}` (Due: `{a.get('due_date', 'TBD')}`)")

                if st.button("View All Blockers & Actions →", key="dash_view_blockers"):
                    st.session_state["nav_page"] = "🛑 Blockers & Actions"
                    st.rerun()


# =============================================================================
# PAGE 2: 📁 DOCUMENTS (INGESTION & VECTOR KNOWLEDGE BASE)
# =============================================================================
elif selected_page == "📁 Documents":
    st.markdown("""
        <div class="page-header">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 2px; text-transform: uppercase;">DOCUMENT REPOSITORY</div>
                <div class="page-title">Project Artifacts &amp; Vector Index</div>
                <div class="page-subtitle">Upload project artifacts (PDF, DOCX, CSV, TXT, XLSX) to build persistent RAG embeddings in ChromaDB.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    up_col1, up_col2 = st.columns([1.2, 1], gap="large")

    with up_col1:
        st.markdown("### 📤 Upload Project Documents")
        st.caption("Supported: Project Proposals (PDF), SRS specifications (DOCX), Meeting Notes (TXT), Task lists (CSV), Defect trackers (XLSX).")

        uploaded_files = st.file_uploader(
            "Upload Artifacts",
            type=["pdf", "docx", "csv", "txt", "xlsx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        auto_analyze = st.checkbox("⚡ Run AI Agent Analysis immediately after indexing", value=True)

        index_btn = st.button("⚡ Index & Analyze Project", type="primary", use_container_width=True)

        if index_btn:
            if not uploaded_files:
                st.warning("Please select at least one document file to upload.")
            else:
                status_container = st.container()
                with status_container:
                    st.markdown("##### 🔄 Indexing Pipeline")
                    progress_bar = st.progress(0)

                    total_files = len(uploaded_files)
                    all_chunks = []
                    processed_registry = []

                    for idx, uploaded_file in enumerate(uploaded_files):
                        file_name = uploaded_file.name

                        if not is_allowed_file(file_name):
                            st.error(f"Unsupported file format: {file_name}")
                            continue

                        # Steps 1 & 2: Ingestion & Text Extraction
                        st.markdown(f"<div style='font-size: 0.9rem; color: #CBD5E1; padding: 2px 0;'>✓ <b>{file_name}</b> uploaded and parsed</div>", unsafe_allow_html=True)

                        try:
                            file_bytes = uploaded_file.read()
                            doc_info = processor.process_file(
                                file_input=file_bytes,
                                filename=file_name,
                                project_name=project_name
                            )
                        except Exception as e:
                            st.error(f"Extraction error for {file_name}: {e}")
                            continue

                        # Step 3: Text Chunking
                        file_chunks = chunker.chunk_text(doc_info["text"], doc_info)
                        all_chunks.extend(file_chunks)

                        file_ext = file_name.split(".")[-1].upper()
                        processed_registry.append({
                            "File Name": file_name,
                            "Type": file_ext,
                            "Status": "✓ Uploaded",
                            "Indexed": "✓ Indexed",
                            "Words": doc_info.get("word_count", 0),
                            "Chunks": len(file_chunks),
                        })

                        progress_bar.progress(int((idx + 1) / total_files * 60))

                    if all_chunks:
                        # Step 4: Embeddings
                        st.markdown("<div style='font-size: 0.9rem; color: #A5B4FC; padding: 2px 0;'>✓ Generating dense vector embeddings...</div>", unsafe_allow_html=True)
                        chunk_texts = [c["text"] for c in all_chunks]
                        embeddings = embedder.embed_texts(chunk_texts)

                        # Step 5: ChromaDB Storage
                        st.markdown(f"<div style='font-size: 0.9rem; color: #4ade80; padding: 2px 0;'>✓ Storing chunks in ChromaDB (<b>{project_name}</b>)...</div>", unsafe_allow_html=True)
                        added_count = vector_store.add_chunks(project_name, all_chunks, embeddings)
                        progress_bar.progress(100)

                        # Save metadata in session state
                        st.session_state[f"uploaded_files_{project_name}"] = processed_registry
                        st.session_state[f"last_indexed_{project_name}"] = datetime.now().strftime("%d %b %Y, %H:%M")

                        st.success(f"Successfully processed {len(processed_registry)} document(s) and stored {added_count} chunks in ChromaDB!")

                        # Run automated analysis if selected
                        if auto_analyze:
                            st.info("Running AI analysis agents on indexed documents...")
                            if run_all_project_agents():
                                st.success("AI Analysis Complete! Navigating to Project Dashboard...")
                                time.sleep(1.5)
                                st.session_state["nav_page"] = "📊 Dashboard"
                                st.rerun()

    with up_col2:
        st.markdown("### 📊 Knowledge Base Status")
        current_docs = st.session_state.get(f"uploaded_files_{project_name}", [])
        last_idx = st.session_state.get(f"last_indexed_{project_name}", "Not yet indexed")

        st.markdown(f"""
            <div class="glass-card">
                <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">INDEX HEALTH</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #38BDF8; margin: 4px 0;">
                    {len(current_docs)} Documents
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.8rem;">
                    {kb_count} Context Chunks
                </div>
                <div style="font-size: 0.88rem; color: #94A3B8;">
                    <strong>Status:</strong> <span class="{'status-badge-ontrack' if kb_count > 0 else 'status-badge-nodata'}">{'Ready' if kb_count > 0 else 'Empty'}</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-top: 6px;">
                    Last Indexed: <strong style="color: #CBD5E1;">{last_idx}</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Document Repository Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📑 Uploaded Documents Registry")
    current_docs = st.session_state.get(f"uploaded_files_{project_name}", [])
    if current_docs:
        df_docs = pd.DataFrame(current_docs)
        st.dataframe(df_docs, use_container_width=True, hide_index=True)
    else:
        if kb_count > 0:
            st.info(f"Knowledge base contains {kb_count} active chunks in ChromaDB. Upload more documents above to augment the knowledge base.")
        else:
            st.caption("No documents registered yet. Upload documents using the form above.")


# =============================================================================
# PAGE 3: 🎯 SCOPE & DELIVERABLES
# =============================================================================
elif selected_page == "🎯 Scope & Deliverables":
    st.markdown("""
        <div class="page-header">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 2px; text-transform: uppercase;">MILESTONE 2 AGENT</div>
                <div class="page-title">Project Scope &amp; Deliverables</div>
                <div class="page-subtitle">Evidence-grounded project goals, deliverables, milestones, timeline, and responsibilities.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    scope_btn = st.button("⚡ Extract / Refresh Scope & Deliverables", type="primary")
    if scope_btn:
        if kb_count == 0:
            st.warning("Please upload and index project documents first.")
        else:
            with st.spinner("Extracting project scope, deliverables, and milestones..."):
                provider = resolve_agent_provider(scope_model)
                agent = ScopeExtractionAgent(
                    provider=provider,
                    retriever=retriever,
                    model_name=groq_risk_model if llm_engine.startswith("Groq") else scope_model
                )
                res = agent.run(project_id=project_name, top_k=top_k)
                if store_agent_result(res, f"scope_data_{project_name}", "Scope & Deliverables extraction"):
                    st.success("Scope & Deliverables extracted successfully.")
                    st.rerun()

    scope_data = st.session_state.get(f"scope_data_{project_name}")
    if not scope_data:
        st.info("No scope data extracted yet. Click **[ ⚡ Extract / Refresh Scope & Deliverables ]** above.")
    else:
        # Project Goals Paragraph
        st.markdown("### 🎯 Project Goals & Objectives")
        goals = scope_data.get("project_goals", [])
        if goals:
            goal_items = []
            for g in goals:
                gt = g.get("goal", "").strip()
                gd = g.get("description", "").strip() if g.get("description") else ""
                if gt and gt != "Not specified in the available project documents.":
                    goal_items.append(f"{gt} {gd}" if (gd and gd != gt) else gt)
            if goal_items:
                st.markdown(f"""
                    <p style="font-size: 1.02rem; line-height: 1.7; color: #334155;">
                        {' '.join(goal_items)}
                    </p>
                """, unsafe_allow_html=True)
            else:
                st.caption("Not specified in the available project documents.")
        else:
            st.caption("Not specified in the available project documents.")

        # Deliverables Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📦 Deliverables")
        deliverables = scope_data.get("deliverables", [])
        if deliverables:
            deliv_rows = []
            for idx, d in enumerate(deliverables, 1):
                d_id = f"D-{idx:02d}"
                deliv_rows.append({
                    "ID": d_id,
                    "Deliverable": d.get("name", "Not specified"),
                    "Description": d.get("description", "Not specified in the available project documents."),
                    "Status": d.get("status", "Identified")
                })
            st.dataframe(pd.DataFrame(deliv_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No deliverables specified in the available project documents.")

        # Milestones Timeline
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📍 Milestones & Target Deadlines")
        milestones = scope_data.get("milestones", [])
        if milestones:
            m_rows = []
            for idx, m in enumerate(milestones, 1):
                m_name = m.get("name") or f"M{idx}"
                m_rows.append({
                    "Milestone": m_name,
                    "Description": m.get("description", "Not specified"),
                    "Deadline": m.get("target_date", "Not specified"),
                    "Status": m.get("status", "Pending")
                })
            st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No milestones specified in the available project documents.")

        # Timeline & Responsibilities Columns
        st.markdown("<br>", unsafe_allow_html=True)
        t_col1, t_col2 = st.columns([1, 1], gap="large")

        with t_col1:
            st.markdown("### 📅 Schedule & Timeline")
            timeline = scope_data.get("timeline", [])
            if timeline:
                timeline_rows = [
                    {
                        "Schedule Item": item.get("label", "Timeline"),
                        "Date / Duration": item.get("value", "Not specified"),
                        "Source": item.get("source", "Not specified"),
                    }
                    for item in timeline
                ]
                st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No timeline dates specified in the available project documents.")

        with t_col2:
            st.markdown("### 👥 Responsibilities")
            responsibilities = scope_data.get("responsibilities", [])
            if responsibilities:
                resp_rows = []
                for r in responsibilities:
                    resp_rows.append({
                        "Person / Role": r.get("person", "Not specified"),
                        "Responsibility": r.get("responsibility", "Not specified")
                    })
                st.dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No responsibilities specified in the available project documents.")

        # Sources Citation
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📚 Grounded Source Evidence")
        src_set = set()
        for g in scope_data.get("project_goals", []):
            if g.get("source") and g["source"] != "Not specified in the available project documents.":
                src_set.add(g["source"] + (f" (Page {g['page']})" if g.get("page") else ""))
        for d in scope_data.get("deliverables", []):
            if d.get("source") and d["source"] != "Not specified in the available project documents.":
                src_set.add(d["source"] + (f" (Page {d['page']})" if d.get("page") else ""))
        if src_set:
            for s in sorted(src_set):
                st.markdown(f"<span class='source-tag'>📄 {s}</span>", unsafe_allow_html=True)
        else:
            st.caption("No explicit sources recorded.")


# =============================================================================
# PAGE 4: ⚠️ RISKS (ANALYSIS & DELIVERY FORECAST)
# =============================================================================
elif selected_page == "⚠️ Risks":
    st.markdown("""
        <div class="page-header">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 2px; text-transform: uppercase;">MILESTONE 2 AGENT</div>
                <div class="page-title">Project Risk Analysis &amp; Delivery Forecast</div>
                <div class="page-subtitle">Evidence-based risk detection, probability/impact assessment, and AI-predicted project delivery dates.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    risk_btn = st.button("⚠️ Detect Risks & Run Delivery Forecast", type="primary")
    if risk_btn:
        if kb_count == 0:
            st.warning("Please upload and index project documents first.")
        else:
            with st.spinner("Analyzing risks and calculating delivery forecast..."):
                provider = resolve_agent_provider(groq_risk_model)
                agent = RiskDetectionAgent(provider=provider, retriever=retriever, model_name=groq_risk_model)
                res = agent.run(project_id=project_name, top_k=top_k)
                st.session_state[f"risk_data_{project_name}"] = res.get("data", {})
                st.success("Risk analysis complete.")
                st.rerun()

    risk_data = st.session_state.get(f"risk_data_{project_name}")
    if not risk_data:
        st.info("No risk analysis data yet. Click **[ ⚠️ Detect Risks & Run Delivery Forecast ]** above.")
    else:
        risks_list = risk_data.get("risks", [])
        high_cnt = len([r for r in risks_list if r.get("severity", "").upper() == "HIGH"])
        med_cnt  = len([r for r in risks_list if r.get("severity", "").upper() == "MEDIUM"])
        low_cnt  = len([r for r in risks_list if r.get("severity", "").upper() == "LOW"])
        del_status = risk_data.get("delivery_status", "Insufficient Data")

        # Top Risk Summary Metric Row
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            st.markdown(f"""
                <div class="kpi-card" style="border-left: 3px solid #EF4444;">
                    <div class="kpi-label">High Severity Risks</div>
                    <div class="kpi-value" style="color: #f87171;">{high_cnt}</div>
                    <div class="kpi-sub">Immediate mitigation needed</div>
                </div>
            """, unsafe_allow_html=True)
        with r_col2:
            st.markdown(f"""
                <div class="kpi-card" style="border-left: 3px solid #F59E0B;">
                    <div class="kpi-label">Medium Severity Risks</div>
                    <div class="kpi-value" style="color: #fbbf24;">{med_cnt}</div>
                    <div class="kpi-sub">Monitor actively</div>
                </div>
            """, unsafe_allow_html=True)
        with r_col3:
            st.markdown(f"""
                <div class="kpi-card" style="border-left: 3px solid #3B82F6;">
                    <div class="kpi-label">Low Severity Risks</div>
                    <div class="kpi-value" style="color: #60a5fa;">{low_cnt}</div>
                    <div class="kpi-sub">Standard tracking</div>
                </div>
            """, unsafe_allow_html=True)
        with r_col4:
            del_badge = "status-badge-ontrack" if del_status == "On Track" else ("status-badge-atrisk" if del_status == "At Risk" else "status-badge-delayed")
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Delivery Status</div>
                    <div class="kpi-value" style="font-size: 1.4rem; padding-top: 6px;">
                        <span class="{del_badge}">{del_status.upper()}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Delivery Forecast Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📅 Delivery Forecast")
        forecast_obj = risk_data.get("delivery_forecast", {})
        planned_date = forecast_obj.get("planned_completion", "Not specified in documents")
        est_date = forecast_obj.get("estimated_completion", "Pending analysis")
        forecast_narrative = forecast_obj.get("forecast_analysis") or forecast_obj.get("reason") or "No forward-looking analysis available."

        st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; gap: 30px; margin-bottom: 1rem;">
                    <div>
                        <span style="font-size: 0.8rem; color: #64748B; font-weight: 700;">PLANNED COMPLETION</span>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #E2E8F0;">{planned_date}</div>
                    </div>
                    <div>
                        <span style="font-size: 0.8rem; color: #64748B; font-weight: 700;">FORECAST COMPLETION</span>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #FCD34D;">{est_date}</div>
                    </div>
                </div>
                <div class="forecast-box" style="margin-bottom: 0;">
                    <strong>Forecast Analysis &amp; Drivers:</strong><br>
                    {forecast_narrative}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Individual Risk Cards
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### ⚠️ Identified Project Risks ({len(risks_list)})")
        if not risks_list:
            st.caption("No specific risks identified in the provided documents.")
        else:
            risk_rows = [
                {
                    "Category": item.get("category", "Not specified"),
                    "Risk": item.get("description", "Not specified"),
                    "Severity": item.get("severity", "Not specified"),
                    "Evidence": item.get("evidence", "Not specified"),
                    "Source": item.get("source", "Not specified"),
                    "Recommended Action": item.get("recommended_action", "Not specified"),
                }
                for item in risks_list
            ]
            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 5: 🛑 BLOCKERS & ACTIONS
# =============================================================================
elif selected_page == "🛑 Blockers & Actions":
    st.markdown("""
        <div class="page-header">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 2px; text-transform: uppercase;">MILESTONE 2 AGENT</div>
                <div class="page-title">Blockers &amp; Action Items</div>
                <div class="page-subtitle">Categorized project blockers (Technical, Dependency, Approval, Resource, Testing) and assigned action items.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    blocker_btn = st.button("🛑 Detect Blockers & Action Items", type="primary")
    if blocker_btn:
        if kb_count == 0:
            st.warning("Please upload and index project documents first.")
        else:
            with st.spinner("Extracting open blockers and action items..."):
                provider = resolve_agent_provider(blocker_model)
                agent = BlockerActionAgent(
                    provider=provider,
                    retriever=retriever,
                    model_name=groq_risk_model if llm_engine.startswith("Groq") else blocker_model
                )
                res = agent.run(project_id=project_name, top_k=top_k)
                if store_agent_result(res, f"blocker_data_{project_name}", "Blocker & Action extraction"):
                    st.success("Blocker & action item identification complete.")
                    st.rerun()

    blocker_data = st.session_state.get(f"blocker_data_{project_name}")
    if not blocker_data:
        st.info("No blocker data extracted yet. Click **[ 🛑 Detect Blockers & Action Items ]** above.")
    else:
        blockers_list = blocker_data.get("blockers", [])
        action_items = blocker_data.get("action_items", [])

        # Open Blockers Section
        st.markdown(f"### 🛑 Open Project Blockers ({len(blockers_list)})")
        if not blockers_list:
            st.caption("No open blockers recorded in the provided documents.")
        else:
            for b in blockers_list:
                cat = b.get("category", "Technical")
                st.markdown(f"""
                    <div class="risk-card" style="border-left: 4px solid #EF4444;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <strong style="font-size: 1.05rem; color: #F8FAFC;">{b.get('title', 'Blocker')}</strong>
                            <span class="sev-high">{cat.upper()}</span>
                        </div>
                        <p style="font-size: 0.95rem; color: #CBD5E1; margin-bottom: 0.6rem;">{b.get('description', '')}</p>
                        <div style="display: flex; gap: 20px; font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.4rem;">
                            <span>Owner: <strong style="color: #F8FAFC;">{b.get('owner', 'Unassigned')}</strong></span>
                            <span>Status: <strong style="color: #fbbf24;">{b.get('status', 'Open')}</strong></span>
                            <span>Due Date: <strong style="color: #CBD5E1;">{b.get('due_date', 'TBD')}</strong></span>
                        </div>
                        <div class="evidence-quote" style="margin-bottom: 0;">
                            <strong>Evidence:</strong> "{b.get('evidence', 'No quote recorded.')}" (Source: <code style="color: #A5B4FC;">{b.get('source', 'N/A')}</code>)
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Action Items Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 📋 Action Items ({len(action_items)})")
        if not action_items:
            st.caption("No action items recorded in the provided documents.")
        else:
            act_rows = []
            for a in action_items:
                act_rows.append({
                    "Action": a.get("action", "Not specified"),
                    "Owner": a.get("assignee") or a.get("owner", "Not specified"),
                    "Due Date": a.get("deadline") or a.get("due_date", "Not specified"),
                    "Priority": a.get("priority", "Not specified"),
                    "Status": a.get("status", "Open"),
                    "Related Blocker / Risk": a.get("related_blocker_risk", "General")
                })
            df_act = pd.DataFrame(act_rows)
            st.dataframe(df_act, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 6: 🩺 PROJECT HEALTH (DETERMINISTIC EVALUATION)
# =============================================================================
elif selected_page == "🩺 Project Health":
    st.markdown("""
        <div class="page-header">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 2px; text-transform: uppercase;">MILESTONE 3 AGENT</div>
                <div class="page-title">Project Health Evaluation</div>
                <div class="page-subtitle">Deterministic 0-100 scoring based on Scope Clarity, Timeline Risk, Blocker Status, and Delivery Risk.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    eval_btn = st.button("📊 Evaluate Project Health", type="primary")
    if eval_btn:
        if kb_count == 0:
            st.warning("Please upload and index project documents first.")
        else:
            with st.spinner("Calculating deterministic project health score..."):
                s_data = st.session_state.get(f"scope_data_{project_name}")
                r_data = st.session_state.get(f"risk_data_{project_name}")
                b_data = st.session_state.get(f"blocker_data_{project_name}")

                health_scorer = ProjectHealthScorer()
                res = health_scorer.evaluate_health(
                    scope_data=s_data,
                    risk_data=r_data,
                    blocker_data=b_data,
                    project_id=project_name,
                    retriever=retriever,
                    top_k=top_k
                )
                st.session_state[f"health_data_{project_name}"] = res.model_dump()
                st.success("Project health evaluated successfully.")
                st.rerun()

    health_data = st.session_state.get(f"health_data_{project_name}")
    if not health_data:
        st.info("Health evaluation not yet calculated. Click **[ 📊 Evaluate Project Health ]** above.")
    else:
        score_val = health_data.get("overall_score", 0.0)
        h_class = health_data.get("classification", "Moderate")
        confidence = health_data.get("confidence", "Medium")
        sufficiency = health_data.get("data_sufficiency", "")
        dimensions = health_data.get("dimensions", {})
        key_factors = health_data.get("key_factors", [])
        recommendations = health_data.get("recommendations", [])

        h_badge = "status-badge-ontrack" if h_class == "Healthy" else ("status-badge-atrisk" if h_class == "Moderate" else "status-badge-delayed")

        # Executive Health Card
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 2.2rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #64748B; letter-spacing: 2px;">OVERALL PROJECT HEALTH</div>
                <div style="font-size: 4.2rem; font-weight: 800; color: #38BDF8; margin: 6px 0; line-height: 1;">
                    {int(score_val)}<span style="font-size: 1.6rem; color: #64748B; font-weight: 500;"> / 100</span>
                </div>
                <div style="margin: 12px 0;">
                    <span class="{h_badge}" style="font-size: 1.1rem; padding: 6px 24px; text-transform: uppercase;">{h_class}</span>
                </div>
                <div style="font-size: 0.88rem; color: #94A3B8;">
                    Confidence: <strong style="color: #F8FAFC;">{confidence}</strong> • <em>{sufficiency}</em>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Dimension Breakdown Grid
        st.markdown("### 📊 Dimension-Wise Health Scores")
        dim_col1, dim_col2 = st.columns(2, gap="large")

        dim_list = [
            ("Scope Clarity", "scope_clarity", dim_col1),
            ("Timeline", "timeline_risk", dim_col2),
            ("Blockers", "blocker_status", dim_col1),
            ("Delivery Risk", "delivery_risk", dim_col2),
        ]

        for dim_title, _, col in dim_list:
            d_info = dimensions.get(dim_title, {})
            d_sc = d_info.get("score", 70.0)
            d_st = d_info.get("status", "Moderate")
            with col:
                st.markdown(f"""
                    <div class="glass-card" style="padding: 1.2rem 1.4rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <strong style="color: #F8FAFC; font-size: 1rem;">{dim_title} (25% Weight)</strong>
                            <span class="{'status-badge-ontrack' if d_st == 'Healthy' else ('status-badge-atrisk' if d_st == 'Moderate' else 'status-badge-delayed')}">{d_st}</span>
                        </div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #38BDF8; margin-bottom: 0.4rem;">{int(d_sc)}%</div>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(int(d_sc))

        # Strategic Factors & Recommendations
        st.markdown("<br>", unsafe_allow_html=True)
        f_col1, f_col2 = st.columns([1, 1], gap="large")

        with f_col1:
            st.markdown("### 🔍 Why is the project currently at risk?")
            if key_factors:
                for kf in key_factors:
                    st.markdown(f"• <span style='color: #CBD5E1; font-size: 0.95rem;'>{kf}</span>", unsafe_allow_html=True)
            else:
                st.caption("No adverse factors recorded.")

        with f_col2:
            st.markdown("### 💡 Strategic Next Steps")
            if recommendations:
                for r in recommendations:
                    st.markdown(f"💡 <span style='color: #CBD5E1; font-size: 0.95rem;'>{r}</span>", unsafe_allow_html=True)
            else:
                st.caption("No explicit recommendations recorded.")


# =============================================================================
# PAGE 7: 📄 DOCUMENTATION (AUTOMATED GENERATION)
# =============================================================================
elif selected_page == "📄 Documentation":
    st.markdown("""
        <div class="page-header">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 2px; text-transform: uppercase;">MILESTONE 3 AGENT</div>
                <div class="page-title">Automated Project Documentation</div>
                <div class="page-subtitle">Generate missing User Stories, formal Risk Registers, and actionable Task Lists directly from project evidence.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    doc_col1, doc_col2 = st.columns(2)
    with doc_col1:
        gen_stories_btn = st.button("📝 Generate User Stories", type="secondary", use_container_width=True)
        gen_actions_btn = st.button("📋 Generate Action Items", type="secondary", use_container_width=True)
    with doc_col2:
        gen_risks_btn   = st.button("⚠️ Generate Risk Register", type="secondary", use_container_width=True)
        gen_all_btn     = st.button("⚡ Generate All Documentation", type="primary", use_container_width=True)

    def get_doc_agent():
        provider = resolve_agent_provider(docgen_model)
        eff_model = groq_risk_model if llm_engine.startswith("Groq") else docgen_model
        return DocumentationAgent(provider=provider, retriever=retriever, model_name=eff_model)

    if gen_stories_btn:
        if kb_count == 0:
            st.warning("Please upload and index documents first.")
        else:
            with st.spinner("Generating grounded User Stories from scope..."):
                d_agent = get_doc_agent()
                s_info = st.session_state.get(f"scope_data_{project_name}")
                res = d_agent.generate_user_stories(project_id=project_name, scope_data=s_info, top_k=top_k)
                st.session_state[f"doc_user_stories_{project_name}"] = res.get("data", {}).get("user_stories", [])
                st.success("User Stories generated.")

    if gen_risks_btn:
        if kb_count == 0:
            st.warning("Please upload and index documents first.")
        else:
            with st.spinner("Generating formal Risk Register..."):
                d_agent = get_doc_agent()
                r_info = st.session_state.get(f"risk_data_{project_name}")
                res = d_agent.generate_risk_register(project_id=project_name, risk_data=r_info, top_k=top_k)
                st.session_state[f"doc_risk_register_{project_name}"] = res.get("data", {}).get("risk_register", [])
                st.success("Risk Register generated.")

    if gen_actions_btn:
        if kb_count == 0:
            st.warning("Please upload and index documents first.")
        else:
            with st.spinner("Generating Action Items..."):
                d_agent = get_doc_agent()
                b_info = st.session_state.get(f"blocker_data_{project_name}")
                res = d_agent.generate_action_items(project_id=project_name, blocker_data=b_info, top_k=top_k)
                st.session_state[f"doc_action_items_{project_name}"] = res.get("data", {}).get("action_items", [])
                st.success("Action Items generated.")

    if gen_all_btn:
        if kb_count == 0:
            st.warning("Please upload and index documents first.")
        else:
            with st.spinner("Generating entire project documentation suite..."):
                d_agent = get_doc_agent()
                s_info = st.session_state.get(f"scope_data_{project_name}")
                r_info = st.session_state.get(f"risk_data_{project_name}")
                b_info = st.session_state.get(f"blocker_data_{project_name}")
                all_res = d_agent.generate_all(
                    project_id=project_name,
                    scope_data=s_info,
                    risk_data=r_info,
                    blocker_data=b_info,
                    top_k=top_k
                )
                data_dict = all_res.get("data", {})
                st.session_state[f"doc_user_stories_{project_name}"] = data_dict.get("user_stories", [])
                st.session_state[f"doc_risk_register_{project_name}"] = data_dict.get("risk_register", [])
                st.session_state[f"doc_action_items_{project_name}"] = data_dict.get("action_items", [])
                st.success("All documentation generated successfully.")

    st.markdown("<br>", unsafe_allow_html=True)
    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["📝 User Stories", "⚠️ Risk Register", "📋 Action Items"])

    user_stories = st.session_state.get(f"doc_user_stories_{project_name}", [])
    risk_register = st.session_state.get(f"doc_risk_register_{project_name}", [])
    action_items_doc = st.session_state.get(f"doc_action_items_{project_name}", [])

    with doc_tab1:
        if not user_stories:
            st.info("Click **[ 📝 Generate User Stories ]** above to create structured user stories.")
        else:
            us_rows = []
            for s in user_stories:
                ac_preview = " | ".join(s.get("acceptance_criteria", [])) if s.get("acceptance_criteria") else "Not specified"
                us_rows.append({
                    "ID": s.get("story_id", "US"),
                    "User Story": s.get("user_story", "Not specified"),
                    "Priority": s.get("priority", "Must Have"),
                    "Acceptance Criteria": ac_preview
                })
            df_us = pd.DataFrame(us_rows)
            st.dataframe(df_us, use_container_width=True, hide_index=True)

            csv_us = df_us.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download User Stories (CSV)", csv_us, f"{project_name}_user_stories.csv", "text/csv")

    with doc_tab2:
        if not risk_register:
            st.info("Click **[ ⚠️ Generate Risk Register ]** above to synthesize the risk register.")
        else:
            rr_rows = []
            for r in risk_register:
                rr_rows.append({
                    "ID": r.get("risk_id", "RR"),
                    "Risk": r.get("risk_description", r.get("risk", "Not specified")),
                    "Category": r.get("category", "Not specified"),
                    "Severity": r.get("severity", "Not specified"),
                    "Mitigation": r.get("mitigation", "Not specified"),
                    "Status": r.get("status", "Open"),
                    "Owner": r.get("owner", "Not specified")
                })
            df_rr = pd.DataFrame(rr_rows)
            st.dataframe(df_rr, use_container_width=True, hide_index=True)

            csv_rr = df_rr.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Risk Register (CSV)", csv_rr, f"{project_name}_risk_register.csv", "text/csv")

    with doc_tab3:
        if not action_items_doc:
            st.info("Click **[ 📋 Generate Action Items ]** above to create structured action items.")
        else:
            act_rows = []
            for a in action_items_doc:
                act_rows.append({
                    "ID": a.get("action_id", "A"),
                    "Action": a.get("action", "Not specified"),
                    "Assignee": a.get("assignee", "Unassigned"),
                    "Deadline": a.get("deadline", "TBD"),
                    "Status": a.get("status", "Pending")
                })
            df_act = pd.DataFrame(act_rows)
            st.dataframe(df_act, use_container_width=True, hide_index=True)

            csv_act = df_act.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Action Items (CSV)", csv_act, f"{project_name}_action_items.csv", "text/csv")


# =============================================================================
# PAGE 8: 💬 AI ASSISTANT (CONVERSATIONAL INTELLIGENCE)
# =============================================================================
elif selected_page == "💬 AI Assistant":
    st.markdown("""
        <div class="page-header">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 2px; text-transform: uppercase;">RAG-GROUNDED AGENT</div>
                <div class="page-title">Conversational Project Assistant</div>
                <div class="page-subtitle">Ask questions about deliverables, risks, deadlines, and blockers — answers are strictly grounded in your project documents.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Suggested Prompts Row
    st.markdown("##### 💡 Suggested Questions")
    sug_c1, sug_c2, sug_c3 = st.columns(3)
    sug1 = sug_c1.button("🎯 Are we on track?", use_container_width=True)
    sug2 = sug_c2.button("⚠️ What are our biggest risks?", use_container_width=True)
    sug3 = sug_c3.button("🛑 What blockers are unresolved?", use_container_width=True)
    sug4 = sug_c1.button("📦 What deliverables are pending?", use_container_width=True)
    sug5 = sug_c2.button("👥 Who is responsible for pending actions?", use_container_width=True)
    sug6 = sug_c3.button("🩺 Why is the project health score low?", use_container_width=True)

    chat_history_key = f"chat_history_{project_name}"
    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []

    # Selected quick prompt
    active_prompt = None
    if sug1: active_prompt = "Are we on track?"
    elif sug2: active_prompt = "What are our biggest risks?"
    elif sug3: active_prompt = "What blockers are currently unresolved?"
    elif sug4: active_prompt = "What deliverables are pending?"
    elif sug5: active_prompt = "Who is responsible for the pending actions?"
    elif sug6: active_prompt = "Why is the project health score low?"

    # Clear Chat Button
    c_btn_col, _ = st.columns([1, 4])
    with c_btn_col:
        if st.button("🗑️ Clear Chat History"):
            st.session_state[chat_history_key] = []
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Display Existing Messages
    for msg in st.session_state[chat_history_key]:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            sources = msg.get("sources", [])
            if sources and role == "assistant":
                badges = "".join([f"<span class='source-tag'>📄 {s}</span>" for s in sources])
                st.markdown(f"<div style='margin-top: 0.5rem;'>{badges}</div>", unsafe_allow_html=True)

    # Chat Input
    user_input = st.chat_input("Ask about your project (e.g. Are we on track? What tasks are delayed?)...")
    prompt_to_execute = active_prompt or user_input

    if prompt_to_execute:
        if kb_count == 0:
            st.warning("Please upload and index project documents first so the assistant can search project knowledge.")
        else:
            # Append User Message
            st.session_state[chat_history_key].append({"role": "user", "content": prompt_to_execute})
            with st.chat_message("user"):
                st.markdown(prompt_to_execute)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing project artifacts and intelligence..."):
                    provider = resolve_agent_provider(chat_model)
                    assistant = ConversationalProjectAssistant(
                        provider=provider,
                        retriever=retriever,
                        model_name=groq_risk_model if llm_engine.startswith("Groq") else chat_model,
                    )
                    assistant.set_history([
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state[chat_history_key][:-1]
                    ])

                    scope_info = st.session_state.get(f"scope_data_{project_name}")
                    risk_info = st.session_state.get(f"risk_data_{project_name}")
                    blocker_info = st.session_state.get(f"blocker_data_{project_name}")
                    health_info_data = st.session_state.get(f"health_data_{project_name}")

                    resp = assistant.ask(
                        question=prompt_to_execute,
                        project_id=project_name,
                        scope_data=scope_info,
                        risk_data=risk_info,
                        blocker_data=blocker_info,
                        health_data=health_info_data,
                        top_k=top_k
                    )

                    st.markdown(resp.answer)
                    if resp.sources:
                        badges = "".join([f"<span class='source-tag'>📄 {s}</span>" for s in resp.sources])
                        st.markdown(f"<div style='margin-top: 0.5rem;'>{badges}</div>", unsafe_allow_html=True)

                    st.session_state[chat_history_key].append({
                        "role": "assistant",
                        "content": resp.answer,
                        "sources": resp.sources
                    })

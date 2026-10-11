"""
AI Project Intelligence & Risk Advisor — Redesigned Modern Product UI

Architecture & Flow:
Upload Documents → Index Chunks in ChromaDB → Run AI Intelligence Agents → Dashboard & Deep Dives → AI Chat Assistant
"""

import atexit
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

# Modern UI Styles and Plotly Visualizations
from styles import inject_css
from ui_charts import (
    create_health_gauge,
    create_risk_priority_matrix,
    create_blocker_urgency_matrix,
    create_scope_timeline,
    COLOR_CORAL,
    COLOR_AMBER,
    COLOR_MINT,
    COLOR_VIOLET,
    COLOR_GREY
)

from utils.helpers import is_allowed_file
from ingestion.document_processor import DocumentProcessor
from rag.chunker import TextChunker
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager, cleanup_chroma_db
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

# Clean up Chroma DB when app process exits or stops
atexit.register(cleanup_chroma_db)

# ── Page Configuration ──
st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject shared design system (palette, Inter typography, airy cards)
inject_css()

# ── Safe DataFrame Renderer (Handles OS Application Control / PyArrow Restrictions) ──
_CAN_USE_PYARROW = False
try:
    import pyarrow  # noqa: F401
    _CAN_USE_PYARROW = True
except (ImportError, Exception):
    _CAN_USE_PYARROW = False


def render_dataframe(df: pd.DataFrame, use_container_width: bool = True, hide_index: bool = True):
    """Safely render a pandas DataFrame without crashing if PyArrow is blocked by OS policy."""
    if _CAN_USE_PYARROW:
        try:
            st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)
            return
        except Exception:
            pass
    # Styled HTML table fallback when pyarrow is blocked by Windows Application Control
    html = df.to_html(classes="modern-data-table", index=not hide_index, escape=False)
    st.markdown(f'<div class="modern-table-container">{html}</div>', unsafe_allow_html=True)


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

# ── Credentials & Model Settings ──
_gemini_key     = os.getenv("GEMINI_API_KEY", "")
_groq_key       = os.getenv("GROQ_API_KEY", "")
scope_model     = os.getenv("GEMINI_SCOPE_MODEL", "gemini-1.5-flash")
blocker_model   = os.getenv("GEMINI_BLOCKER_MODEL", "gemini-1.5-flash")
docgen_model    = os.getenv("GEMINI_DOCGEN_MODEL", "gemini-1.5-flash")
chat_model      = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
groq_risk_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-20b")

_gemini_valid   = bool(_gemini_key) and not _gemini_key.startswith("AQ.") and _gemini_key != "your_gemini_api_key_here"
_groq_valid     = bool(_groq_key) and _groq_key != "your_groq_api_key_here"


def resolve_agent_provider(gemini_model_id: str):
    """Resolve provider without exposing secrets in the UI."""
    llm_engine = st.session_state.get("llm_engine", "Groq")
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


def run_all_project_agents(proj_name: str, top_k_val: int):
    """Execute Scope, Risk, Blocker, and Health evaluation in sequence."""
    collection = vector_store.get_or_create_collection(proj_name)
    if collection.count() == 0:
        st.warning("No documents indexed in this workspace yet. Please upload and index documents first.")
        return False

    with st.spinner("⚡ AI Agents are analyzing project documents..."):
        # Step 1: Scope & Deliverables
        scope_provider = resolve_agent_provider(scope_model)
        scope_agent = ScopeExtractionAgent(
            provider=scope_provider,
            retriever=retriever,
            model_name=groq_risk_model if st.session_state.get("llm_engine", "Groq").startswith("Groq") else scope_model
        )
        s_res = scope_agent.run(project_id=proj_name, top_k=top_k_val)
        scope_ok = store_agent_result(s_res, f"scope_data_{proj_name}", "Scope & Deliverables extraction")
        scope_data = st.session_state.get(f"scope_data_{proj_name}", {})

        # Step 2: Risks & Delivery Forecasting
        risk_provider = resolve_agent_provider(groq_risk_model)
        risk_agent = RiskDetectionAgent(
            provider=risk_provider,
            retriever=retriever,
            model_name=groq_risk_model
        )
        r_res = risk_agent.run(project_id=proj_name, top_k=top_k_val)
        risk_ok = store_agent_result(r_res, f"risk_data_{proj_name}", "Risk analysis")
        risk_data = st.session_state.get(f"risk_data_{proj_name}", {})

        # Step 3: Blockers & Action Items
        blocker_provider = resolve_agent_provider(blocker_model)
        blocker_agent = BlockerActionAgent(
            provider=blocker_provider,
            retriever=retriever,
            model_name=groq_risk_model if st.session_state.get("llm_engine", "Groq").startswith("Groq") else blocker_model
        )
        b_res = blocker_agent.run(project_id=proj_name, top_k=top_k_val)
        blocker_ok = store_agent_result(b_res, f"blocker_data_{proj_name}", "Blocker & Action extraction")
        blocker_data = st.session_state.get(f"blocker_data_{proj_name}", {})

        # Step 4: Health Scoring
        health_scorer = ProjectHealthScorer()
        h_res = health_scorer.evaluate_health(
            scope_data=scope_data,
            risk_data=risk_data,
            blocker_data=blocker_data,
            project_id=proj_name,
            retriever=retriever,
            top_k=top_k_val
        )
        st.session_state[f"health_data_{proj_name}"] = h_res.model_dump()

    return scope_ok and risk_ok and blocker_ok


# ── Top Brand Header & Workspace Selector ──
top_col1, top_col2 = st.columns([2.5, 1.2], gap="medium")

with top_col1:
    st.markdown("""
        <div class="brand-container">
            <div class="brand-logo-text">
                ⚡ AI Risk &amp; Project Advisor
                <span class="brand-badge">Intelligence Platform</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with top_col2:
    ws_col, eng_col = st.columns(2)
    with ws_col:
        project_name = st.text_input(
            "Workspace",
            value=st.session_state.get("project_name", "Main Workspace"),
            placeholder="Workspace Name",
            label_visibility="collapsed"
        )
        if not project_name or not project_name.strip():
            project_name = "Main Workspace"
        st.session_state["project_name"] = project_name

    with eng_col:
        llm_engine = st.selectbox(
            "AI Engine",
            options=["Groq (Fast)", "Google Gemini"],
            index=0,
            label_visibility="collapsed"
        )
        st.session_state["llm_engine"] = llm_engine

# Knowledge Base & Chunks Count
collection = vector_store.get_or_create_collection(project_name)
kb_count = collection.count()
top_k = retriever.automatic_top_k(kb_count)

# Retrieve cached session state for active workspace
scope_data = st.session_state.get(f"scope_data_{project_name}")
risk_data = st.session_state.get(f"risk_data_{project_name}")
blocker_data = st.session_state.get(f"blocker_data_{project_name}")
health_data = st.session_state.get(f"health_data_{project_name}")
doc_registry = st.session_state.get(f"uploaded_files_{project_name}", [])
last_indexed = st.session_state.get(f"last_indexed_{project_name}", "Not yet indexed")


# ── Top Navigation Tabs (Pill Buttons) ──
nav_tabs = st.tabs([
    "📊 Dashboard",
    "📁 Manage Documents",
    "🎯 Scope · Blockers · Risks",
    "📄 Document Generation",
    "🩺 Health Score"
])


# =============================================================================
# TAB 1: 📊 DASHBOARD (EXECUTIVE OVERVIEW)
# =============================================================================
with nav_tabs[0]:
    if kb_count == 0:
        # Centered Illustration Empty State
        st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">📁</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #1F2340; margin-bottom: 0.5rem;">
                    Upload documents to get started
                </div>
                <div style="font-size: 0.95rem; color: #7A7F9A; line-height: 1.6; margin-bottom: 1.5rem;">
                    Upload your project artifacts (PDF proposals, DOCX specs, CSV tasks, TXT notes, or XLSX trackers) in <b>Manage Documents</b> to generate AI project intelligence.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Row 1: Health Gauge + 4 KPI Cards
        score_val = health_data.get("overall_score", 0.0) if health_data else 0.0
        health_class = health_data.get("classification", "Pending") if health_data else "Pending"
        risks_list = risk_data.get("risks", []) if risk_data else []
        crit_risks = len([r for r in risks_list if str(r.get("severity", "")).upper() in ["HIGH", "CRITICAL"]])
        blockers_list = blocker_data.get("blockers", []) if blocker_data else []
        open_blockers = len([b for b in blockers_list if str(b.get("status", "Open")).lower() != "resolved"])
        act_items = blocker_data.get("action_items", []) if blocker_data else []
        open_actions = len(act_items)
        num_docs = len(doc_registry) if doc_registry else 1

        r1_gauge, r1_kpi1, r1_kpi2, r1_kpi3, r1_kpi4 = st.columns([1.3, 1, 1, 1, 1], gap="medium")

        with r1_gauge:
            with st.container(border=True):
                st.plotly_chart(create_health_gauge(score_val, health_class), use_container_width=True, config={"displayModeBar": False}, key="dash_health_gauge")
                badge_style = "badge-mint" if score_val >= 80 else ("badge-amber" if score_val >= 60 else "badge-coral")
                st.markdown(f"<div style='text-align: center; margin-top: -12px;'><span class='{badge_style}'>{health_class.upper()} STATUS</span></div>", unsafe_allow_html=True)

        with r1_kpi1:
            st.markdown(f"""
                <div class="kpi-wrapper">
                    <div class="kpi-icon-circle kpi-icon-violet">📄</div>
                    <div class="kpi-info">
                        <div class="kpi-title">Artifacts</div>
                        <div class="kpi-val">{num_docs}</div>
                        <div class="kpi-subtext">Processed &amp; Ready</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r1_kpi2:
            st.markdown(f"""
                <div class="kpi-wrapper">
                    <div class="kpi-icon-circle kpi-icon-coral">⚠️</div>
                    <div class="kpi-info">
                        <div class="kpi-title">Risks</div>
                        <div class="kpi-val">{len(risks_list)}</div>
                        <div class="kpi-subtext"><span style="color:#EF6B73;font-weight:700;">{crit_risks} Critical</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r1_kpi3:
            st.markdown(f"""
                <div class="kpi-wrapper">
                    <div class="kpi-icon-circle kpi-icon-amber">🛑</div>
                    <div class="kpi-info">
                        <div class="kpi-title">Blockers</div>
                        <div class="kpi-val">{len(blockers_list)}</div>
                        <div class="kpi-subtext"><span style="color:#F5B547;font-weight:700;">{open_blockers} Open</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r1_kpi4:
            st.markdown(f"""
                <div class="kpi-wrapper">
                    <div class="kpi-icon-circle kpi-icon-mint">📋</div>
                    <div class="kpi-info">
                        <div class="kpi-title">Actions</div>
                        <div class="kpi-val">{open_actions}</div>
                        <div class="kpi-subtext">Assigned tasks</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Row 2: Risk Priority Matrix | Blocker Urgency Matrix
        r2_col1, r2_col2 = st.columns([1, 1], gap="large")

        with r2_col1:
            with st.container(border=True):
                head_col1, head_col2 = st.columns([1.5, 1])
                with head_col1:
                    st.markdown("##### ⚠️ Risk Priority Matrix")
                    st.caption(f"{len(risks_list)} risks plotted across probability & impact")
                with head_col2:
                    risk_filter = st.selectbox("Filter Level", ["All", "High", "Medium", "Low"], key="dash_rf", label_visibility="collapsed")
                st.plotly_chart(create_risk_priority_matrix(risks_list, filter_level=risk_filter), use_container_width=True, config={"displayModeBar": False}, key="dash_risk_matrix")

        with r2_col2:
            with st.container(border=True):
                b_head1, b_head2 = st.columns([1.5, 1])
                with b_head1:
                    st.markdown("##### 🛑 Blocker Urgency Matrix")
                    st.caption(f"{len(blockers_list)} blockers plotted across aging & severity")
                with b_head2:
                    block_filter = st.selectbox("Filter Category", ["All", "Technical", "Dependency", "Approval", "Resource", "Testing"], key="dash_bf", label_visibility="collapsed")
                st.plotly_chart(create_blocker_urgency_matrix(blockers_list, filter_cat=block_filter), use_container_width=True, config={"displayModeBar": False}, key="dash_blocker_matrix")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Row 3: Scope Timeline | Health Breakdown
        r3_col1, r3_col2 = st.columns([1.4, 1], gap="large")

        with r3_col1:
            with st.container(border=True):
                st.markdown("##### 📅 Scope & Deliverables Timeline")
                st.caption("Milestone progression and target deadlines grounded in document evidence")
                delivs = scope_data.get("deliverables", []) if scope_data else []
                milestones = scope_data.get("milestones", []) if scope_data else []
                timeline_items = delivs if delivs else milestones
                st.plotly_chart(create_scope_timeline(timeline_items), use_container_width=True, config={"displayModeBar": False}, key="dash_scope_timeline")

        with r3_col2:
            with st.container(border=True):
                st.markdown("##### 🩺 Health Dimension Breakdown")
                st.caption("Evaluation across core project dimensions")
                if health_data:
                    dims = health_data.get("dimensions", {})
                    dim_map = [
                        ("Scope Clarity", "scope_clarity"),
                        ("Timeline Risk", "timeline_risk"),
                        ("Blocker Status", "blocker_status"),
                        ("Delivery Risk", "delivery_risk")
                    ]
                    for label, key in dim_map:
                        d_info = dims.get(label, {})
                        d_score = d_info.get("score", 70.0)
                        d_status = d_info.get("status", "Moderate")
                        st.write(f"**{label}** — {int(d_score)}% ({d_status})")
                        st.progress(int(d_score) / 100)
                else:
                    st.caption("Run project analysis to view health breakdown.")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Row 4: Generated Documents Cards with "Open →" Navigation
        st.markdown("##### 📄 Project Documentation Suite")
        r4_c1, r4_c2, r4_c3 = st.columns(3, gap="medium")

        user_stories = st.session_state.get(f"doc_user_stories_{project_name}", [])
        risk_reg = st.session_state.get(f"doc_risk_register_{project_name}", [])
        actions_doc = st.session_state.get(f"doc_action_items_{project_name}", [])

        with r4_c1:
            with st.container(border=True):
                st.markdown(f"**📝 User Stories**")
                st.markdown(f"<div style='font-size: 1.4rem; font-weight: 800; color: #7C5CFC;'>{len(user_stories)} Generated</div>", unsafe_allow_html=True)
                st.caption("Agile user stories with acceptance criteria")

        with r4_c2:
            with st.container(border=True):
                st.markdown(f"**⚠️ Risk Register**")
                st.markdown(f"<div style='font-size: 1.4rem; font-weight: 800; color: #EF6B73;'>{len(risk_reg)} Identified</div>", unsafe_allow_html=True)
                st.caption("Categorized risk log with mitigation plans")

        with r4_c3:
            with st.container(border=True):
                st.markdown(f"**📋 Action Items**")
                st.markdown(f"<div style='font-size: 1.4rem; font-weight: 800; color: #34C38F;'>{len(actions_doc)} Tracked</div>", unsafe_allow_html=True)
                st.caption("Owner-assigned tasks with due deadlines")


# =============================================================================
# TAB 2: 📁 MANAGE DOCUMENTS (INGESTION & RE-RUN)
# =============================================================================
with nav_tabs[1]:
    st.markdown("""
        <div class="page-title">Manage Project Documents</div>
        <div class="page-subtitle">Upload your project files (PDF, Word, Excel, CSV, Text) to analyze project health, risks, and timelines.</div>
    """, unsafe_allow_html=True)


    up_col, stats_col = st.columns([1.3, 1], gap="large")

    with up_col:
        with st.container(border=True):
            st.markdown("##### 📤 Upload Project Files")
            uploaded_files = st.file_uploader(
                "Upload Project Files",
                type=["pdf", "docx", "csv", "txt", "xlsx"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

            auto_analyze = st.checkbox("⚡ Run AI Analysis immediately after uploading", value=True)

            idx_btn = st.button("⚡ Process & Analyze Documents", type="primary", use_container_width=True)

            if idx_btn:
                if not uploaded_files:
                    st.warning("Please select at least one document file to upload.")
                else:
                    total_files = len(uploaded_files)
                    all_chunks = []
                    processed_registry = []
                    prog = st.progress(0)

                    with st.spinner("Processing documents & analyzing project..."):
                        for idx, uf in enumerate(uploaded_files):
                            file_name = uf.name
                            if not is_allowed_file(file_name):
                                st.error(f"Unsupported format: {file_name}")
                                continue

                            try:
                                file_bytes = uf.read()
                                doc_info = processor.process_file(
                                    file_input=file_bytes,
                                    filename=file_name,
                                    project_name=project_name
                                )
                                file_chunks = chunker.chunk_text(doc_info["text"], doc_info)
                                all_chunks.extend(file_chunks)

                                processed_registry.append({
                                    "Document Name": file_name,
                                    "Format": file_name.split(".")[-1].upper(),
                                    "Status": "Ready",
                                    "Word Count": doc_info.get("word_count", 0),
                                    "Sections": len(file_chunks)
                                })
                            except Exception as e:
                                st.error(f"Error extracting {file_name}: {e}")

                            prog.progress(int((idx + 1) / total_files * 70))

                        if all_chunks:
                            chunk_texts = [c["text"] for c in all_chunks]
                            embeddings = embedder.embed_texts(chunk_texts)
                            added_count = vector_store.add_chunks(project_name, all_chunks, embeddings)
                            prog.progress(100)

                            st.session_state[f"uploaded_files_{project_name}"] = processed_registry
                            st.session_state[f"last_indexed_{project_name}"] = datetime.now().strftime("%d %b %Y, %H:%M")
                            st.success(f"✓ Successfully processed and saved {len(processed_registry)} document(s).")

                            if auto_analyze:
                                if run_all_project_agents(project_name, top_k):
                                    st.success("✓ AI Analysis complete!")
                                    time.sleep(1.0)
                                    st.rerun()

    with stats_col:
        with st.container(border=True):
            st.markdown("##### 📊 Document Repository Status")
            st.markdown(f"""
                <div style="font-size: 2.2rem; font-weight: 800; color: #7C5CFC; margin: 4px 0;">
                    {len(doc_registry)} <span style="font-size: 1.1rem; color: #7A7F9A; font-weight: 500;">Documents</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #1F2340; margin-bottom: 0.8rem;">
                    Status: <span style="color: #34C38F;">{'Ready for Analysis' if len(doc_registry) > 0 else 'No Documents'}</span>
                </div>
                <div style="font-size: 0.85rem; color: #7A7F9A; margin-bottom: 1rem;">
                    Last Updated: <strong style="color: #1F2340;">{last_indexed}</strong>
                </div>
            """, unsafe_allow_html=True)

            if kb_count > 0:
                if st.button("🔄 Re-run AI Analysis", type="primary", use_container_width=True):
                    if run_all_project_agents(project_name, top_k):
                        st.success("AI Analysis refreshed successfully.")
                        time.sleep(0.5)
                        st.rerun()

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                if st.button("🗑️ Clear Uploaded Project Data", type="secondary", use_container_width=True):
                    vector_store.delete_project_collection(project_name)
                    for k in [
                        f"uploaded_files_{project_name}",
                        f"last_indexed_{project_name}",
                        f"scope_data_{project_name}",
                        f"risk_data_{project_name}",
                        f"blocker_data_{project_name}",
                        f"health_data_{project_name}",
                        f"doc_user_stories_{project_name}",
                        f"doc_risk_register_{project_name}",
                        f"doc_action_items_{project_name}",
                        f"chat_history_{project_name}"
                    ]:
                        st.session_state.pop(k, None)
                    st.success(f"Project data for '{project_name}' has been cleared.")
                    time.sleep(0.5)
                    st.rerun()

    # Ingested Files Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📑 Uploaded Documents")
    if doc_registry:
        df_docs = pd.DataFrame(doc_registry)
        render_dataframe(df_docs, use_container_width=True, hide_index=True)
    else:
        st.caption("No documents registered yet. Upload documents using the form above.")


# =============================================================================
# TAB 3: 🎯 SCOPE · BLOCKERS · RISKS (COMBINED 3-TAB DEEP DIVE)
# =============================================================================
with nav_tabs[2]:
    st.markdown("""
        <div class="page-title">Scope, Blockers &amp; Risks</div>
        <div class="page-subtitle">Deep dive into project deliverables, active blockers, and AI risk detection.</div>
    """, unsafe_allow_html=True)

    inner_scope_tab, inner_blocker_tab, inner_risk_tab = st.tabs([
        "🎯 Scope & Deliverables",
        "🛑 Blockers & Actions",
        "⚠️ Risks & Forecasting"
    ])

    # Inner Tab 1: Scope
    with inner_scope_tab:
        if not scope_data:
            st.info("No scope data extracted yet. Click **[ ⚡ Index & Analyze Documents ]** in Manage Documents.")
        else:
            NS = "Not specified in the available project documents."

            # ── 1. Project Goals & Objectives (Paragraph Narrative + Separate Table) ──
            with st.container(border=True):
                st.markdown("##### 🎯 Project Goals & Objectives")
                goals = scope_data.get("project_goals", [])
                
                narrative_paragraphs = []
                goal_rows = []
                
                if goals:
                    for idx, g in enumerate(goals, 1):
                        if isinstance(g, dict):
                            gt = str(g.get("goal") or "").strip()
                            gd = str(g.get("description") or "").strip()
                            src = str(g.get("source") or g.get("document_name") or "").strip()
                            if not gt or gt == NS:
                                continue
                            if gd and gd != NS and gd != gt:
                                narrative_paragraphs.append(f"<strong>{gt}</strong>: {gd}")
                            else:
                                narrative_paragraphs.append(f"<strong>{gt}</strong>")
                            goal_rows.append({
                                "ID": f"G-{idx:02d}",
                                "Goal / Objective": gt,
                                "Description & Expected Outcome": gd if gd and gd != NS and gd != gt else "Primary documented project goal.",
                                "Source": src if src and src != NS else "—"
                            })
                        elif isinstance(g, str) and g.strip() and g.strip() != NS:
                            narrative_paragraphs.append(f"<strong>{g.strip()}</strong>")
                            goal_rows.append({
                                "ID": f"G-{idx:02d}",
                                "Goal / Objective": g.strip(),
                                "Description & Expected Outcome": "Primary documented project goal.",
                                "Source": "—"
                            })

                # Narrative Paragraphs View
                if narrative_paragraphs:
                    paragraphs_html = "".join([f"<p>{p}</p>" for p in narrative_paragraphs])
                    st.markdown(f"""
                        <div class="narrative-card">
                            <div class="narrative-card-header">📖 Strategic Narrative &amp; Intent</div>
                            {paragraphs_html}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.caption("No explicit goals recorded in narrative form.")

                # Separate Objectives Table
                if goal_rows:
                    st.markdown("<div style='font-size:0.86rem; font-weight:700; color:#5B5875; text-transform:uppercase; letter-spacing:0.04em; margin: 0.6rem 0 0.4rem 0;'>📋 Goals &amp; Objectives Table</div>", unsafe_allow_html=True)
                    render_dataframe(pd.DataFrame(goal_rows), use_container_width=True, hide_index=True)

            # ── 2. Milestones & Schedule Timeline (Visual Chart + Separate Table) ──
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("##### 📍 Milestones & Schedule Timeline")
                milestones = scope_data.get("milestones", []) or scope_data.get("deliverables", [])
                
                # Visual Schedule Timeline Gantt Chart
                st.plotly_chart(create_scope_timeline(milestones), use_container_width=True, config={"displayModeBar": False}, key="scope_page_timeline")

                # Separate Milestones & Schedule Table
                milestone_rows = []
                raw_milestones = scope_data.get("milestones", [])
                if raw_milestones:
                    for idx, m in enumerate(raw_milestones, 1):
                        if isinstance(m, dict):
                            m_name = str(m.get("name") or "").strip()
                            if not m_name or m_name == NS:
                                continue
                            m_desc = str(m.get("description") or "").strip()
                            m_target = str(m.get("target_date") or "").strip()
                            m_status = str(m.get("status") or "Planned").strip()
                            m_src = str(m.get("source") or m.get("document_name") or "").strip()
                            milestone_rows.append({
                                "ID": f"M-{idx:02d}",
                                "Milestone / Phase": m_name,
                                "Description": m_desc if m_desc and m_desc != NS else "—",
                                "Target Date / Deadline": m_target if m_target and m_target != NS else "TBD",
                                "Status": m_status if m_status and m_status != NS else "Planned",
                                "Source": m_src if m_src and m_src != NS else "—"
                            })

                # Fallback to timeline_refs or deliverables if no explicit milestones list
                if not milestone_rows:
                    timeline_refs = scope_data.get("timeline", [])
                    if timeline_refs:
                        for idx, t in enumerate(timeline_refs, 1):
                            lbl = str(t.get("label") or "").strip()
                            val = str(t.get("value") or "").strip()
                            deadline = str(t.get("deadline") or t.get("end_date") or "").strip()
                            src = str(t.get("source") or "").strip()
                            if not lbl or lbl == NS:
                                continue
                            milestone_rows.append({
                                "ID": f"MS-{idx:02d}",
                                "Milestone / Phase": lbl,
                                "Description": f"Timeline reference: {val}" if val and val != NS else "—",
                                "Target Date / Deadline": deadline if deadline and deadline != NS else (val if val and val != NS else "TBD"),
                                "Status": "Scheduled",
                                "Source": src if src and src != NS else "—"
                            })
                    elif delivs_fallback := scope_data.get("deliverables", []):
                        for idx, d in enumerate(delivs_fallback, 1):
                            d_name = str(d.get("name") or "").strip()
                            if not d_name or d_name == NS:
                                continue
                            milestone_rows.append({
                                "ID": f"MS-{idx:02d}",
                                "Milestone / Phase": d_name,
                                "Description": str(d.get("description") or "—").strip(),
                                "Target Date / Deadline": "In Scope",
                                "Status": str(d.get("status") or "Identified").strip(),
                                "Source": str(d.get("source") or "—").strip()
                            })

                if milestone_rows:
                    st.markdown("<div style='font-size:0.86rem; font-weight:700; color:#5B5875; text-transform:uppercase; letter-spacing:0.04em; margin: 0.8rem 0 0.4rem 0;'>🗓️ Milestones &amp; Schedule Breakdown Table</div>", unsafe_allow_html=True)
                    render_dataframe(pd.DataFrame(milestone_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No specific milestone dates or schedule items documented.")


            # ── Deliverables Table ─────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("##### 📦 Deliverables")
                delivs = scope_data.get("deliverables", [])
                if delivs:
                    deliv_rows = []
                    for idx, d in enumerate(delivs, 1):
                        name = str(d.get("name") or "").strip()
                        if not name or name == NS:
                            continue
                        deliv_rows.append({
                            "ID": f"D-{idx:02d}",
                            "Deliverable": name,
                            "Description": str(d.get("description") or "—").strip(),
                            "Status": str(d.get("status") or "Identified").strip(),
                            "Source": str(d.get("source") or "—").strip()
                        })
                    if deliv_rows:
                        render_dataframe(pd.DataFrame(deliv_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No deliverables documented.")
                else:
                    st.caption("No deliverables documented.")

            # ── Responsibilities Table ─────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("##### 👤 Responsibilities — Who Owns What")
                responsibilities = scope_data.get("responsibilities", [])
                if responsibilities:
                    resp_rows = []
                    for r in responsibilities:
                        person = str(r.get("person") or "").strip()
                        task = str(r.get("responsibility") or "").strip()
                        related = str(r.get("related_deliverable") or "").strip()
                        src = str(r.get("source") or "").strip()
                        if not person or person == NS:
                            continue
                        if not task or task == NS:
                            continue
                        resp_rows.append({
                            "Person / Team": person,
                            "Responsibility": task,
                            "Related Deliverable": related if related and related != NS else "—",
                            "Source": src if src and src != NS else "—"
                        })
                    if resp_rows:
                        render_dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No responsibilities explicitly documented.")
                else:
                    st.caption("No responsibilities explicitly documented.")

            # ── Timeline References Table ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("##### 🗓️ Timeline References")
                timeline_refs = scope_data.get("timeline", [])
                if timeline_refs:
                    tl_rows = []
                    for t in timeline_refs:
                        label = str(t.get("label") or "").strip()
                        value = str(t.get("value") or "").strip()
                        start = str(t.get("start_date") or "").strip()
                        end = str(t.get("end_date") or "").strip()
                        deadline = str(t.get("deadline") or "").strip()
                        src = str(t.get("source") or "").strip()
                        if not label or label == NS:
                            continue
                        tl_rows.append({
                            "Reference": label,
                            "Value / Date": value if value and value != NS else "—",
                            "Start": start if start and start != NS else "—",
                            "End / Deadline": (deadline or end) if (deadline or end) and (deadline or end) != NS else "—",
                            "Source": src if src and src != NS else "—"
                        })
                    if tl_rows:
                        render_dataframe(pd.DataFrame(tl_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No timeline references documented.")
                else:
                    st.caption("No timeline references documented.")



    # Inner Tab 2: Blockers
    with inner_blocker_tab:
        if not blocker_data:
            st.info("No blocker data extracted yet.")
        else:
            st.markdown("##### 🛑 Blocker Urgency Matrix")
            b_list = blocker_data.get("blockers", [])
            st.plotly_chart(create_blocker_urgency_matrix(b_list), use_container_width=True, config={"displayModeBar": False}, key="blockers_page_matrix")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📋 Action Items Table")
            actions = blocker_data.get("action_items", [])
            if actions:
                a_rows = [
                    {
                        "Action": a.get("action", "Not specified"),
                        "Owner": a.get("assignee") or a.get("owner", "Team"),
                        "Deadline": a.get("deadline") or a.get("due_date", "TBD"),
                        "Priority": a.get("priority", "Medium"),
                        "Status": a.get("status", "Open")
                    }
                    for a in actions
                ]
                render_dataframe(pd.DataFrame(a_rows), use_container_width=True, hide_index=True)

    # Inner Tab 3: Risks
    with inner_risk_tab:
        if not risk_data:
            st.info("No risk analysis data yet.")
        else:
            r_list = risk_data.get("risks", [])
            st.markdown("##### ⚠️ Risk Priority Matrix")
            st.plotly_chart(create_risk_priority_matrix(r_list), use_container_width=True, config={"displayModeBar": False}, key="risks_page_matrix")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"##### ⚠️ Filterable Risks Log ({len(r_list)})")
            if r_list:
                r_rows = [
                    {
                        "Category": r.get("category", "General"),
                        "Risk": r.get("description", r.get("title", "Risk")),
                        "Severity": r.get("severity", "Medium"),
                        "Evidence": r.get("evidence", "N/A"),
                        "Recommended Action": r.get("recommended_action", "Monitor")
                    }
                    for r in r_list
                ]
                render_dataframe(pd.DataFrame(r_rows), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: 📄 DOCUMENT GENERATION
# =============================================================================
with nav_tabs[3]:
    st.markdown("""
        <div class="page-title">Document Generation</div>
        <div class="page-subtitle">Generate grounded Agile User Stories, formal Risk Registers, and actionable Task Lists.</div>
    """, unsafe_allow_html=True)

    doc_t1, doc_t2, doc_t3 = st.tabs([
        "📝 User Stories",
        "⚠️ Risk Register",
        "📋 Action Items"
    ])

    def get_doc_agent_instance():
        provider = resolve_agent_provider(docgen_model)
        eff_model = groq_risk_model if st.session_state.get("llm_engine", "Groq").startswith("Groq") else docgen_model
        return DocumentationAgent(provider=provider, retriever=retriever, model_name=eff_model)

    with doc_t1:
        if st.button("⚡ Generate User Stories", type="primary"):
            if kb_count == 0:
                st.warning("Please upload and index documents first.")
            else:
                with st.spinner("Synthesizing user stories..."):
                    d_agent = get_doc_agent_instance()
                    s_info = st.session_state.get(f"scope_data_{project_name}")
                    res = d_agent.generate_user_stories(project_id=project_name, scope_data=s_info, top_k=top_k)
                    st.session_state[f"doc_user_stories_{project_name}"] = res.get("data", {}).get("user_stories", [])
                    st.success("User stories generated successfully.")

        stories = st.session_state.get(f"doc_user_stories_{project_name}", [])
        if stories:
            df_stories = pd.DataFrame([
                {
                    "ID": s.get("story_id", f"US-{i:02d}"),
                    "User Story": s.get("user_story", "Not specified"),
                    "Priority": s.get("priority", "Must Have"),
                    "Acceptance Criteria": " | ".join(s.get("acceptance_criteria", []))
                }
                for i, s in enumerate(stories, 1)
            ])
            render_dataframe(df_stories, use_container_width=True, hide_index=True)
            st.download_button("📥 Download User Stories (CSV)", df_stories.to_csv(index=False).encode('utf-8'), f"{project_name}_user_stories.csv", "text/csv")
        else:
            st.caption("No user stories generated yet.")

    with doc_t2:
        if st.button("⚡ Generate Risk Register", type="primary"):
            if kb_count == 0:
                st.warning("Please upload and index documents first.")
            else:
                with st.spinner("Synthesizing formal risk register..."):
                    d_agent = get_doc_agent_instance()
                    r_info = st.session_state.get(f"risk_data_{project_name}")
                    res = d_agent.generate_risk_register(project_id=project_name, risk_data=r_info, top_k=top_k)
                    st.session_state[f"doc_risk_register_{project_name}"] = res.get("data", {}).get("risk_register", [])
                    st.success("Risk register generated successfully.")

        r_reg = st.session_state.get(f"doc_risk_register_{project_name}", [])
        if r_reg:
            df_r_reg = pd.DataFrame([
                {
                    "ID": r.get("risk_id", f"RR-{i:02d}"),
                    "Risk": r.get("risk_description", r.get("risk", "Not specified")),
                    "Category": r.get("category", "General"),
                    "Severity": r.get("severity", "Medium"),
                    "Mitigation": r.get("mitigation", "Monitor"),
                    "Owner": r.get("owner", "Team")
                }
                for i, r in enumerate(r_reg, 1)
            ])
            render_dataframe(df_r_reg, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Risk Register (CSV)", df_r_reg.to_csv(index=False).encode('utf-8'), f"{project_name}_risk_register.csv", "text/csv")
        else:
            st.caption("No risk register generated yet.")

    with doc_t3:
        if st.button("⚡ Generate Action Items", type="primary"):
            if kb_count == 0:
                st.warning("Please upload and index documents first.")
            else:
                with st.spinner("Synthesizing action items..."):
                    d_agent = get_doc_agent_instance()
                    b_info = st.session_state.get(f"blocker_data_{project_name}")
                    res = d_agent.generate_action_items(project_id=project_name, blocker_data=b_info, top_k=top_k)
                    st.session_state[f"doc_action_items_{project_name}"] = res.get("data", {}).get("action_items", [])
                    st.success("Action items generated successfully.")

        a_doc = st.session_state.get(f"doc_action_items_{project_name}", [])
        if a_doc:
            df_a_doc = pd.DataFrame([
                {
                    "ID": a.get("action_id", f"ACT-{i:02d}"),
                    "Action": a.get("action", "Not specified"),
                    "Assignee": a.get("assignee", "Unassigned"),
                    "Deadline": a.get("deadline", "TBD"),
                    "Status": a.get("status", "Open")
                }
                for i, a in enumerate(a_doc, 1)
            ])
            render_dataframe(df_a_doc, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Action Items (CSV)", df_a_doc.to_csv(index=False).encode('utf-8'), f"{project_name}_action_items.csv", "text/csv")
        else:
            st.caption("No action items generated yet.")


# =============================================================================
# TAB 5: 🩺 HEALTH SCORE (DIMENSION BREAKDOWN & REASONS)
# =============================================================================
with nav_tabs[4]:
    st.markdown("""
        <div class="page-title">Project Health Score</div>
        <div class="page-subtitle">Deterministic evaluation across Scope Clarity, Timeline Risk, Blocker Status, and Delivery Risk.</div>
    """, unsafe_allow_html=True)

    if not health_data:
        st.info("Health evaluation not yet calculated. Run analysis from **Manage Documents**.")
    else:
        score_val = health_data.get("overall_score", 0.0)
        h_class = health_data.get("classification", "Moderate")
        confidence = health_data.get("confidence", "Medium")
        dimensions = health_data.get("dimensions", {})
        key_factors = health_data.get("key_factors", [])
        recommendations = health_data.get("recommendations", [])

        h_col1, h_col2 = st.columns([1.1, 1.4], gap="large")

        with h_col1:
            with st.container(border=True):
                st.plotly_chart(create_health_gauge(score_val, h_class), use_container_width=True, config={"displayModeBar": False}, key="health_page_gauge")
                badge_cls = "badge-mint" if score_val >= 80 else ("badge-amber" if score_val >= 60 else "badge-coral")
                st.markdown(f"""
                    <div style="text-align: center; margin-top: -10px;">
                        <span class="{badge_cls}" style="font-size: 0.9rem; padding: 6px 18px;">{h_class.upper()} STATUS</span>
                        <div style="font-size: 0.82rem; color: #7A7F9A; margin-top: 8px;">Confidence: <strong>{confidence}</strong></div>
                    </div>
                """, unsafe_allow_html=True)

        with h_col2:
            with st.container(border=True):
                st.markdown("##### 📊 Dimension Scores & Reasons")
                dim_items = [
                    ("Scope Clarity", "scope_clarity"),
                    ("Timeline Risk", "timeline_risk"),
                    ("Blocker Status", "blocker_status"),
                    ("Delivery Risk", "delivery_risk")
                ]
                for label, key in dim_items:
                    d_obj = dimensions.get(label, {})
                    d_sc = d_obj.get("score", 70.0)
                    d_stat = d_obj.get("status", "Moderate")
                    d_rsn = d_obj.get("reason", "Standard operational parameters.")
                    st.write(f"**{label}** — {int(d_sc)}% ({d_stat})")
                    st.caption(f"Reason: {d_rsn}")
                    st.progress(int(d_sc) / 100)

        st.markdown("<br>", unsafe_allow_html=True)
        fac_col, rec_col = st.columns([1, 1], gap="large")

        with fac_col:
            with st.container(border=True):
                st.markdown("##### 🔍 Why is the project at risk?")
                if key_factors:
                    for kf in key_factors:
                        st.markdown(f"• {kf}")
                else:
                    st.caption("No adverse factors identified.")

        with rec_col:
            with st.container(border=True):
                st.markdown("##### 💡 Strategic Next Steps")
                if recommendations:
                    for r in recommendations:
                        st.markdown(f"💡 {r}")
                else:
                    st.caption("No explicit recommendations recorded.")


# =============================================================================
# FLOATING AI PROJECT ASSISTANT (BOTTOM-RIGHT WIDGET)
# =============================================================================
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = False

chat_history_key = f"chat_history_{project_name}"
if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []

# Sidebar or Bottom Toggle for Chat Widget
with st.sidebar:
    st.caption("ASSISTANT")
    toggle_label = "💬 Open AI Chat Assistant" if not st.session_state["chat_open"] else "✕ Minimize AI Assistant"
    if st.button(toggle_label, use_container_width=True):
        st.session_state["chat_open"] = not st.session_state["chat_open"]
        st.rerun()

if st.session_state["chat_open"]:
    with st.expander("💬 AI Project Assistant (Grounded Chat)", expanded=True):
        chat_container = st.container(height=280)
        with chat_container:
            for msg in st.session_state[chat_history_key]:
                role = msg.get("role", "assistant")
                with st.chat_message(role):
                    st.markdown(msg.get("content", ""))
                    sources = msg.get("sources", [])
                    if sources and role == "assistant":
                        chips = "".join([f"<span class='source-chip'>📄 {s}</span>" for s in sources])
                        st.markdown(f"<div style='margin-top: 4px;'>{chips}</div>", unsafe_allow_html=True)

        user_chat_input = st.chat_input("Ask a question grounded in project artifacts...")
        if user_chat_input:
            if kb_count == 0:
                st.warning("Please upload and index documents first.")
            else:
                st.session_state[chat_history_key].append({"role": "user", "content": user_chat_input})
                provider = resolve_agent_provider(chat_model)
                assistant = ConversationalProjectAssistant(
                    provider=provider,
                    retriever=retriever,
                    model_name=groq_risk_model if st.session_state.get("llm_engine", "Groq").startswith("Groq") else chat_model
                )
                assistant.set_history([
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state[chat_history_key][:-1]
                ])

                resp = assistant.ask(
                    question=user_chat_input,
                    project_id=project_name,
                    scope_data=st.session_state.get(f"scope_data_{project_name}"),
                    risk_data=st.session_state.get(f"risk_data_{project_name}"),
                    blocker_data=st.session_state.get(f"blocker_data_{project_name}"),
                    health_data=st.session_state.get(f"health_data_{project_name}"),
                    top_k=top_k
                )

                st.session_state[chat_history_key].append({
                    "role": "assistant",
                    "content": resp.answer,
                    "sources": resp.sources
                })
                st.rerun()

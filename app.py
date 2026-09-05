"""
AI Project Intelligence & Risk Advisor — Professional Dashboard UI

Interactive web interface for uploading project artifacts (PDF, DOCX, CSV, TXT),
building a vector knowledge base in ChromaDB, and querying project intelligence with grounded answers.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from utils.helpers import is_allowed_file, format_source_attribution
from ingestion.document_processor import DocumentProcessor
from rag.chunker import TextChunker
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.retriever import Retriever
from rag.qa import AnswerGenerator

# Page configuration
st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Professional CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        color: #FFFFFF;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* Card Panels */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Badge styling */
    .file-badge {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        border: 1px solid #CBD5E1;
    }
    
    .source-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
        border: 1px solid #BFDBFE;
    }
    
    /* Custom Answer Box */
    .answer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 1.2rem 1.4rem;
        border-radius: 8px;
        font-size: 1.02rem;
        color: #1E293B;
        line-height: 1.6;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    
    /* Step Status Timeline */
    .status-step {
        font-size: 0.92rem;
        color: #0F172A;
        padding: 4px 0;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize RAG Pipeline components with caching
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


# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown("### ⚡ AI Risk Advisor")
    st.caption("Project Intelligence Platform")
    
    st.markdown("---")
    st.markdown("##### 📁 Project Workspace")
    project_name = st.text_input(
        "Workspace",
        value="Main Workspace",
        label_visibility="collapsed"
    )
    if not project_name or not project_name.strip():
        project_name = "Main Workspace"

    
    st.markdown("##### ⚙️ Retrieval Depth")
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=10, value=5)
    
    st.markdown("##### 🔑 API Key Configuration")
    user_api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Stored securely in memory. Also loaded automatically from .env file."
    )

    st.markdown("---")
    st.markdown("##### 📄 Supported Formats")
    st.markdown("""
        <span class="file-badge">PDF</span>
        <span class="file-badge">DOCX</span>
        <span class="file-badge">CSV</span>
        <span class="file-badge">TXT</span>
    """, unsafe_allow_html=True)




# Top Executive Banner
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">AI Project Intelligence & Risk Advisor</div>
        <div class="hero-subtitle">Upload project artifacts, generate persistent vector knowledge, and extract grounded project insights.</div>
    </div>
""", unsafe_allow_html=True)


# Main Workspace Grid
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Ingest Project Artifacts")
    st.caption("Select proposal PDFs, SRS docx files, sprint meeting notes, or task CSVs.")
    
    uploaded_files = st.file_uploader(
        "Upload Artifacts",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f"**Selected Artifacts ({len(uploaded_files)})**")
        for f in uploaded_files:
            st.caption(f"📄 {f.name} • {f.size / 1024:.1f} KB")

    process_btn = st.button("⚡ Index & Process Artifacts", type="primary", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one project artifact before processing.")
        else:
            status_box = st.container()
            with status_box:
                st.markdown("##### 🔄 Processing Pipeline")
                progress_bar = st.progress(0)
                
                total_files = len(uploaded_files)
                all_chunks = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    file_name = uploaded_file.name
                    
                    if not is_allowed_file(file_name):
                        st.error(f"Unsupported file format: {file_name}")
                        continue

                    # Step 1 & 2: Ingest & Clean
                    st.markdown(f"<div class='status-step'>✓ <b>{file_name}</b> uploaded</div>", unsafe_allow_html=True)
                    
                    try:
                        file_bytes = uploaded_file.read()
                        doc_info = processor.process_file(
                            file_input=file_bytes,
                            filename=file_name,
                            project_name=project_name
                        )
                        st.caption(f"Extracted {doc_info['word_count']} words")
                    except Exception as e:
                        st.error(f"Extraction error for {file_name}: {e}")
                        continue

                    # Step 3: Chunking
                    file_chunks = chunker.chunk_text(doc_info["text"], doc_info)
                    all_chunks.extend(file_chunks)
                    st.caption(f"Segmented into {len(file_chunks)} context chunks")

                    progress_bar.progress(int((idx + 1) / total_files * 70))

                if all_chunks:
                    # Step 4: Embeddings
                    st.markdown("<div class='status-step'>✓ Generating dense vector embeddings...</div>", unsafe_allow_html=True)
                    chunk_texts = [c["text"] for c in all_chunks]
                    embeddings = embedder.embed_texts(chunk_texts)

                    # Step 5: Index ChromaDB
                    st.markdown(f"<div class='status-step'>✓ Storing chunks in ChromaDB workspace (<b>{project_name}</b>)...</div>", unsafe_allow_html=True)
                    added_count = vector_store.add_chunks(project_name, all_chunks, embeddings)
                    progress_bar.progress(100)

                    st.success(f"Successfully processed {total_files} document(s) and stored {added_count} chunks.")


with col2:
    st.subheader("2. Grounded Project Q&A")
    st.caption("Ask questions about deliverables, project scope, blockers, or deadlines.")

    # Search Query Input
    user_query = st.text_input(
        "Project Question",
        placeholder="e.g., What tasks are currently incomplete?",
        label_visibility="collapsed"
    )


    ask_btn = st.button("🔍 Search Project Intelligence", type="secondary", use_container_width=True)

    if ask_btn or user_query:
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving grounded knowledge & generating answer..."):
                retrieved_chunks = retriever.retrieve(
                    query=user_query,
                    project_name=project_name,
                    top_k=top_k
                )

                answer, sources, chunks_used = qa_generator.generate_answer(
                    query=user_query,
                    retrieved_chunks=retrieved_chunks,
                    api_key=user_api_key
                )


            st.markdown("##### 💬 Grounded AI Answer")
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

            # Grounded Sources Badges
            st.markdown("##### 📚 Verified Document Sources")
            if sources:
                source_badges = "".join([f"<span class='source-tag'>📄 {src}</span>" for src in sources])
                st.markdown(f"<div>{source_badges}</div>", unsafe_allow_html=True)
            else:
                st.caption("No sources found in knowledge base.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Retrieved Context Inspection
            with st.expander("🔍 Inspect Retrieved Context Chunks & Relevance Scores"):
                if not chunks_used:
                    st.info("No relevant chunks retrieved from ChromaDB for this query.")
                else:
                    for i, chunk in enumerate(chunks_used, 1):
                        meta = chunk.get("metadata", {})
                        dist = chunk.get("distance", 0.0)
                        st.markdown(f"**Chunk {i}** • Source: `{meta.get('source', 'N/A')}` • Similarity Distance: `{dist:.4f}`")
                        st.caption(f"_{chunk['text']}_")
                        st.divider()

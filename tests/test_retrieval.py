"""
Integration tests for RAG Retrieval & Grounded Q&A pipeline.
Tests end-to-end embedding, ChromaDB indexing, semantic search, and grounding checks.
"""

import os
import shutil
import pytest
from generate_sample_data import generate_all_sample_files
from ingestion.document_processor import DocumentProcessor
from rag.chunker import TextChunker
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.retriever import Retriever
from rag.qa import AnswerGenerator


TEST_PROJECT_NAME = "Test Retrieval Project"
TEST_CHROMA_DIR = "./data/test_chroma"


@pytest.mark.parametrize(
    ("chunk_count", "expected_depth"),
    [(0, 3), (1, 3), (25, 5), (100, 10), (500, 10)],
)
def test_automatic_retrieval_depth_scales_with_workspace_size(chunk_count, expected_depth):
    assert Retriever.automatic_top_k(chunk_count) == expected_depth


@pytest.fixture(scope="module", autouse=True)
def setup_vector_store():
    # 1. Generate sample data
    generate_all_sample_files("./sample_data")

    # 2. Setup isolated test ChromaDB
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=300, overlap=30)
    embedder = EmbeddingManager()
    vector_store = VectorStoreManager(persist_dir=TEST_CHROMA_DIR)

    # Ingest and index all sample documents
    sample_files = [
        ("sample_proposal.pdf", "./sample_data/sample_proposal.pdf"),
        ("sample_srs.docx", "./sample_data/sample_srs.docx"),
        ("sample_meeting_notes.txt", "./sample_data/sample_meeting_notes.txt"),
        ("sample_tasks.csv", "./sample_data/sample_tasks.csv"),
    ]

    all_chunks = []
    for fname, fpath in sample_files:
        doc_info = processor.process_file(fpath, fname, TEST_PROJECT_NAME)
        chunks = chunker.chunk_text(doc_info["text"], doc_info)
        all_chunks.extend(chunks)

    # Embed and index
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    vector_store.add_chunks(TEST_PROJECT_NAME, all_chunks, embeddings)

    yield {
        "vector_store": vector_store,
        "embedder": embedder,
        "retriever": Retriever(embedder, vector_store)
    }

    # Cleanup test chroma database
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)


def test_retrieval_project_objective(setup_vector_store):
    retriever = setup_vector_store["retriever"]
    query = "What is the objective of the project?"
    results = retriever.retrieve(query, TEST_PROJECT_NAME, top_k=3)

    assert len(results) > 0
    top_text = results[0]["text"]
    assert "Student Project Management System" in top_text or "objective" in top_text.lower()


def test_retrieval_project_blockers(setup_vector_store):
    retriever = setup_vector_store["retriever"]
    query = "What is currently blocking the project?"
    results = retriever.retrieve(query, TEST_PROJECT_NAME, top_k=3)

    assert len(results) > 0
    top_text = " ".join([r["text"] for r in results])
    assert "Payment" in top_text or "API credentials" in top_text or "BLOCKED" in top_text


def test_grounded_qa_out_of_context(setup_vector_store):
    retriever = setup_vector_store["retriever"]
    qa = AnswerGenerator()  # No API key required for fallback test

    query = "What is the launch date for the Mars rover mission?"
    results = retriever.retrieve(query, TEST_PROJECT_NAME, top_k=3)

    answer, sources, retrieved_chunks = qa.generate_answer(query, results)
    
    # Verify sources and retrieved chunks are passed cleanly
    assert len(retrieved_chunks) > 0
    assert len(sources) > 0


def test_retrieval_project_workspace_isolation(setup_vector_store):
    """Verify Chroma collections scoped by workspace never return another project's chunks."""
    embedder = setup_vector_store["embedder"]
    vector_store = setup_vector_store["vector_store"]
    retriever = setup_vector_store["retriever"]
    chunker = TextChunker(chunk_size=300, overlap=30)

    def index_workspace(name: str, text: str, source: str):
        doc_info = {
            "text": text,
            "source": source,
            "file_type": "txt",
            "project_name": name,
            "word_count": len(text.split()),
        }
        chunks = chunker.chunk_text(text, doc_info)
        embeddings = embedder.embed_texts([c["text"] for c in chunks])
        vector_store.add_chunks(name, chunks, embeddings)

    index_workspace(
        "Isolation_Project_A",
        "UNIQUE_MARKER_ALPHA_ONLY documented risk for workspace A payment integration.",
        "alpha_notes.txt",
    )
    index_workspace(
        "Isolation_Project_B",
        "UNIQUE_MARKER_BETA_ONLY documented blocker for workspace B database migration.",
        "beta_notes.txt",
    )

    results_a = retriever.retrieve("What are the risks?", "Isolation_Project_A", top_k=3)
    results_b = retriever.retrieve("What are the risks?", "Isolation_Project_B", top_k=3)
    text_a = " ".join(r["text"] for r in results_a)
    text_b = " ".join(r["text"] for r in results_b)

    assert "UNIQUE_MARKER_ALPHA_ONLY" in text_a
    assert "UNIQUE_MARKER_BETA_ONLY" not in text_a
    assert "UNIQUE_MARKER_BETA_ONLY" in text_b
    assert "UNIQUE_MARKER_ALPHA_ONLY" not in text_b

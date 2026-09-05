"""
Unit tests for text chunking strategy.
Verifies chunk word limits, overlap retention, and metadata attached to each chunk.
"""

from rag.chunker import TextChunker, chunk_document


def test_chunking_word_limit_and_overlap():
    # Sample text of 100 words
    words = [f"word_{i}" for i in range(100)]
    sample_text = " ".join(words)

    doc_info = {
        "text": sample_text,
        "source": "test_doc.txt",
        "file_type": "txt",
        "project_name": "Test Project"
    }

    # Set chunk_size=40, overlap=10
    chunker = TextChunker(chunk_size=40, overlap=10)
    chunks = chunker.chunk_text(sample_text, doc_info)

    assert len(chunks) > 1
    # Check first chunk has 40 words
    first_chunk_words = chunks[0]["text"].split()
    assert len(first_chunk_words) == 40
    assert first_chunk_words[0] == "word_0"
    assert first_chunk_words[-1] == "word_39"

    # Check second chunk overlap: step = 40 - 10 = 30. Second chunk starts at word_30
    second_chunk_words = chunks[1]["text"].split()
    assert second_chunk_words[0] == "word_30"
    assert second_chunk_words[9] == "word_39"  # Overlap words preserved!

    # Check metadata propagation
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "test_doc.txt"
        assert chunk["metadata"]["project_name"] == "Test Project"
        assert "chunk_id" in chunk

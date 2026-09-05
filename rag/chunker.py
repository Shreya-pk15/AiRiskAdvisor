"""
Text Chunking Module.

===============================================================================
STUDENT EDUCATIONAL NOTES — WHY WE CHUNK & USE OVERLAP:
===============================================================================
1. WHY ARE WE CHUNKING?
   - Context Window & Embedding Limits: Embedding models (e.g., Sentence Transformers)
     have input length limits (typically 256 or 512 tokens). Passing huge documents
     truncates content or dilutes semantic representation.
   - Precision Retrieval: Chunking allows the vector store to pinpoint the exact 
     paragraph or section relevant to a user query, rather than retrieving an entire document.

2. WHY IS OVERLAP NEEDED?
   - Context Boundary Protection: Splitting text hard at 500 words might split a 
     crucial sentence, requirement, or meeting decision across two separate chunks.
   - Overlap (e.g., 50 words) ensures that key concepts at chunk boundaries are 
     preserved in both chunks, preventing loss of context during retrieval.

3. WHY SHOULDN'T WE EMBED AN ENTIRE 20-PAGE PDF AS ONE VECTOR?
   - Semantic Loss / Information Dilution: A single 384-dimensional embedding vector
     cannot capture 20 pages of diverse technical facts, deadlines, and risks.
     Individual specific facts (e.g., "API credentials delayed") get lost in the average.
===============================================================================
"""

from typing import List, Dict, Any
import uuid


class TextChunker:
    """
    Sliding-window text chunker using word counts with configurable size and overlap.
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        """
        Args:
            chunk_size (int): Target number of words per chunk (default: 400 words ~500 tokens).
            overlap (int): Number of overlapping words between consecutive chunks (default: 50 words).
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be strictly smaller than chunk_size.")
        
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(
        self,
        text: str,
        document_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Splits a normalized text string into overlapping chunks while attaching metadata.

        Args:
            text (str): Extracted & cleaned document text.
            document_info (Dict[str, Any]): Metadata dict containing source, file_type, project_name, etc.

        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries containing text, metadata, and identifiers.
        """
        if not text.strip():
            return []

        words = text.split()
        if not words:
            return []

        chunks = []
        step = self.chunk_size - self.overlap
        chunk_index = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_str = " ".join(chunk_words)

            chunk_id = f"{document_info.get('source', 'doc')}_chunk_{chunk_index}_{uuid.uuid4().hex[:6]}"
            
            chunk_data = {
                "chunk_id": chunk_id,
                "text": chunk_str,
                "chunk_index": chunk_index,
                "metadata": {
                    "source": document_info.get("source", "unknown"),
                    "file_type": document_info.get("file_type", "unknown"),
                    "project_name": document_info.get("project_name", "Default Project"),
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_words)
                }
            }
            chunks.append(chunk_data)
            chunk_index += 1

            # If we reach the end of the words list, stop
            if i + self.chunk_size >= len(words):
                break

        return chunks


def chunk_document(
    document_info: Dict[str, Any],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """Helper function to chunk a document dictionary."""
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_text(document_info["text"], document_info)

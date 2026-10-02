"""
Retrieval Module for Semantic Search.

Converts query into vector embedding, searches project collection in ChromaDB,
and returns top-k relevant text chunks with metadata.
"""

import math
from typing import List, Dict, Any, Optional
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager


class Retriever:
    """Handles semantic similarity search over project collections."""

    @staticmethod
    def automatic_top_k(chunk_count: int) -> int:
        """Scale retrieval depth with workspace size while bounding model context."""
        return min(10, max(3, math.ceil(math.sqrt(max(1, chunk_count)))))

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store_manager: Optional[VectorStoreManager] = None
    ):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store_manager = vector_store_manager or VectorStoreManager()

    def retrieve(
        self,
        query: str,
        project_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant text chunks for a given query string within a specific project.

        Args:
            query (str): User question string.
            project_name (str): Associated project name.
            top_k (int): Number of top chunks to retrieve (default: 5).

        Returns:
            List[Dict[str, Any]]: Retrieved text chunks with distance metrics and source metadata.
        """
        if not query.strip():
            return []

        # 1. Embed query
        query_embedding = self.embedding_manager.embed_query(query)

        # 2. Query vector store
        retrieved_chunks = self.vector_store_manager.query(
            project_name=project_name,
            query_embedding=query_embedding,
            top_k=top_k
        )

        return retrieved_chunks

    def retrieve_all(self, project_name: str) -> List[Dict[str, Any]]:
        """Return all chunks from one project collection for comprehensive analysis."""
        return self.vector_store_manager.get_all_chunks(project_name=project_name)

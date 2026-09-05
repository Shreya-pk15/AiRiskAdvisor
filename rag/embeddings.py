"""
Embeddings Generation Module using Sentence Transformers.

Uses lightweight, fast, local embedding models suitable for student projects
(e.g., 'all-MiniLM-L6-v2', generating 384-dimensional dense vectors).
"""

import os
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Wrapper around SentenceTransformer for generating vector embeddings."""

    def __init__(self, model_name: str = None):
        """
        Initialize Sentence Transformer model.
        Reads model_name from parameter or env variable EMBEDDING_MODEL_NAME.
        """
        if not model_name:
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model on first access."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a list of text strings (chunks).

        Args:
            texts (List[str]): Batch of text chunk strings.

        Returns:
            List[List[float]]: Matrix of float vector embeddings.
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate vector embedding for a single search query string.

        Args:
            query (str): User question text.

        Returns:
            List[float]: Single 1D float vector embedding.
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty.")
            
        embedding = self.model.encode(query, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()

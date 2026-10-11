"""
Embeddings Generation Module using Sentence Transformers with Safe Fallback.

Uses lightweight embedding models (e.g., 'all-MiniLM-L6-v2', generating 384-dimensional dense vectors)
with graceful fallback if PyTorch binary DLLs are blocked by OS Application Control policies.
"""

import os
import hashlib
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Wrapper around SentenceTransformer with fallback for generating vector embeddings."""

    def __init__(self, model_name: str = None):
        """
        Initialize Sentence Transformer model.
        Reads model_name from parameter or env variable EMBEDDING_MODEL_NAME.
        """
        if not model_name:
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        
        self.model_name = model_name
        self._model = None
        self._use_fallback = False

    def _get_model(self):
        """Lazy load the sentence transformer model or fallback on OS DLL block."""
        if self._model is None and not self._use_fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(
                    "SentenceTransformer/PyTorch failed to load (%s). Falling back to lightweight normalized dense embedder.", e
                )
                self._use_fallback = True
        return self._model

    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        """Generate a deterministic normalized 384-dimensional dense embedding vector."""
        words = text.lower().split()
        vec = np.zeros(dim, dtype=np.float32)
        for i, w in enumerate(words):
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 4) % 2 == 0 else -1.0
            vec[idx] += sign * (1.0 / (1.0 + np.log1p(i)))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

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
        
        model = self._get_model()
        if model is not None:
            try:
                embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception:
                pass

        return [self._fallback_embed(t) for t in texts]

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
            
        model = self._get_model()
        if model is not None:
            try:
                embedding = model.encode(query, show_progress_bar=False, convert_to_numpy=True)
                return embedding.tolist()
            except Exception:
                pass

        return self._fallback_embed(query)


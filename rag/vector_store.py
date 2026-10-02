"""
Vector Store Module using ChromaDB.

Manages persistent local vector collections separated by project name to ensure
complete isolation (e.g., Project A collection will never return results for Project B).
"""

import os
import re
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional


class VectorStoreManager:
    """Manages persistent ChromaDB vector store operations."""

    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize ChromaDB persistent client.

        Args:
            persist_dir (str, optional): Directory path for persistent Chroma storage.
        """
        if not persist_dir:
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

    def _sanitize_collection_name(self, project_name: str) -> str:
        """
        Sanitize a project name to comply with ChromaDB collection name rules:
        - Must be between 3 and 63 characters
        - Must start and end with an alphanumeric character
        - Contains only alphanumeric, underscore, or hyphen
        """
        if not project_name or not project_name.strip():
            return "default_project"

        # Replace non-alphanumeric chars with underscores
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", project_name.strip().lower())
        # Collapse multiple underscores
        clean_name = re.sub(r"_+", "_", clean_name).strip("_")
        
        if len(clean_name) < 3:
            clean_name = f"proj_{clean_name}"
            clean_name = clean_name.rstrip("_")
            
        # Ensure starts and ends with an alphanumeric character
        while clean_name and not clean_name[-1].isalnum():
            clean_name = clean_name[:-1]
            
        while clean_name and not clean_name[0].isalnum():
            clean_name = clean_name[1:]
            
        if len(clean_name) < 3:
            clean_name = "default_project"
            
        if len(clean_name) > 63:
            clean_name = clean_name[:63]
            while clean_name and not clean_name[-1].isalnum():
                clean_name = clean_name[:-1]
                
        return clean_name


    def get_or_create_collection(self, project_name: str):
        """
        Get or create a dedicated ChromaDB collection for the given project.

        Args:
            project_name (str): Project identifier string.

        Returns:
            chromadb.Collection: Dedicated collection instance.
        """
        coll_name = self._sanitize_collection_name(project_name)
        return self.client.get_or_create_collection(
            name=coll_name,
            metadata={"project_name": project_name}
        )

    def add_chunks(
        self,
        project_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Store text chunks and their embeddings in the project's ChromaDB collection.

        Args:
            project_name (str): Associated project name.
            chunks (List[Dict[str, Any]]): List of chunk objects with 'text' and 'metadata'.
            embeddings (List[List[float]]): Corresponding dense vector embeddings.

        Returns:
            int: Number of chunks added.
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks must match the number of embeddings.")

        collection = self.get_or_create_collection(project_name)

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        
        # Format metadata to ensure primitive types for ChromaDB
        metadatas = []
        for chunk in chunks:
            raw_meta = chunk.get("metadata", {})
            safe_meta = {}
            for k, v in raw_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)
            metadatas.append(safe_meta)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(chunks)

    def query(
        self,
        project_name: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the project collection for top_k most similar text chunks.

        Args:
            project_name (str): Target project.
            query_embedding (List[float]): User question vector embedding.
            top_k (int): Number of results to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved chunks with distance scores and metadata.
        """
        collection = self.get_or_create_collection(project_name)

        count = collection.count()
        if count == 0:
            return []

        # Ensure top_k does not exceed total count in collection
        actual_top_k = min(top_k, count)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)

            for doc, meta, dist, cid in zip(docs, metas, dists, ids):
                formatted_results.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist)
                })

        return formatted_results

    def get_all_chunks(self, project_name: str) -> List[Dict[str, Any]]:
        """Return every stored chunk in the project's isolated collection."""
        collection = self.get_or_create_collection(project_name)
        results = collection.get(include=["documents", "metadatas"])

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or [{}] * len(documents)
        chunk_ids = results.get("ids") or [""] * len(documents)

        return [
            {"chunk_id": chunk_id, "text": document, "metadata": metadata or {}}
            for chunk_id, document, metadata in zip(chunk_ids, documents, metadatas)
        ]

    def delete_project_collection(self, project_name: str):
        """Delete an entire project collection from ChromaDB."""
        coll_name = self._sanitize_collection_name(project_name)
        try:
            self.client.delete_collection(name=coll_name)
        except Exception:
            pass

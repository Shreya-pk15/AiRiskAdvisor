import logging
import re
from abc import ABC
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Reusable base for all project-intelligence agents."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    @staticmethod
    def _normalize_chunks(retrieved_chunks: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        if not retrieved_chunks:
            return []

        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for chunk in retrieved_chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            source = str(chunk.get("metadata", {}).get("source", "Unknown Document")).strip() or "Unknown Document"
            signature = (source, text)
            if signature in seen:
                continue
            seen.add(signature)
            normalized.append({
                "text": text,
                "metadata": chunk.get("metadata", {}),
                "source": source,
            })
        return normalized

    @staticmethod
    def _search_targeted_chunks(
        retriever: Any,
        project_id: str,
        top_k: int,
        queries: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Merge targeted searches by similarity and cap context at the requested depth."""
        limit = max(1, int(top_k))
        best_chunks: Dict[Tuple[str, str], Tuple[float, int, Dict[str, Any]]] = {}
        sequence = 0

        for query in queries:
            for chunk in retriever.retrieve(query=query, project_name=project_id, top_k=limit):
                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue
                metadata = chunk.get("metadata", {})
                source = str(metadata.get("source") or chunk.get("source") or "Unknown Document")
                key = (source, text)
                try:
                    distance = float(chunk.get("distance", float("inf")))
                except (TypeError, ValueError):
                    distance = float("inf")

                existing = best_chunks.get(key)
                if existing is None or distance < existing[0]:
                    best_chunks[key] = (distance, sequence, chunk)
                sequence += 1

        ranked_chunks = sorted(best_chunks.values(), key=lambda item: (item[0], item[1]))
        return [chunk for _, _, chunk in ranked_chunks[:limit]]

    @staticmethod
    def _extract_source(chunk: Dict[str, Any]) -> str:
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        source = metadata.get("source") or chunk.get("source") or "Unknown Document"
        return str(source)

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return "Not specified in the available project documents."
        text = str(value).strip()
        return text if text else "Not specified in the available project documents."

    @staticmethod
    def _deduplicate(items: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for item in items:
            val = item.strip()
            if not val:
                continue
            key = re.sub(r"\s+", " ", val).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(val)
        return out

    @staticmethod
    def _find_match(pattern: str, text: str) -> str | None:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        value = match.group(1).strip()
        return value if value else None

    def _validate_model(self, model_cls: type[BaseModel], payload: Dict[str, Any]) -> BaseModel:
        try:
            return model_cls.model_validate(payload)
        except ValidationError as exc:
            logger.warning("Validation failed for %s: %s", model_cls.__name__, exc)
            raise ValueError(f"Invalid structured output for {model_cls.__name__}: {exc}") from exc

    def _summarize_context(self, retrieved_chunks: Sequence[Dict[str, Any]] | None) -> str:
        chunks = self._normalize_chunks(retrieved_chunks)
        if not chunks:
            return "No project context was available for analysis."
        return "\n\n".join(f"[Source: {self._extract_source(chunk)}]\n{chunk.get('text', '')}" for chunk in chunks)

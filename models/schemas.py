"""
Document data models for AI Project Intelligence & Risk Advisor.

Defines the core data structures:
- Project: Represents a software project workspace.
- Document: Represents an uploaded document file (PDF, DOCX, CSV, TXT).
- Chunk: Represents a segmented text slice prepared for embedding & vector indexing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class Project:
    """Represents a project workspace containing multiple documents."""
    project_name: str
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "created_at": self.created_at
        }


@dataclass
class Document:
    """Represents an ingested project document."""
    filename: str
    file_type: str
    text: str
    project_id: str
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    uploaded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "project_id": self.project_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "uploaded_at": self.uploaded_at
        }


@dataclass
class Chunk:
    """
    Represents a chunk of text extracted from a document,
    ready for embedding generation and storage in ChromaDB.
    """
    chunk_id: str
    document_id: str
    project_id: str
    source: str
    file_type: str
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert chunk metadata to a ChromaDB-compatible primitive key-value dictionary."""
        base_meta = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "source": self.source,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index
        }
        # Merge any additional metadata, ensuring scalar types (str, int, float, bool)
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                base_meta[k] = v
            else:
                base_meta[k] = str(v)
        return base_meta

"""
Scope and Deliverable Extraction Agent.

Extracts project goals, deliverables, milestones, timelines, and responsibilities
from uploaded project documents using grounded RAG retrieval and Google Gemini API.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from agents.prompts import SCOPE_PROMPT
from agents.schemas import (
    DeliverableItem,
    MilestoneItem,
    ProjectGoal,
    ResponsibilityItem,
    ScopeExtractionOutput,
    TimelineItem,
)
from llm.base_provider import BaseLLMProvider
from llm.gemini_provider import GeminiProvider
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

FALLBACK_NOT_SPECIFIED = "Not specified in the available project documents."


class ScopeExtractionAgent(BaseAgent):
    """
    Scope & Deliverable Extraction Agent.

    Uses RAG retrieval with target queries filtered by project_id / project_name
    and sends context to Gemini to produce a structured Pydantic ScopeExtractionOutput.
    """

    TARGET_QUERIES = [
        "project goals objectives scope expected outcomes and deliverables features modules systems reports APIs documentation",
        "milestones planned sprint actual sprint project phases releases deadlines target dates schedule",
        "responsibilities assignments person team task role owner",
    ]
    MILESTONE_DETAIL_QUERY = "planned sprint actual sprint product backlog milestones phase release iteration"

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        retriever: Optional[Retriever] = None,
        model_name: Optional[str] = None
    ):
        super().__init__(model_name=model_name)
        self.provider = provider
        self.retriever = retriever or Retriever()

    def run(self, project_id: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute full Scope & Deliverable Extraction workflow for a given project_id.

        Args:
            project_id (str): Project identifier / workspace name.
            top_k (int): Maximum number of distinct context chunks across targeted queries.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "data": ScopeExtractionOutput dictionary
                - "metadata": Execution metadata (agent_name, provider, execution_time_seconds, status, error)
        """
        start_time = time.time()
        agent_name = "Scope & Deliverable Agent"
        provider_name = "Gemini"

        # Initialize provider if not supplied — prefer Groq if Gemini key is invalid
        if self.provider is None:
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            use_groq = (
                not gemini_key
                or gemini_key.startswith("AQ.")
                or gemini_key == "your_gemini_api_key_here"
            )
            if use_groq and os.getenv("GROQ_API_KEY"):
                try:
                    from llm.groq_provider import GroqProvider
                    groq_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-120b")
                    self.provider = GroqProvider(model_name=groq_model)
                    provider_name = "Groq"
                except Exception as e:
                    exec_time = round(time.time() - start_time, 2)
                    return {
                        "data": self._empty_scope_output().model_dump(),
                        "metadata": {
                            "agent_name": agent_name,
                            "provider": provider_name,
                            "execution_time_seconds": exec_time,
                            "status": "Failed",
                            "error": f"Failed to initialize Groq Provider: {e}"
                        }
                    }
            else:
                try:
                    self.provider = GeminiProvider(model_name=self.model_name)
                except Exception as e:
                    # Last resort: try Groq
                    if os.getenv("GROQ_API_KEY"):
                        try:
                            from llm.groq_provider import GroqProvider
                            groq_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-120b")
                            self.provider = GroqProvider(model_name=groq_model)
                            provider_name = "Groq"
                        except Exception as e2:
                            exec_time = round(time.time() - start_time, 2)
                            return {
                                "data": self._empty_scope_output().model_dump(),
                                "metadata": {
                                    "agent_name": agent_name,
                                    "provider": provider_name,
                                    "execution_time_seconds": exec_time,
                                    "status": "Failed",
                                    "error": f"Failed to initialize LLM Provider: {e2}"
                                }
                            }
                    else:
                        exec_time = round(time.time() - start_time, 2)
                        return {
                            "data": self._empty_scope_output().model_dump(),
                            "metadata": {
                                "agent_name": agent_name,
                                "provider": provider_name,
                                "execution_time_seconds": exec_time,
                                "status": "Failed",
                                "error": f"Failed to initialize LLM Provider: {e}"
                            }
                        }

        try:
            # 1. Targeted RAG Retrieval filtered by project_id
            retrieved_chunks = self._retrieve_targeted_chunks(project_id=project_id, top_k=top_k)

            if not retrieved_chunks:
                exec_time = round(time.time() - start_time, 2)
                empty_res = self._empty_scope_output()
                return {
                    "data": empty_res.model_dump(),
                    "metadata": {
                        "agent_name": agent_name,
                        "provider": provider_name,
                        "execution_time_seconds": exec_time,
                        "status": "Success",
                        "error": None,
                        "note": "No relevant project context found in knowledge base."
                    }
                }

            # 2. Extract structured scope using single LLM call
            scope_output = self._extract_with_llm(retrieved_chunks)
            exec_time = round(time.time() - start_time, 2)

            return {
                "data": scope_output.model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "provider": provider_name,
                    "execution_time_seconds": exec_time,
                    "status": "Success",
                    "error": None
                }
            }

        except Exception as exc:
            logger.error("Scope extraction agent error: %s", exc, exc_info=True)
            exec_time = round(time.time() - start_time, 2)
            return {
                "data": self._empty_scope_output().model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "provider": provider_name,
                    "execution_time_seconds": exec_time,
                    "status": "Failed",
                    "error": str(exc)
                }
            }

    def _retrieve_targeted_chunks(self, project_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve balanced scope context and include omitted structured milestone rows."""
        chunks = self._search_targeted_chunks(
            self.retriever, project_id, top_k, self.TARGET_QUERIES
        )
        if self._extract_structured_milestones(self._normalize_chunks(chunks)):
            return chunks

        detail_chunks = self.retriever.retrieve(
            query=self.MILESTONE_DETAIL_QUERY,
            project_name=project_id,
            top_k=max(8, int(top_k) * 2),
        )
        existing = {
            (chunk.get("source", ""), chunk.get("text", ""))
            for chunk in self._normalize_chunks(chunks)
        }
        for chunk in detail_chunks:
            normalized = self._normalize_chunks([chunk])
            if not self._extract_structured_milestones(normalized):
                continue
            key = (normalized[0]["source"], normalized[0]["text"])
            if key not in existing:
                chunks.append(chunk)
                existing.add(key)
        return chunks

    def _extract_with_llm(self, retrieved_chunks: Sequence[Dict[str, Any]]) -> ScopeExtractionOutput:
        """Construct context and generate structured output via GeminiProvider."""
        normalized_chunks = self._normalize_chunks(retrieved_chunks)
        
        # Build concise context block including chunk_id and source metadata
        context_lines = []
        unique_sources = set()

        for idx, chunk in enumerate(normalized_chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source") or chunk.get("source") or "Unknown Document"
            doc_id = meta.get("document_id", "N/A")
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            page = meta.get("page", "N/A")

            unique_sources.add(src)

            context_lines.append(
                f"--- CHUNK {idx} | Source: {src} | Page: {page} | ChunkID: {chunk_id} | DocID: {doc_id} ---\n"
                f"{chunk.get('text', '')}\n"
            )

        context_text = "\n".join(context_lines)
        prompt = SCOPE_PROMPT.format(context=context_text)

        system_instruction = (
            "You are an expert AI Scope and Deliverable Extraction Agent for software projects. "
            "Your task is to analyze document context and extract project goals, deliverables, milestones, "
            "timelines, and responsibilities into a structured JSON schema. "
            "GROUNDING RULE: Extract ONLY information explicitly supported by the text context. "
            "If any field or item is not mentioned, use exact text: 'Not specified in the available project documents.' "
            "Do NOT invent facts, dates, or team members."
        )

        # Call Gemini Provider with structured response schema
        response_dict = self.provider.generate_structured(
            prompt=prompt,
            response_schema=ScopeExtractionOutput,
            system_instruction=system_instruction
        )

        # Ensure model validation and sanitize missing fields
        parsed_output = self._validate_model(ScopeExtractionOutput, response_dict)
        
        # Fill in default sources if omitted
        if not parsed_output.sources and unique_sources:
            parsed_output.sources = self._deduplicate(list(unique_sources))

        known_responsibilities = {
            (item.person.strip().casefold(), item.responsibility.strip().casefold())
            for item in parsed_output.responsibilities
        }
        for item in self._extract_structured_responsibilities(normalized_chunks):
            key = (item.person.casefold(), item.responsibility.casefold())
            if key not in known_responsibilities:
                parsed_output.responsibilities.append(item)
                known_responsibilities.add(key)

        known_milestones = {
            (item.name.strip().casefold(), (item.target_date or "").strip().casefold())
            for item in parsed_output.milestones
        }
        for item in self._extract_structured_milestones(normalized_chunks):
            key = (item.name.casefold(), (item.target_date or "").casefold())
            if key not in known_milestones:
                parsed_output.milestones.append(item)
                known_milestones.add(key)

        return self._apply_grounding_sanitization(parsed_output)

    @staticmethod
    def _extract_structured_responsibilities(
        chunks: Sequence[Dict[str, Any]],
    ) -> List[ResponsibilityItem]:
        """Recover explicit task/assignee pairs from retrieved table rows."""
        responsibilities = []
        seen = set()
        task_labels = {"task", "task_name", "work_item", "activity", "responsibility"}
        person_labels = {"assignee", "owner", "assigned_to", "responsible_person"}

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source = metadata.get("source") or chunk.get("source") or "Unknown Document"
            for line in chunk.get("text", "").splitlines():
                evidence = line.strip()
                line = re.sub(r"^\s*Record\s+\d+:\s*", "", line)
                fields = {}
                for label, value in re.findall(r"(?:^|\|)\s*([^:|]+):\s*([^|]+)", line):
                    normalized_label = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
                    fields[normalized_label] = value.strip()

                task = next((fields[label] for label in task_labels if fields.get(label)), None)
                person = next((fields[label] for label in person_labels if fields.get(label)), None)
                if not task or not person:
                    continue

                key = (person.casefold(), task.casefold())
                if key in seen:
                    continue
                seen.add(key)
                page = metadata.get("page")
                responsibilities.append(ResponsibilityItem(
                    person=person,
                    responsibility=task,
                    source=str(source),
                    document_name=str(source),
                    document_id=metadata.get("document_id"),
                    page=page if isinstance(page, int) else None,
                    chunk_id=metadata.get("chunk_id"),
                    evidence=evidence,
                ))

        return responsibilities

    @staticmethod
    def _extract_structured_milestones(
        chunks: Sequence[Dict[str, Any]],
    ) -> List[MilestoneItem]:
        """Recover explicit milestone and sprint fields from retrieved table rows."""
        milestones = []
        seen = set()
        milestone_labels = {
            "planned_sprint", "actual_sprint", "milestone", "phase", "release", "iteration"
        }
        date_labels = {"target_date", "deadline", "due_date", "delivery_date"}

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source = metadata.get("source") or chunk.get("source") or "Unknown Document"
            text = chunk.get("text", "")
            rows = re.split(r"\bRow\s+\d+:\s*", text)
            if len(rows) == 1:
                rows = text.splitlines()

            for row in rows:
                fields = {}
                for label, value in re.findall(r"(?:^|\|)\s*([^:|]+):\s*([^|]+)", row):
                    normalized_label = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
                    fields[normalized_label] = value.strip()

                label = next((name for name in milestone_labels if fields.get(name)), None)
                if not label:
                    continue

                name = fields[label]
                target_date = next((fields[name] for name in date_labels if fields.get(name)), None)
                key = (name.casefold(), (target_date or "").casefold())
                if key in seen:
                    continue
                seen.add(key)
                page = metadata.get("page")
                milestones.append(MilestoneItem(
                    name=name,
                    target_date=target_date,
                    source=str(source),
                    document_name=str(source),
                    document_id=metadata.get("document_id"),
                    page=page if isinstance(page, int) else None,
                    chunk_id=metadata.get("chunk_id"),
                    evidence=row.strip(),
                ))

        return milestones

    def _apply_grounding_sanitization(self, output: ScopeExtractionOutput) -> ScopeExtractionOutput:
        """Enforce strict grounding rules across all extracted items."""
        for g in output.project_goals:
            if not g.source or g.source.strip() == "":
                g.source = FALLBACK_NOT_SPECIFIED
            if not g.evidence or g.evidence.strip() == "":
                g.evidence = FALLBACK_NOT_SPECIFIED

        for d in output.deliverables:
            if not d.source or d.source.strip() == "":
                d.source = FALLBACK_NOT_SPECIFIED
            if not d.evidence or d.evidence.strip() == "":
                d.evidence = FALLBACK_NOT_SPECIFIED

        for m in output.milestones:
            if not m.source or m.source.strip() == "":
                m.source = FALLBACK_NOT_SPECIFIED
            if not m.evidence or m.evidence.strip() == "":
                m.evidence = FALLBACK_NOT_SPECIFIED

        for t in output.timeline:
            if not t.source or t.source.strip() == "":
                t.source = FALLBACK_NOT_SPECIFIED
            if not t.evidence or t.evidence.strip() == "":
                t.evidence = FALLBACK_NOT_SPECIFIED

        for r in output.responsibilities:
            if not r.source or r.source.strip() == "":
                r.source = FALLBACK_NOT_SPECIFIED
            if not r.evidence or r.evidence.strip() == "":
                r.evidence = FALLBACK_NOT_SPECIFIED

        return output

    def _empty_scope_output(self) -> ScopeExtractionOutput:
        """Return an empty grounded ScopeExtractionOutput instance."""
        return ScopeExtractionOutput(
            project_goals=[],
            deliverables=[],
            milestones=[],
            timeline=[],
            responsibilities=[],
            sources=[]
        )

    def analyze_context(
        self,
        retrieved_chunks: Sequence[Dict[str, Any]] | None,
        project_name: str | None = None
    ) -> ScopeExtractionOutput:
        """
        Backwards-compatible interface method for orchestrator usage.
        """
        if not retrieved_chunks:
            return self._empty_scope_output()

        if self.provider is not None:
            try:
                return self._extract_with_llm(retrieved_chunks)
            except Exception as e:
                logger.warning("LLM extraction failed in analyze_context, falling back: %s", e)

        # Basic rule-based extraction fallback for context analysis
        chunks = self._normalize_chunks(retrieved_chunks)
        goals, deliverables, milestones, timeline, responsibilities = [], [], [], [], []
        sources = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            source = self._extract_source(chunk)
            sources.add(source)

            if "objective" in text.lower() or "goal" in text.lower():
                goals.append(ProjectGoal(goal=text[:150], source=source, evidence=text[:150]))
            if "deliverable" in text.lower() or "feature" in text.lower():
                deliverables.append(DeliverableItem(name=text[:80], description=text[:150], source=source, evidence=text[:150]))
            if "milestone" in text.lower() or "deadline" in text.lower():
                milestones.append(MilestoneItem(name=text[:80], source=source, evidence=text[:150]))

        return ScopeExtractionOutput(
            project_goals=goals,
            deliverables=deliverables,
            milestones=milestones,
            timeline=timeline,
            responsibilities=responsibilities,
            sources=list(sources)
        )

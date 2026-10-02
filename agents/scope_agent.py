"""
Scope and Deliverable Extraction Agent.

Extracts project goals, deliverables, milestones, timelines, and responsibilities
from uploaded project documents using grounded RAG retrieval and Google Gemini API.
"""

import logging
import os
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
        "project goals objectives scope expected outcomes",
        "project deliverables features modules systems reports APIs documentation",
        "milestones timeline deadlines target dates start date end date",
        "responsibilities assignments person team task role owner",
    ]

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
        """Retrieve the most relevant chunks, capped by the configured depth."""
        return self._search_targeted_chunks(
            self.retriever, project_id, top_k, self.TARGET_QUERIES
        )

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

        return self._apply_grounding_sanitization(parsed_output)

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

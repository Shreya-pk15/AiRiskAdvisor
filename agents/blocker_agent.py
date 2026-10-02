"""
Blocker and Action Item Identification Agent.

Extracts blockers, pending decisions, unresolved issues, and action items from uploaded
project documents using grounded RAG retrieval and Google Gemini API.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from agents.prompts import BLOCKER_PROMPT
from agents.schemas import (
    ActionItem,
    BlockerActionOutput,
    BlockerItem,
    PendingDecision,
    UnresolvedIssue,
)
from llm.base_provider import BaseLLMProvider
from llm.gemini_provider import GeminiProvider
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

FALLBACK_NOT_SPECIFIED = "Not specified in the available project documents."


class BlockerActionAgent(BaseAgent):
    """
    Blocker & Action Item Identification Agent.

    Uses RAG retrieval with target queries filtered by project_id and sends context
    to Gemini to produce a structured Pydantic BlockerActionOutput.
    """

    TARGET_QUERIES = [
        "meeting blockers unresolved issues",
        "pending decisions technology approval clarification",
        "action items assigned tasks owner deadline",
        "sprint progress issues defect tracker updates",
        "tasks waiting for people teams approval dependency",
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
        Execute full Blocker & Action Item identification workflow for a given project_id.

        Args:
            project_id (str): Project identifier / workspace name.
            top_k (int): Maximum number of distinct context chunks across targeted queries.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "data": BlockerActionOutput dictionary
                - "metadata": Execution metadata (agent_name, provider, execution_time_seconds, status, error)
        """
        start_time = time.time()
        agent_name = "Blocker & Action Agent"
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
                        "data": self._empty_blocker_output().model_dump(),
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
                    blocker_model = self.model_name or os.getenv("GEMINI_BLOCKER_MODEL") or os.getenv("GEMINI_SCOPE_MODEL") or "gemini-3.6-flash"
                    self.provider = GeminiProvider(model_name=blocker_model)
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
                                "data": self._empty_blocker_output().model_dump(),
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
                            "data": self._empty_blocker_output().model_dump(),
                            "metadata": {
                                "agent_name": agent_name,
                                "provider": provider_name,
                                "execution_time_seconds": exec_time,
                                "status": "Failed",
                                "error": f"Failed to initialize Gemini Provider: {e}"
                            }
                        }

        try:
            # 1. RAG Retrieval filtered by project_id
            retrieved_chunks = self._retrieve_targeted_chunks(project_id=project_id, top_k=top_k)

            if not retrieved_chunks:
                exec_time = round(time.time() - start_time, 2)
                empty_res = self._empty_blocker_output()
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

            # 2. Extract structured output via single Gemini call
            blocker_output = self._extract_with_llm(retrieved_chunks)
            exec_time = round(time.time() - start_time, 2)

            return {
                "data": blocker_output.model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "provider": provider_name,
                    "execution_time_seconds": exec_time,
                    "status": "Success",
                    "error": None
                }
            }

        except Exception as exc:
            logger.error("Blocker and action agent error: %s", exc, exc_info=True)
            exec_time = round(time.time() - start_time, 2)
            return {
                "data": self._empty_blocker_output().model_dump(),
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

    def _extract_with_llm(self, retrieved_chunks: Sequence[Dict[str, Any]]) -> BlockerActionOutput:
        """Construct context and generate structured output via GeminiProvider."""
        normalized_chunks = self._normalize_chunks(retrieved_chunks)

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
        prompt = BLOCKER_PROMPT.format(context=context_text)

        system_instruction = (
            "You are an expert AI Blocker and Action Item Identification Agent for software projects. "
            "Analyze project context strictly using grounded document evidence. "
            "Extract blockers, pending decisions, unresolved issues, and action items. "
            "GROUNDING RULE: Do NOT invent assignees, deadlines, priorities, blockers, decisions, or issue details. "
            "If an attribute is not present in the documents, use exact text: 'Not specified in the available project documents.'"
        )

        response_dict = self.provider.generate_structured(
            prompt=prompt,
            response_schema=BlockerActionOutput,
            system_instruction=system_instruction
        )

        parsed_output = self._validate_model(BlockerActionOutput, response_dict)

        if not parsed_output.sources and unique_sources:
            parsed_output.sources = self._deduplicate(list(unique_sources))

        return self._apply_grounding_sanitization(parsed_output)

    def _apply_grounding_sanitization(self, output: BlockerActionOutput) -> BlockerActionOutput:
        """Enforce strict grounding rules and fallbacks across extracted items."""
        for b in output.blockers:
            if not b.source or b.source.strip() == "":
                b.source = FALLBACK_NOT_SPECIFIED
            if not b.evidence or b.evidence.strip() == "":
                b.evidence = FALLBACK_NOT_SPECIFIED
            if not b.impact or b.impact.strip() == "":
                b.impact = FALLBACK_NOT_SPECIFIED

        for d in output.pending_decisions:
            if not d.source or d.source.strip() == "":
                d.source = FALLBACK_NOT_SPECIFIED
            if not d.evidence or d.evidence.strip() == "":
                d.evidence = FALLBACK_NOT_SPECIFIED
            if not d.owner or d.owner.strip() == "":
                d.owner = FALLBACK_NOT_SPECIFIED

        for u in output.unresolved_issues:
            if not u.source or u.source.strip() == "":
                u.source = FALLBACK_NOT_SPECIFIED
            if not u.evidence or u.evidence.strip() == "":
                u.evidence = FALLBACK_NOT_SPECIFIED

        for a in output.action_items:
            if not a.source or a.source.strip() == "":
                a.source = FALLBACK_NOT_SPECIFIED
            if not a.evidence or a.evidence.strip() == "":
                a.evidence = FALLBACK_NOT_SPECIFIED
            if not a.assignee or a.assignee.strip() == "":
                a.assignee = FALLBACK_NOT_SPECIFIED
            if not a.deadline or a.deadline.strip() == "":
                a.deadline = FALLBACK_NOT_SPECIFIED
            if not a.priority or a.priority.strip() == "":
                a.priority = FALLBACK_NOT_SPECIFIED

        return output

    def _empty_blocker_output(self) -> BlockerActionOutput:
        """Return an empty grounded BlockerActionOutput instance."""
        return BlockerActionOutput(
            blockers=[],
            pending_decisions=[],
            unresolved_issues=[],
            action_items=[],
            sources=[]
        )

    def analyze_context(
        self,
        retrieved_chunks: Sequence[Dict[str, Any]] | None,
        project_name: str | None = None
    ) -> BlockerActionOutput:
        """
        Backwards-compatible interface for orchestrator usage.
        """
        if not retrieved_chunks:
            return self._empty_blocker_output()

        if self.provider is not None:
            try:
                return self._extract_with_llm(retrieved_chunks)
            except Exception as e:
                logger.warning("LLM extraction failed in analyze_context, falling back: %s", e)

        # Rule-based fallback for context analysis
        chunks = self._normalize_chunks(retrieved_chunks)
        blockers, decisions, issues, action_items = [], [], [], []
        sources = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            source = self._extract_source(chunk)
            sources.add(source)

            lower = text.lower()
            if "block" in lower or "waiting" in lower:
                blockers.append(BlockerItem(description=text[:150], source=source, evidence=text[:150]))
            if "decision" in lower or "pending" in lower:
                decisions.append(PendingDecision(decision=text[:150], source=source, evidence=text[:150]))
            if "issue" in lower or "defect" in lower or "bug" in lower:
                issues.append(UnresolvedIssue(issue=text[:150], source=source, evidence=text[:150]))
            if "action" in lower or "todo" in lower or "task" in lower:
                action_items.append(ActionItem(action=text[:150], source=source, evidence=text[:150]))

        return BlockerActionOutput(
            blockers=blockers,
            pending_decisions=decisions,
            unresolved_issues=issues,
            action_items=action_items,
            sources=list(sources)
        )

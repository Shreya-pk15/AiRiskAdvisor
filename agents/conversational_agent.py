"""
Conversational Project Intelligence Assistant.

Milestone 3 — Part 3:
Allows natural-language interactive Q&A grounded in uploaded project artifacts
and existing Milestone 2 & 3 structured intelligence (Scope, Risks, Blockers, Health Score).
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from agents.prompts import CONVERSATIONAL_ASSISTANT_PROMPT
from agents.schemas import ChatMessage, ConversationalResponse
from llm.base_provider import BaseLLMProvider
from llm.gemini_provider import GeminiProvider
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

FALLBACK_NO_INFO = "I could not find enough information about this in the uploaded project documents."
MAX_RETRIEVAL_DISTANCE = 1.75


class ConversationalProjectAssistant(BaseAgent):
    """
    Conversational Project Intelligence Assistant.

    Provides grounded multi-turn conversational intelligence using ChromaDB retrieval
    and existing Scope, Risk, Blocker, and Health Score outputs.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        retriever: Optional[Retriever] = None,
        model_name: Optional[str] = None,
        max_history_turns: int = 6,
    ):
        super().__init__(model_name=model_name)
        self.provider = provider
        self.retriever = retriever or Retriever()
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, Any]] = []

    def _ensure_provider(self) -> BaseLLMProvider:
        """Lazily initialize provider if not already supplied."""
        if self.provider is not None:
            return self.provider

        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key or gemini_key.startswith("AQ.") or gemini_key == "your_gemini_api_key_here":
            if os.getenv("GROQ_API_KEY"):
                from llm.groq_provider import GroqProvider
                groq_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-120b")
                self.provider = GroqProvider(model_name=groq_model)
                return self.provider

        model = (
            self.model_name
            or os.getenv("GEMINI_CHAT_MODEL")
            or os.getenv("GEMINI_SCOPE_MODEL")
            or os.getenv("GEMINI_MODEL_NAME")
            or "gemini-3.8-flash"
        )
        try:
            self.provider = GeminiProvider(model_name=model)
            return self.provider
        except Exception as e:
            if os.getenv("GROQ_API_KEY"):
                from llm.groq_provider import GroqProvider
                groq_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-120b")
                self.provider = GroqProvider(model_name=groq_model)
                return self.provider
            raise RuntimeError(f"Failed to initialize LLM Provider for Assistant: {e}") from e

    def ask(
        self,
        question: str,
        project_id: str,
        scope_data: Optional[Dict[str, Any]] = None,
        risk_data: Optional[Dict[str, Any]] = None,
        blocker_data: Optional[Dict[str, Any]] = None,
        health_data: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> ConversationalResponse:
        """
        Answer user question strictly grounded in project documents and structured intelligence.

        Args:
            question (str): User's natural-language inquiry.
            project_id (str): Project workspace identifier for ChromaDB isolation.
            scope_data (Optional[Dict]): Extracted Scope & Deliverables output.
            risk_data (Optional[Dict]): Extracted Risk Detection output.
            blocker_data (Optional[Dict]): Extracted Blocker & Action Items output.
            health_data (Optional[Dict]): Calculated Project Health output.
            top_k (int): Number of context chunks to retrieve.

        Returns:
            ConversationalResponse: Pydantic-validated answer, sources, and metadata.
        """
        if not project_id or not project_id.strip():
            return ConversationalResponse(
                answer="Please select or specify a project workspace before asking questions.",
                sources=[],
                chunks_used=[],
                project_id="",
                status="Failed",
                error="Missing project_id",
            )

        clean_question = question.strip()
        if not clean_question:
            return ConversationalResponse(
                answer="Please enter a valid project question.",
                sources=[],
                chunks_used=[],
                project_id=project_id,
                status="Failed",
                error="Empty question",
            )

        # 1. RAG Retrieval filtered by project_id
        # Build search query: if follow-up with pronouns, augment with last user question
        search_query = clean_question
        if len(self.history) >= 2 and any(pronoun in clean_question.lower() for pronoun in ["they", "them", "these", "those", "this", "it", "who"]):
            last_user_msg = next((h["content"] for h in reversed(self.history) if h.get("role") == "user"), "")
            if last_user_msg:
                search_query = f"{last_user_msg} {clean_question}"

        retrieved_chunks = []
        try:
            retrieved_chunks = self.retriever.retrieve(
                query=search_query,
                project_name=project_id,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning("Retrieval failed for project %s: %s", project_id, e)

        retrieval_distances = [
            float(chunk["distance"])
            for chunk in retrieved_chunks
            if chunk.get("distance") is not None
        ]
        if retrieval_distances and min(retrieval_distances) > MAX_RETRIEVAL_DISTANCE:
            return ConversationalResponse(
                answer=FALLBACK_NO_INFO,
                sources=[],
                chunks_used=[],
                project_id=project_id,
                status="Success",
            )

        # Extract citation sources
        sources = set()
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            src = meta.get("source") or c.get("source") or "Unknown Document"
            page = meta.get("page")
            if page and str(page).lower() != "n/a" and str(page) != "None":
                sources.add(f"{src}, Page {page}")
            else:
                sources.add(str(src))

        # 2. Build Structured Project Intelligence block
        intelligence_blocks = []

        # (a) Project Health Score
        if health_data:
            score = health_data.get("overall_score")
            cls_name = health_data.get("classification")
            conf = health_data.get("confidence")
            factors = health_data.get("key_factors", [])
            intelligence_blocks.append(
                f"PROJECT HEALTH SCORE: {score}/100 | Classification: {cls_name} | Confidence: {conf}\n"
                f"Key Health Factors:\n" + "\n".join(f"- {f}" for f in factors[:4])
            )

        # (b) Risk Agent Summary
        if risk_data:
            deliv_status = risk_data.get("delivery_status", "Insufficient Data")
            risks_list = risk_data.get("risks", [])
            risk_lines = [
                f"- [{r.get('severity', 'Medium')}] {r.get('description', '')} (Mitigation: {r.get('recommended_action', 'N/A')})"
                for r in risks_list[:5]
            ]
            intelligence_blocks.append(
                f"DELIVERY STATUS: {deliv_status}\n"
                f"IDENTIFIED RISKS ({len(risks_list)}):\n" + "\n".join(risk_lines)
            )

        # (c) Blocker Agent Summary
        if blocker_data:
            blockers = blocker_data.get("blockers", [])
            issues = blocker_data.get("unresolved_issues", [])
            actions = blocker_data.get("action_items", [])
            decisions = blocker_data.get("pending_decisions", [])

            blocker_lines = [f"- {b.get('description', '')} (Status: {b.get('status', 'Active')})" for b in blockers[:4]]
            issue_lines = [f"- {i.get('issue', '')}" for i in issues[:3]]
            action_lines = [f"- {a.get('action', '')} (Assignee: {a.get('assignee', 'Not specified')})" for a in actions[:4]]

            intelligence_blocks.append(
                f"ACTIVE BLOCKERS ({len(blockers)}):\n" + ("\n".join(blocker_lines) if blocker_lines else "None recorded.\n") +
                f"UNRESOLVED ISSUES ({len(issues)}):\n" + ("\n".join(issue_lines) if issue_lines else "None recorded.\n") +
                f"ACTION ITEMS ({len(actions)}):\n" + ("\n".join(action_lines) if action_lines else "None recorded.\n") +
                f"PENDING DECISIONS ({len(decisions)})."
            )

        # (d) Scope Agent Summary
        if scope_data:
            goals = [g.get("goal", "") for g in scope_data.get("project_goals", [])]
            delivs = [d.get("name", "") for d in scope_data.get("deliverables", [])]
            milestones = [f"{m.get('name', '')} (Target: {m.get('target_date', 'N/A')})" for m in scope_data.get("milestones", [])]
            intelligence_blocks.append(
                f"PROJECT GOALS: {', '.join(goals[:3]) if goals else 'Not specified'}\n"
                f"DELIVERABLES: {', '.join(delivs[:5]) if delivs else 'Not specified'}\n"
                f"MILESTONES: {', '.join(milestones[:4]) if milestones else 'Not specified'}"
            )

        project_intelligence_text = "\n\n".join(intelligence_blocks) if intelligence_blocks else "No prior agent outputs recorded."

        # 3. Build Retrieved Document Context block
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source") or chunk.get("source") or "Unknown"
            page = meta.get("page", "N/A")
            context_blocks.append(f"[Chunk {idx} | Source: {src}, Page: {page}]\n{chunk.get('text', '')}")

        retrieved_context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant document chunks retrieved."

        # 4. Build Conversation History text
        recent_history = self.history[-(self.max_history_turns * 2):]
        history_lines = []
        for turn in recent_history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            history_lines.append(f"{role}: {content}")
        conversation_history_text = "\n".join(history_lines) if history_lines else "No previous conversation."

        # Handle total absence of context
        if not retrieved_chunks and not intelligence_blocks:
            return ConversationalResponse(
                answer=FALLBACK_NO_INFO,
                sources=[],
                chunks_used=[],
                project_id=project_id,
                status="Success",
            )

        # 5. Build Prompt
        prompt = CONVERSATIONAL_ASSISTANT_PROMPT.format(
            conversation_history=conversation_history_text,
            project_intelligence=project_intelligence_text,
            retrieved_context=retrieved_context_text,
            question=clean_question,
        )

        system_instruction = (
            "You are an expert AI Project Intelligence Assistant for software development teams. "
            "Your answers must be strictly grounded in the provided project documents and agent outputs. "
            "Never invent facts, tasks, or dates. If information is not in the context, reply: "
            "'I could not find enough information about this in the uploaded project documents.'"
        )

        # 6. Call LLM
        try:
            provider = self._ensure_provider()
            raw_answer = provider.generate_text(prompt=prompt, system_instruction=system_instruction)
            answer_text = raw_answer.strip() if raw_answer else FALLBACK_NO_INFO
        except Exception as exc:
            logger.error("Conversational assistant LLM generation failed: %s", exc, exc_info=True)
            # Offline grounded fallback if API call fails
            answer_text = self._fallback_grounded_answer(clean_question, retrieved_chunks, project_intelligence_text)
            if not answer_text:
                return ConversationalResponse(
                    answer=FALLBACK_NO_INFO,
                    sources=sorted(list(sources)),
                    chunks_used=retrieved_chunks,
                    project_id=project_id,
                    status="Failed",
                    error=str(exc),
                )

        # Clean source formatting
        final_sources = sorted(list(sources))

        # Update History
        self.history.append({"role": "user", "content": clean_question})
        self.history.append({"role": "assistant", "content": answer_text, "sources": final_sources})

        # Trim History
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]

        return ConversationalResponse(
            answer=answer_text,
            sources=final_sources,
            chunks_used=retrieved_chunks,
            project_id=project_id,
            status="Success",
            error=None,
        )

    def _fallback_grounded_answer(
        self, question: str, retrieved_chunks: List[Dict[str, Any]], intelligence_text: str
    ) -> str:
        """Deterministic fallback when LLM API is unavailable."""
        q_lower = question.lower()
        if "track" in q_lower or "health" in q_lower:
            if "PROJECT HEALTH SCORE:" in intelligence_text:
                lines = [line for line in intelligence_text.split("\n") if "HEALTH" in line or "DELIVERY STATUS" in line]
                return "\n".join(lines)

        if "risk" in q_lower:
            if "IDENTIFIED RISKS" in intelligence_text:
                return intelligence_text

        if "block" in q_lower:
            if "ACTIVE BLOCKERS" in intelligence_text:
                return intelligence_text

        if retrieved_chunks:
            top_snippet = retrieved_chunks[0].get("text", "")[:300]
            src = retrieved_chunks[0].get("metadata", {}).get("source", "Project Document")
            return f"{top_snippet}\n\n[Source: {src}]"

        return FALLBACK_NO_INFO

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.history = []

    def get_history(self) -> List[Dict[str, Any]]:
        """Return shallow copy of conversation history."""
        return list(self.history)

    def set_history(self, history: List[Dict[str, Any]]) -> None:
        """Set or restore conversation history."""
        self.history = list(history)

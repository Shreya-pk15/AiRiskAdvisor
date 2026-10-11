"""
Blocker and Action Item Identification Agent.

Extracts blockers, pending decisions, unresolved issues, and action items from uploaded
project documents using grounded RAG retrieval and Google Gemini API.
"""

import logging
import os
import re
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
        "task records owner due date priority status",
        "tasks backlog sprint deliverables assignments",
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
        provider_name = "Rule-Based"

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
                    groq_model = os.getenv("GROQ_RISK_MODEL", "qwen/qwen3.8-27b")
                    self.provider = GroqProvider(model_name=groq_model)
                    provider_name = "Groq"
                except Exception as e:
                    logger.warning("Groq provider init failed, will use rule-based fallback: %s", e)
            else:
                try:
                    blocker_model = self.model_name or os.getenv("GEMINI_BLOCKER_MODEL") or os.getenv("GEMINI_SCOPE_MODEL") or "gemini-1.5-flash"
                    self.provider = GeminiProvider(model_name=blocker_model)
                    provider_name = "Gemini"
                except Exception as e:
                    # Last resort: try Groq
                    if os.getenv("GROQ_API_KEY"):
                        try:
                            from llm.groq_provider import GroqProvider
                            groq_model = os.getenv("GROQ_RISK_MODEL", "qwen/qwen3.8-27b")
                            self.provider = GroqProvider(model_name=groq_model)
                            provider_name = "Groq"
                        except Exception as e2:
                            logger.warning("All providers failed to init, will use rule-based fallback: %s", e2)
                    else:
                        logger.warning("No LLM provider available, will use rule-based fallback: %s", e)

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

            # 2. Try LLM extraction first, fall back to rule-based on any failure
            blocker_output = None
            llm_error = None
            if self.provider is not None:
                try:
                    blocker_output = self._extract_with_llm(retrieved_chunks)
                except Exception as llm_exc:
                    llm_error = str(llm_exc)
                    logger.warning("LLM extraction failed, falling back to rule-based: %s", llm_exc)

            # Rule-based fallback if LLM failed or unavailable
            if blocker_output is None:
                provider_name = "Rule-Based"
                blocker_output = self.analyze_context(retrieved_chunks)
            else:
                # If LLM succeeded but missed structured action items present in chunks, supplement them
                rule_fallback = self.analyze_context(retrieved_chunks)
                if not blocker_output.action_items and rule_fallback.action_items:
                    blocker_output.action_items = rule_fallback.action_items
                if not blocker_output.blockers and rule_fallback.blockers:
                    blocker_output.blockers = rule_fallback.blockers

            exec_time = round(time.time() - start_time, 2)
            note = f"LLM unavailable, used rule-based extraction. LLM error: {llm_error}" if llm_error else None

            result_meta = {
                "agent_name": agent_name,
                "provider": provider_name,
                "execution_time_seconds": exec_time,
                "status": "Success",
                "error": None
            }
            if note:
                result_meta["note"] = note

            return {
                "data": blocker_output.model_dump(),
                "metadata": result_meta
            }

        except Exception as exc:
            logger.error("Blocker and action agent error: %s", exc, exc_info=True)
            exec_time = round(time.time() - start_time, 2)
            # Even on total failure, return Success with empty data so UI shows gracefully
            return {
                "data": self._empty_blocker_output().model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "provider": provider_name,
                    "execution_time_seconds": exec_time,
                    "status": "Success",
                    "error": None,
                    "note": f"Analysis could not be completed: {exc}"
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
        context_text, unique_sources = self._build_budgeted_context(normalized_chunks, max_chars=9000)
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
        blockers: List[BlockerItem] = []
        decisions: List[PendingDecision] = []
        issues: List[UnresolvedIssue] = []
        action_items: List[ActionItem] = []
        sources = set()
        act_counter = 1

        for chunk in chunks:
            text = chunk.get("text", "")
            source = self._extract_source(chunk)
            sources.add(source)

            # Normalize text: place inline 'Record \d+:' onto newlines
            normalized_text = re.sub(r'(?i)\s+(Record\s*\d+:)', r'\n\1', text)
            lines = [l.strip() for l in normalized_text.split('\n') if l.strip()]

            for line in lines:
                lower = line.lower()

                # Case 1: Pipe-delimited structured record (e.g. 'Record 1: task: ... | owner: ... | due_date: ...')
                if "|" in line and any(k in lower for k in ["task", "owner", "due", "status", "priority", "assignee"]):
                    cleaned_line = re.sub(r'(?i)^Record\s*\d+:\s*', '', line).strip()
                    parts = [p.strip() for p in cleaned_line.split("|")]
                    fields = {}
                    for p in parts:
                        if ":" in p:
                            k, v = p.split(":", 1)
                            fields[k.strip().lower().replace(" ", "_")] = v.strip()

                    task_desc = (
                        fields.get("task")
                        or fields.get("task_name")
                        or fields.get("action")
                        or fields.get("action_item")
                        or fields.get("item")
                        or fields.get("title")
                    )
                    if task_desc:
                        assignee = (
                            fields.get("owner")
                            or fields.get("assignee")
                            or fields.get("assigned_to")
                            or fields.get("lead")
                            or FALLBACK_NOT_SPECIFIED
                        )
                        deadline = (
                            fields.get("due_date")
                            or fields.get("deadline")
                            or fields.get("target_date")
                            or fields.get("due")
                            or FALLBACK_NOT_SPECIFIED
                        )
                        status = fields.get("status") or fields.get("state") or "Open"
                        priority = fields.get("priority") or fields.get("severity") or "Medium"
                        notes = fields.get("notes") or fields.get("note") or fields.get("details") or ""
                        evidence = line[:250]

                        action_items.append(
                            ActionItem(
                                action_id=f"ACT-{act_counter:02d}",
                                action=task_desc,
                                assignee=assignee,
                                deadline=deadline,
                                status=status,
                                priority=priority,
                                evidence=evidence,
                                source=source
                            )
                        )
                        act_counter += 1

                        # If status or notes indicate a blocker, also record it
                        if "block" in status.lower() or "block" in notes.lower() or "waiting" in notes.lower():
                            blockers.append(
                                BlockerItem(
                                    title=f"Blocker on {task_desc}",
                                    description=f"{task_desc}: {notes}" if notes else f"{task_desc} is {status}",
                                    impact=f"Delivery risk for {task_desc}",
                                    status="Active",
                                    owner=assignee if assignee != FALLBACK_NOT_SPECIFIED else "Unassigned",
                                    evidence=evidence,
                                    source=source
                                )
                            )
                        continue

                # Case 2: Blocker detection
                if any(k in lower for k in ["blocker:", "blocked:", "is blocked", "currently blocked", "waiting for"]):
                    desc = re.sub(r'(?i)^(?:[0-9]+\.|\-|\*)\s*(?:blocker|blocked):\s*', '', line).strip()
                    blockers.append(
                        BlockerItem(
                            title=desc[:60],
                            description=desc,
                            impact="Schedule and delivery delay",
                            status="Active",
                            evidence=line[:250],
                            source=source
                        )
                    )

                # Case 3: Pending decision detection
                elif any(k in lower for k in ["decision:", "pending decision:", "decided:"]):
                    desc = re.sub(r'(?i)^(?:[0-9]+\.|\-|\*)\s*(?:decision|pending decision):\s*', '', line).strip()
                    decisions.append(
                        PendingDecision(
                            decision=desc,
                            status="Pending",
                            evidence=line[:250],
                            source=source
                        )
                    )

                # Case 4: Unresolved issue / bug detection
                elif any(k in lower for k in ["issue:", "defect:", "bug:", "impediment:"]):
                    desc = re.sub(r'(?i)^(?:[0-9]+\.|\-|\*)\s*(?:issue|defect|bug):\s*', '', line).strip()
                    issues.append(
                        UnresolvedIssue(
                            issue=desc,
                            status="Open",
                            evidence=line[:250],
                            source=source
                        )
                    )

                # Case 5: Explicit action item or task line
                elif any(k in lower for k in ["action item:", "action:", "todo:", "task:"]):
                    desc = re.sub(r'(?i)^(?:[0-9]+\.|\-|\*)\s*(?:action item|action|todo|task):\s*', '', line).strip()
                    action_items.append(
                        ActionItem(
                            action_id=f"ACT-{act_counter:02d}",
                            action=desc,
                            assignee=FALLBACK_NOT_SPECIFIED,
                            deadline=FALLBACK_NOT_SPECIFIED,
                            status="Open",
                            priority="Medium",
                            evidence=line[:250],
                            source=source
                        )
                    )
                    act_counter += 1

        return BlockerActionOutput(
            blockers=blockers,
            pending_decisions=decisions,
            unresolved_issues=issues,
            action_items=action_items,
            sources=list(sources)
        )

"""
Documentation Generation Agent.

Milestone 3 — Part 1:
Uses existing project knowledge base and Milestone 2 agent outputs (Scope, Risk, Blocker)
to generate missing project documentation:
1. User Stories (from project scope, deliverables, and milestones)
2. Risk Register (from identified risks and delivery forecasts)
3. Action Item List (from blockers, pending decisions, and action items)
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from agents.blocker_agent import BlockerActionAgent
from agents.prompts import (
    ACTION_ITEMS_DOC_PROMPT,
    RISK_REGISTER_PROMPT,
    USER_STORIES_PROMPT,
)
from agents.risk_agent import RiskDetectionAgent
from agents.schemas import (
    ActionItem,
    ActionItemsOutput,
    DocumentationGenerationResult,
    RiskRegisterItem,
    RiskRegisterOutput,
    UserStoriesOutput,
    UserStory,
)
from agents.scope_agent import ScopeExtractionAgent
from llm.base_provider import BaseLLMProvider
from llm.gemini_provider import GeminiProvider
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

FALLBACK_NOT_SPECIFIED = "Not specified in the available project documents."


class DocumentationAgent(BaseAgent):
    """
    Documentation Generation Agent.

    Reuses Milestone 2 outputs (Scope Agent, Risk Agent, Blocker Agent) to synthesize
    formal project documentation (User Stories, Risk Register, Action Item List)
    with strict source grounding and Pydantic validation.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        retriever: Optional[Retriever] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__(model_name=model_name)
        self.provider = provider
        self.retriever = retriever or Retriever()

    def _ensure_provider(self) -> BaseLLMProvider:
        """Lazily initialize provider if not already supplied."""
        if self.provider is not None:
            return self.provider

        # Check Groq first if Gemini API key is missing or invalid
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key or gemini_key.startswith("AQ.") or gemini_key == "your_gemini_api_key_here":
            if os.getenv("GROQ_API_KEY"):
                from llm.groq_provider import GroqProvider
                groq_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-120b")
                self.provider = GroqProvider(model_name=groq_model)
                return self.provider

        model = (
            self.model_name
            or os.getenv("GEMINI_DOCGEN_MODEL")
            or os.getenv("GEMINI_SCOPE_MODEL")
            or os.getenv("GEMINI_MODEL_NAME")
            or "gemini-3.8-flash"
        )
        try:
            self.provider = GeminiProvider(model_name=model)
            return self.provider
        except Exception as e:
            # Fall back to Groq if Gemini fails to initialize and Groq key exists
            if os.getenv("GROQ_API_KEY"):
                from llm.groq_provider import GroqProvider
                groq_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-120b")
                self.provider = GroqProvider(model_name=groq_model)
                return self.provider
            raise RuntimeError(f"Failed to initialize LLM Provider for Documentation Agent: {e}") from e

    def _execute_structured(
        self,
        prompt: str,
        response_schema: Any,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute structured generation with automatic Groq fallback on Gemini failure."""
        provider = self._ensure_provider()
        try:
            return provider.generate_structured(
                prompt=prompt,
                response_schema=response_schema,
                system_instruction=system_instruction,
            )
        except Exception as exc:
            logger.warning("Primary provider failed (%s); attempting fallback to Groq.", exc)
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                from llm.groq_provider import GroqProvider
                groq_model = os.getenv("GROQ_RISK_MODEL", "openai/gpt-oss-120b")
                groq_prov = GroqProvider(model_name=groq_model, api_key=groq_key)
                self.provider = groq_prov
                return groq_prov.generate_structured(
                    prompt=prompt,
                    response_schema=response_schema,
                    system_instruction=system_instruction,
                )
            raise exc

    # -------------------------------------------------------------------------
    # 1. User Stories Generation
    # -------------------------------------------------------------------------

    def generate_user_stories(
        self,
        project_id: str,
        scope_data: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate structured user stories from the project's actual scope and deliverables.

        Args:
            project_id (str): Project identifier / workspace name.
            scope_data (Optional[Dict[str, Any]]): Existing Scope Agent output to reuse.
            top_k (int): Top-K context chunks if Scope Agent must be executed.

        Returns:
            Dict[str, Any]: Structured dictionary with "data" (UserStoriesOutput) and "metadata".
        """
        start_time = time.time()
        agent_name = "Documentation Generator"
        doc_type = "User Stories"

        try:
            provider = self._ensure_provider()
            provider_name = type(provider).__name__.replace("Provider", "")
        except Exception as exc:
            return {
                "data": UserStoriesOutput().model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": "Unknown",
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Failed",
                    "error": str(exc),
                },
            }

        try:
            # 1. Reuse existing scope output or run Scope Agent once
            if not scope_data:
                scope_agent = ScopeExtractionAgent(
                    provider=self.provider, retriever=self.retriever, model_name=self.model_name
                )
                scope_res = scope_agent.run(project_id=project_id, top_k=top_k)
                scope_data = scope_res.get("data", {})

            goals = scope_data.get("project_goals", [])
            deliverables = scope_data.get("deliverables", [])
            milestones = scope_data.get("milestones", [])
            timeline = scope_data.get("timeline", [])

            if not goals and not deliverables:
                return {
                    "data": UserStoriesOutput().model_dump(),
                    "metadata": {
                        "agent_name": agent_name,
                        "document_type": doc_type,
                        "provider": provider_name,
                        "execution_time_seconds": round(time.time() - start_time, 2),
                        "status": "Success",
                        "error": None,
                        "note": "No scope goals or deliverables found in knowledge base.",
                    },
                }

            # 2. Build structured scope context for LLM prompt
            context_blocks = []
            sources = set(scope_data.get("sources", []))

            if goals:
                context_blocks.append("--- PROJECT GOALS ---")
                for g in goals:
                    src = g.get("source", FALLBACK_NOT_SPECIFIED)
                    sources.add(src)
                    context_blocks.append(
                        f"Goal: {g.get('goal', 'N/A')}\n"
                        f"Description: {g.get('description') or 'N/A'}\n"
                        f"Source: {src}\nEvidence: {g.get('evidence', FALLBACK_NOT_SPECIFIED)}"
                    )

            if deliverables:
                context_blocks.append("\n--- PROJECT DELIVERABLES ---")
                for d in deliverables:
                    src = d.get("source", FALLBACK_NOT_SPECIFIED)
                    sources.add(src)
                    context_blocks.append(
                        f"Deliverable: {d.get('name', 'N/A')}\n"
                        f"Description: {d.get('description') or 'N/A'}\n"
                        f"Source: {src}\nEvidence: {d.get('evidence', FALLBACK_NOT_SPECIFIED)}"
                    )

            if milestones:
                context_blocks.append("\n--- MILESTONES ---")
                for m in milestones:
                    src = m.get("source", FALLBACK_NOT_SPECIFIED)
                    sources.add(src)
                    context_blocks.append(
                        f"Milestone: {m.get('name', 'N/A')} (Target: {m.get('target_date', 'N/A')})\n"
                        f"Source: {src}"
                    )

            if timeline:
                context_blocks.append("\n--- TIMELINE ---")
                for t in timeline:
                    context_blocks.append(f"{t.get('label', 'Timeline')}: {t.get('value', 'N/A')}")

            context_text = "\n\n".join(context_blocks)
            prompt = USER_STORIES_PROMPT.format(context=context_text)

            system_instruction = (
                "You are an expert Agile Product Owner. Generate formal, structured User Stories "
                "strictly from the provided project scope and deliverables. "
                "GROUNDING RULE: Do NOT invent features or capabilities not explicitly mentioned or "
                "directly derived from the deliverables. Assign MoSCoW priorities ('Must Have', 'Should Have', "
                "'Could Have', 'Won't Have') grounded in the project goals."
            )

            # 3. Single LLM call with structured output validation (with Groq fallback)
            response_dict = self._execute_structured(
                prompt=prompt,
                response_schema=UserStoriesOutput,
                system_instruction=system_instruction,
            )

            parsed_output: UserStoriesOutput = self._validate_model(UserStoriesOutput, response_dict)  # type: ignore

            # 4. Strict grounding sanitization & sequential numbering
            cleaned_stories: List[UserStory] = []
            for idx, story in enumerate(parsed_output.user_stories, 1):
                story_id = story.story_id or f"US-{idx:02d}"
                if not story_id.startswith("US-"):
                    story_id = f"US-{idx:02d}"

                user_story_text = story.user_story.strip() if story.user_story else FALLBACK_NOT_SPECIFIED
                desc = story.description.strip() if story.description else FALLBACK_NOT_SPECIFIED
                prio = story.priority.strip() if story.priority else "Must Have"
                if prio not in {"Must Have", "Should Have", "Could Have", "Won't Have"}:
                    prio = "Must Have"

                deps = story.dependencies.strip() if story.dependencies else FALLBACK_NOT_SPECIFIED
                src = story.source.strip() if story.source else FALLBACK_NOT_SPECIFIED
                ev = story.evidence.strip() if story.evidence else FALLBACK_NOT_SPECIFIED

                ac = [c.strip() for c in story.acceptance_criteria if c and c.strip()]
                if not ac:
                    ac = [f"Deliverable '{story_id}' implemented as described."]

                cleaned_stories.append(
                    UserStory(
                        story_id=story_id,
                        user_story=user_story_text,
                        description=desc,
                        priority=prio,
                        acceptance_criteria=ac,
                        dependencies=deps,
                        source=src,
                        evidence=ev,
                    )
                )

            deduped_sources = self._deduplicate(list(sources.union(parsed_output.sources)))
            result_output = UserStoriesOutput(user_stories=cleaned_stories, sources=deduped_sources)

            return {
                "data": result_output.model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": provider_name,
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Success",
                    "error": None,
                },
            }

        except Exception as exc:
            logger.error("User story generation error: %s", exc, exc_info=True)
            return {
                "data": UserStoriesOutput().model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": provider_name if "provider_name" in locals() else "Unknown",
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Failed",
                    "error": str(exc),
                },
            }

    # -------------------------------------------------------------------------
    # 2. Risk Register Generation
    # -------------------------------------------------------------------------

    def generate_risk_register(
        self,
        project_id: str,
        risk_data: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate structured risk register from Risk Agent output and supporting evidence.

        Args:
            project_id (str): Project identifier / workspace name.
            risk_data (Optional[Dict[str, Any]]): Existing Risk Agent output to reuse.
            top_k (int): Top-K context chunks if Risk Agent must be executed.

        Returns:
            Dict[str, Any]: Structured dictionary with "data" (RiskRegisterOutput) and "metadata".
        """
        start_time = time.time()
        agent_name = "Documentation Generator"
        doc_type = "Risk Register"

        try:
            provider = self._ensure_provider()
            provider_name = type(provider).__name__.replace("Provider", "")
        except Exception as exc:
            return {
                "data": RiskRegisterOutput().model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": "Unknown",
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Failed",
                    "error": str(exc),
                },
            }

        try:
            # 1. Reuse existing risk output or run Risk Agent once
            if not risk_data or not risk_data.get("risks"):
                risk_agent = RiskDetectionAgent(
                    provider=self.provider, retriever=self.retriever, model_name=self.model_name
                )
                risk_res = risk_agent.run(project_id=project_id, top_k=top_k)
                risk_data = risk_res.get("data", {})

            risks = risk_data.get("risks", [])
            forecast = risk_data.get("delivery_forecast", {})

            if not risks:
                return {
                    "data": RiskRegisterOutput().model_dump(),
                    "metadata": {
                        "agent_name": agent_name,
                        "document_type": doc_type,
                        "provider": provider_name,
                        "execution_time_seconds": round(time.time() - start_time, 2),
                        "status": "Success",
                        "error": None,
                        "note": "No risks identified in project knowledge base.",
                    },
                }

            # 2. Build structured risk context for LLM prompt
            context_blocks = ["--- IDENTIFIED RISKS ---"]
            sources = set(risk_data.get("sources", []))

            for r in risks:
                src = r.get("source", FALLBACK_NOT_SPECIFIED)
                sources.add(src)
                context_blocks.append(
                    f"Risk ID: {r.get('risk_id', 'N/A')}\n"
                    f"Category: {r.get('category', 'Technical')}\n"
                    f"Description: {r.get('description', 'N/A')}\n"
                    f"Severity: {r.get('severity', 'Medium')}\n"
                    f"Recommended Action: {r.get('recommended_action', 'Not specified')}\n"
                    f"Evidence: {r.get('evidence', FALLBACK_NOT_SPECIFIED)}\n"
                    f"Source: {src}"
                )

            if forecast:
                context_blocks.append("\n--- DELIVERY FORECAST CONTEXT ---")
                if forecast.get("reason"):
                    context_blocks.append(f"Status Reason: {forecast['reason']}")
                if forecast.get("concerns"):
                    context_blocks.append(f"Documented Concerns: {', '.join(forecast['concerns'])}")

            context_text = "\n\n".join(context_blocks)
            prompt = RISK_REGISTER_PROMPT.format(context=context_text)

            system_instruction = (
                "You are an expert Project Risk Manager. Formulate a structured Risk Register from the "
                "provided risk evidence. "
                "GROUNDING RULE: Do NOT invent risk owners, probability ratings, or deadlines if they are "
                "not documented in the evidence. If unavailable, strictly output: "
                "'Not specified in the available project documents.'"
            )

            # 3. Single LLM call with structured output validation (with Groq fallback)
            response_dict = self._execute_structured(
                prompt=prompt,
                response_schema=RiskRegisterOutput,
                system_instruction=system_instruction,
            )

            parsed_output: RiskRegisterOutput = self._validate_model(RiskRegisterOutput, response_dict)  # type: ignore

            # 4. Strict grounding sanitization & sequential numbering
            cleaned_risks: List[RiskRegisterItem] = []
            for idx, item in enumerate(parsed_output.risk_register, 1):
                risk_id = item.risk_id or f"R-{idx:02d}"
                if not risk_id.startswith("R-"):
                    risk_id = f"R-{idx:02d}"

                desc = item.risk_description.strip() if item.risk_description else FALLBACK_NOT_SPECIFIED
                cat = item.category.strip() if item.category else "Technical"
                prob = item.probability.strip() if item.probability else FALLBACK_NOT_SPECIFIED
                imp = item.impact.strip() if item.impact else FALLBACK_NOT_SPECIFIED
                sev = item.severity.strip() if item.severity in {"Low", "Medium", "High"} else "Medium"
                mit = item.mitigation.strip() if item.mitigation else FALLBACK_NOT_SPECIFIED
                owner = item.owner.strip() if item.owner else FALLBACK_NOT_SPECIFIED
                status = item.status.strip() if item.status else "Open"
                src = item.source.strip() if item.source else FALLBACK_NOT_SPECIFIED
                ev = item.evidence.strip() if item.evidence else FALLBACK_NOT_SPECIFIED

                cleaned_risks.append(
                    RiskRegisterItem(
                        risk_id=risk_id,
                        risk_description=desc,
                        category=cat,
                        probability=prob,
                        impact=imp,
                        severity=sev,
                        mitigation=mit,
                        owner=owner,
                        status=status,
                        source=src,
                        evidence=ev,
                    )
                )

            deduped_sources = self._deduplicate(list(sources.union(parsed_output.sources)))
            result_output = RiskRegisterOutput(risk_register=cleaned_risks, sources=deduped_sources)

            return {
                "data": result_output.model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": provider_name,
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Success",
                    "error": None,
                },
            }

        except Exception as exc:
            logger.error("Risk register generation error: %s", exc, exc_info=True)
            return {
                "data": RiskRegisterOutput().model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": provider_name if "provider_name" in locals() else "Unknown",
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Failed",
                    "error": str(exc),
                },
            }

    # -------------------------------------------------------------------------
    # 3. Action Item List Generation
    # -------------------------------------------------------------------------

    def generate_action_items(
        self,
        project_id: str,
        blocker_data: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate structured action item list from Blocker Agent output and supporting evidence.

        Args:
            project_id (str): Project identifier / workspace name.
            blocker_data (Optional[Dict[str, Any]]): Existing Blocker Agent output to reuse.
            top_k (int): Top-K context chunks if Blocker Agent must be executed.

        Returns:
            Dict[str, Any]: Structured dictionary with "data" (ActionItemsOutput) and "metadata".
        """
        start_time = time.time()
        agent_name = "Documentation Generator"
        doc_type = "Action Items"

        try:
            provider = self._ensure_provider()
            provider_name = type(provider).__name__.replace("Provider", "")
        except Exception as exc:
            return {
                "data": ActionItemsOutput().model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": "Unknown",
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Failed",
                    "error": str(exc),
                },
            }

        try:
            # 1. Reuse existing blocker output or run Blocker Agent once
            if not blocker_data or not any(
                blocker_data.get(key)
                for key in ("action_items", "blockers", "pending_decisions", "unresolved_issues")
            ):
                blocker_agent = BlockerActionAgent(
                    provider=self.provider, retriever=self.retriever, model_name=self.model_name
                )
                blocker_res = blocker_agent.run(project_id=project_id, top_k=top_k)
                blocker_data = blocker_res.get("data", {})

            actions = blocker_data.get("action_items", [])
            blockers = blocker_data.get("blockers", [])
            decisions = blocker_data.get("pending_decisions", [])
            issues = blocker_data.get("unresolved_issues", [])

            if not actions and not blockers and not decisions and not issues:
                return {
                    "data": ActionItemsOutput().model_dump(),
                    "metadata": {
                        "agent_name": agent_name,
                        "document_type": doc_type,
                        "provider": provider_name,
                        "execution_time_seconds": round(time.time() - start_time, 2),
                        "status": "Success",
                        "error": None,
                        "note": "No action items, blockers, or issues found in knowledge base.",
                    },
                }

            # 2. Build structured blocker & action context for LLM prompt
            context_blocks = []
            sources = set(blocker_data.get("sources", []))

            if actions:
                context_blocks.append("--- DOCUMENTED ACTION ITEMS ---")
                for a in actions:
                    src = a.get("source", FALLBACK_NOT_SPECIFIED)
                    sources.add(src)
                    context_blocks.append(
                        f"Action: {a.get('action', 'N/A')}\n"
                        f"Assignee: {a.get('assignee', FALLBACK_NOT_SPECIFIED)}\n"
                        f"Deadline: {a.get('deadline', FALLBACK_NOT_SPECIFIED)}\n"
                        f"Priority: {a.get('priority', FALLBACK_NOT_SPECIFIED)}\n"
                        f"Status: {a.get('status', 'Open')}\n"
                        f"Source: {src}\nEvidence: {a.get('evidence', FALLBACK_NOT_SPECIFIED)}"
                    )

            if blockers:
                context_blocks.append("\n--- DOCUMENTED BLOCKERS ---")
                for b in blockers:
                    src = b.get("source", FALLBACK_NOT_SPECIFIED)
                    sources.add(src)
                    context_blocks.append(
                        f"Blocker: {b.get('description', 'N/A')}\n"
                        f"Impact: {b.get('impact', FALLBACK_NOT_SPECIFIED)}\n"
                        f"Owner: {b.get('owner', 'Not specified')}\n"
                        f"Status: {b.get('status', 'Active')}\n"
                        f"Source: {src}"
                    )

            if decisions:
                context_blocks.append("\n--- PENDING DECISIONS ---")
                for d in decisions:
                    src = d.get("source", FALLBACK_NOT_SPECIFIED)
                    sources.add(src)
                    context_blocks.append(
                        f"Pending Decision: {d.get('decision', 'N/A')}\n"
                        f"Owner: {d.get('owner', FALLBACK_NOT_SPECIFIED)}\n"
                        f"Source: {src}"
                    )

            if issues:
                context_blocks.append("\n--- UNRESOLVED ISSUES ---")
                for i in issues:
                    src = i.get("source", FALLBACK_NOT_SPECIFIED)
                    sources.add(src)
                    context_blocks.append(
                        f"Issue: {i.get('issue', 'N/A')}\n"
                        f"Status: {i.get('status', 'Open')}\n"
                        f"Source: {src}"
                    )

            context_text = "\n\n".join(context_blocks)
            prompt = ACTION_ITEMS_DOC_PROMPT.format(context=context_text)

            system_instruction = (
                "You are an expert Agile Scrum Master. Generate a formal, structured Action Item List "
                "from the provided project actions, blockers, and decisions. "
                "GROUNDING RULE: Do NOT invent actions unsupported by the documents. "
                "If assignee, priority, or deadline is missing from the documents, strictly mark: "
                "'Not specified in the available project documents.'"
            )

            # 3. Single LLM call with structured output validation (with Groq fallback)
            response_dict = self._execute_structured(
                prompt=prompt,
                response_schema=ActionItemsOutput,
                system_instruction=system_instruction,
            )

            parsed_output: ActionItemsOutput = self._validate_model(ActionItemsOutput, response_dict)  # type: ignore

            # 4. Strict grounding sanitization & sequential numbering
            cleaned_actions: List[ActionItem] = []
            for idx, item in enumerate(parsed_output.action_items, 1):
                action_id = item.action_id or f"ACT-{idx:02d}"
                if not action_id.startswith("ACT-"):
                    action_id = f"ACT-{idx:02d}"

                act_text = item.action.strip() if item.action else FALLBACK_NOT_SPECIFIED
                assignee = item.assignee.strip() if item.assignee else FALLBACK_NOT_SPECIFIED
                prio = item.priority.strip() if item.priority else FALLBACK_NOT_SPECIFIED
                dl = item.deadline.strip() if item.deadline else FALLBACK_NOT_SPECIFIED
                stat = item.status.strip() if item.status else "Open"
                rel = item.related_blocker_risk.strip() if item.related_blocker_risk else FALLBACK_NOT_SPECIFIED
                src = item.source.strip() if item.source else FALLBACK_NOT_SPECIFIED
                ev = item.evidence.strip() if item.evidence else FALLBACK_NOT_SPECIFIED

                cleaned_actions.append(
                    ActionItem(
                        action_id=action_id,
                        action=act_text,
                        assignee=assignee,
                        priority=prio,
                        deadline=dl,
                        status=stat,
                        related_blocker_risk=rel,
                        source=src,
                        evidence=ev,
                    )
                )

            deduped_sources = self._deduplicate(list(sources.union(parsed_output.sources)))
            result_output = ActionItemsOutput(action_items=cleaned_actions, sources=deduped_sources)

            return {
                "data": result_output.model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": provider_name,
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Success",
                    "error": None,
                },
            }

        except Exception as exc:
            logger.error("Action items generation error: %s", exc, exc_info=True)
            return {
                "data": ActionItemsOutput().model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "document_type": doc_type,
                    "provider": provider_name if "provider_name" in locals() else "Unknown",
                    "execution_time_seconds": round(time.time() - start_time, 2),
                    "status": "Failed",
                    "error": str(exc),
                },
            }

    # -------------------------------------------------------------------------
    # 4. Generate All Documents
    # -------------------------------------------------------------------------

    def generate_all(
        self,
        project_id: str,
        scope_data: Optional[Dict[str, Any]] = None,
        risk_data: Optional[Dict[str, Any]] = None,
        blocker_data: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate all three document types in a unified workflow reusing Milestone 2 outputs.

        Returns:
            Dict[str, Any]: Complete documentation payload with individual sections and metadata.
        """
        start_time = time.time()
        stories_res = self.generate_user_stories(project_id=project_id, scope_data=scope_data, top_k=top_k)
        risks_res = self.generate_risk_register(project_id=project_id, risk_data=risk_data, top_k=top_k)
        actions_res = self.generate_action_items(project_id=project_id, blocker_data=blocker_data, top_k=top_k)

        all_sources = set()
        all_sources.update(stories_res.get("data", {}).get("sources", []))
        all_sources.update(risks_res.get("data", {}).get("sources", []))
        all_sources.update(actions_res.get("data", {}).get("sources", []))

        # Check for any errors
        errors = [
            f"{res['metadata']['document_type']}: {res['metadata']['error']}"
            for res in [stories_res, risks_res, actions_res]
            if res.get("metadata", {}).get("status") == "Failed"
        ]
        overall_status = "Failed" if len(errors) == 3 else ("Partial Success" if errors else "Success")

        doc_result = DocumentationGenerationResult(
            user_stories=[UserStory(**s) for s in stories_res.get("data", {}).get("user_stories", [])],
            risk_register=[RiskRegisterItem(**r) for r in risks_res.get("data", {}).get("risk_register", [])],
            action_items=[ActionItem(**a) for a in actions_res.get("data", {}).get("action_items", [])],
            sources=self._deduplicate(list(all_sources)),
        )

        return {
            "data": doc_result.model_dump(),
            "metadata": {
                "agent_name": "Documentation Generation Agent",
                "document_type": "All Documents",
                "execution_time_seconds": round(time.time() - start_time, 2),
                "status": overall_status,
                "error": "; ".join(errors) if errors else None,
            },
        }

    # -------------------------------------------------------------------------
    # Rule-Based Context Analysis (Offline / Orchestrator Fallback)
    # -------------------------------------------------------------------------

    def analyze_context(
        self,
        retrieved_chunks: Sequence[Dict[str, Any]] | None,
        project_name: str | None = None,
    ) -> DocumentationGenerationResult:
        """
        Backwards-compatible rule-based extraction fallback for context analysis.
        """
        if not retrieved_chunks:
            return DocumentationGenerationResult()

        chunks = self._normalize_chunks(retrieved_chunks)
        stories: List[UserStory] = []
        risks: List[RiskRegisterItem] = []
        actions: List[ActionItem] = []
        sources = set()

        for idx, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            source = self._extract_source(chunk)
            sources.add(source)

            # Heuristic user story from deliverables
            if "deliverable" in text.lower() or "feature" in text.lower() or "objective" in text.lower():
                stories.append(
                    UserStory(
                        story_id=f"US-{len(stories) + 1:02d}",
                        user_story=f"As a project user, I want {text[:80]} so that the project objective is achieved.",
                        description=text[:150],
                        priority="Must Have",
                        acceptance_criteria=[f"Implement {text[:50]}"],
                        dependencies=FALLBACK_NOT_SPECIFIED,
                        source=source,
                        evidence=text[:150],
                    )
                )

            # Heuristic risk register item
            if "risk" in text.lower() or "delay" in text.lower() or "blocked" in text.lower():
                risks.append(
                    RiskRegisterItem(
                        risk_id=f"R-{len(risks) + 1:02d}",
                        risk_description=text[:150],
                        category="Schedule / Technical" if "delay" in text.lower() else "Technical",
                        probability="Medium",
                        impact="High" if "blocked" in text.lower() else "Medium",
                        severity="High" if "blocked" in text.lower() else "Medium",
                        mitigation=FALLBACK_NOT_SPECIFIED,
                        owner=FALLBACK_NOT_SPECIFIED,
                        status="Open",
                        source=source,
                        evidence=text[:150],
                    )
                )

            # Heuristic action item
            if "action" in text.lower() or "task" in text.lower() or "blocked" in text.lower():
                actions.append(
                    ActionItem(
                        action_id=f"ACT-{len(actions) + 1:02d}",
                        action=text[:150],
                        assignee=FALLBACK_NOT_SPECIFIED,
                        priority="High" if "blocked" in text.lower() else "Medium",
                        deadline=FALLBACK_NOT_SPECIFIED,
                        status="Open",
                        related_blocker_risk=text[:100] if "blocked" in text.lower() else FALLBACK_NOT_SPECIFIED,
                        source=source,
                        evidence=text[:150],
                    )
                )

        return DocumentationGenerationResult(
            user_stories=stories,
            risk_register=risks,
            action_items=actions,
            sources=list(sources),
        )

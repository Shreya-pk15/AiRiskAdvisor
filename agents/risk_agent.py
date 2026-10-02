"""
Risk Detection and Delivery Forecasting Agent.

Analyzes project context to identify schedule, dependency, resource, technical,
quality, planning, and delivery risks using RAG retrieval and Groq API.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from agents.prompts import RISK_PROMPT
from agents.schemas import DeliveryForecast, RiskDetectionOutput, RiskItem
from llm.base_provider import BaseLLMProvider
from llm.groq_provider import GroqProvider
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

FALLBACK_NOT_SPECIFIED = "Not specified in the available project documents."


class RiskDetectionAgent(BaseAgent):
    """
    Risk Detection and Delivery Forecasting Agent.

    Uses RAG retrieval with target queries filtered by project_id and sends context
    to Groq to produce a structured Pydantic RiskDetectionOutput.
    """

    TARGET_QUERIES = [
        "overdue incomplete delayed tasks",
        "project dependencies blockers",
        "defects unresolved issues",
        "deadlines schedule delivery",
        "project progress risks",
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
        Execute full Risk Detection & Delivery Forecasting workflow for a given project_id.

        Args:
            project_id (str): Project identifier / workspace name.
            top_k (int): Maximum number of distinct context chunks across targeted queries.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "data": RiskDetectionOutput dictionary
                - "metadata": Execution metadata (agent_name, provider, execution_time_seconds, status, error)
        """
        start_time = time.time()
        agent_name = "Risk & Delivery Agent"
        provider_name = "Groq"

        # Initialize provider if not supplied
        if self.provider is None:
            try:
                self.provider = GroqProvider(model_name=self.model_name)
            except Exception as e:
                exec_time = round(time.time() - start_time, 2)
                return {
                    "data": self._empty_risk_output().model_dump(),
                    "metadata": {
                        "agent_name": agent_name,
                        "provider": provider_name,
                        "execution_time_seconds": exec_time,
                        "status": "Failed",
                        "error": f"Failed to initialize Groq Provider: {e}"
                    }
                }

        try:
            # 1. RAG Retrieval filtered by project_id
            retrieved_chunks = self._retrieve_targeted_chunks(project_id=project_id, top_k=top_k)

            if not retrieved_chunks:
                exec_time = round(time.time() - start_time, 2)
                empty_res = self._empty_risk_output()
                empty_res.delivery_status = "Insufficient Data"
                empty_res.delivery_forecast.reason = "No project documents found in knowledge base."
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

            # 2. Extract structured risk analysis via single Groq call
            risk_output = self._extract_with_llm(retrieved_chunks)
            exec_time = round(time.time() - start_time, 2)

            return {
                "data": risk_output.model_dump(),
                "metadata": {
                    "agent_name": agent_name,
                    "provider": provider_name,
                    "execution_time_seconds": exec_time,
                    "status": "Success",
                    "error": None
                }
            }

        except Exception as exc:
            logger.error("Risk detection agent error: %s", exc, exc_info=True)
            exec_time = round(time.time() - start_time, 2)
            return {
                "data": self._empty_risk_output().model_dump(),
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

    def _extract_with_llm(self, retrieved_chunks: Sequence[Dict[str, Any]]) -> RiskDetectionOutput:
        """Construct context and generate structured risk response via GroqProvider."""
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
        prompt = RISK_PROMPT.format(context=context_text)

        system_instruction = (
            "You are an expert AI Risk Detection and Delivery Forecasting Agent for software projects. "
            "Analyze project context strictly using grounded document evidence. "
            "Categorize risks as Schedule, Dependency, Resource, Technical, Quality, Planning, or Delivery. "
            "Assign severity (Low, Medium, High) based strictly on evidence. "
            "Determine delivery status: On Track, At Risk, Delayed, or Insufficient Data. "
            "Do NOT invent facts, dates, severity, or delays. If there is no risk evidence, return an empty risk list."
        )

        response_dict = self.provider.generate_structured(
            prompt=prompt,
            response_schema=RiskDetectionOutput,
            system_instruction=system_instruction
        )

        parsed_output = self._validate_model(RiskDetectionOutput, response_dict)

        if not parsed_output.sources and unique_sources:
            parsed_output.sources = self._deduplicate(list(unique_sources))

        return self._apply_grounding_sanitization(parsed_output)

    def _apply_grounding_sanitization(self, output: RiskDetectionOutput) -> RiskDetectionOutput:
        """Enforce strict grounding and fallback sanitization."""
        # Sanitize individual risks
        for r in output.risks:
            if not r.source or r.source.strip() == "":
                r.source = FALLBACK_NOT_SPECIFIED
            if not r.evidence or r.evidence.strip() == "":
                r.evidence = FALLBACK_NOT_SPECIFIED

        # Align delivery forecast status with top-level status
        if output.delivery_status:
            output.delivery_forecast.status = output.delivery_status

        return output

    def _empty_risk_output(self) -> RiskDetectionOutput:
        """Return an empty grounded RiskDetectionOutput instance."""
        return RiskDetectionOutput(
            risks=[],
            delivery_status="Insufficient Data",
            delivery_forecast=DeliveryForecast(
                status="Insufficient Data",
                reason="No documented evidence available.",
                forecast_analysis="No forward-looking analysis available due to lack of evidence."
            ),
            sources=[]
        )

    def analyze_context(
        self,
        retrieved_chunks: Sequence[Dict[str, Any]] | None,
        project_name: str | None = None
    ) -> RiskDetectionOutput:
        """
        Backwards-compatible interface for orchestrator usage.
        """
        if not retrieved_chunks:
            return self._empty_risk_output()

        if self.provider is not None:
            try:
                return self._extract_with_llm(retrieved_chunks)
            except Exception as e:
                logger.warning("Groq extraction failed in analyze_context, falling back: %s", e)

        # Basic fallback for context analysis
        chunks = self._normalize_chunks(retrieved_chunks)
        risks = []
        sources = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            source = self._extract_source(chunk)
            sources.add(source)

            if any(k in text.lower() for k in ["delay", "blocker", "overdue", "defect", "risk"]):
                risks.append(
                    RiskItem(
                        category="Schedule",
                        description=text[:150],
                        severity="Medium",
                        evidence=text[:150],
                        source=source
                    )
                )

        status = "At Risk" if risks else "On Track"
        return RiskDetectionOutput(
            risks=risks,
            delivery_status=status,
            delivery_forecast=DeliveryForecast(
                status=status,
                reason="Derived from context keyword analysis.",
                forecast_analysis="Potential schedule impact based on context signals."
            ),
            sources=list(sources)
        )

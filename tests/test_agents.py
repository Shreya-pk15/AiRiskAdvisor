import pytest

from agents.scope_agent import ScopeExtractionAgent
from agents.risk_agent import RiskDetectionAgent
from agents.blocker_agent import BlockerActionAgent
from agents.orchestrator import AgentOrchestrator


SAMPLE_CONTEXT = [
    {
        "text": "Project Title: Student Project Management System (SPMS). The primary objective is to provide a unified platform for undergraduate software development teams to collaborate, track sprint deliverables, submit progress reports, and assess project risks in real time.",
        "metadata": {"source": "sample_proposal.pdf"},
    },
    {
        "text": "Key deliverables: Document Ingestion Pipeline, Grounded Vector Retrieval Architecture, Interactive Project Health Dashboard. Milestones: Week 1 ingestion, Week 2 retrieval, Week 4 risk forecasting agent.",
        "metadata": {"source": "sample_srs.docx"},
    },
    {
        "text": "Payment Integration Task is currently BLOCKED. The payment gateway provider has not issued the production API credentials. John cannot proceed until credentials arrive. Decision: Next sprint will focus on testing ChromaDB retrieval accuracy.",
        "metadata": {"source": "sample_meeting_notes.txt"},
    },
]


def test_scope_agent_extracts_project_goals_and_milestones():
    agent = ScopeExtractionAgent()
    result = agent.analyze_context(SAMPLE_CONTEXT, project_name="SPMS")

    assert result.project_goals
    assert any("objective" in goal.goal.lower() or "platform" in goal.goal.lower() for goal in result.project_goals)
    assert result.deliverables
    assert result.milestones
    assert result.timeline or result.timeline == []


def test_risk_agent_detects_dependency_and_delivery_risk():
    agent = RiskDetectionAgent()
    result = agent.analyze_context(SAMPLE_CONTEXT, project_name="SPMS")

    assert result.risks
    assert any(r.category.lower() in {"schedule", "dependency", "delivery"} for r in result.risks)
    assert result.delivery_forecast is not None
    assert result.delivery_forecast.status in {"On Track", "At Risk", "Delayed", "Insufficient Data"}


def test_blocker_agent_extracts_pending_decisions_and_actions():
    agent = BlockerActionAgent()
    result = agent.analyze_context(SAMPLE_CONTEXT, project_name="SPMS")

    assert result.blockers or result.pending_decisions or result.unresolved_issues or result.action_items
    assert result.action_items or result.pending_decisions


def test_orchestrator_returns_structured_intelligence():
    orchestrator = AgentOrchestrator(project_name="SPMS")
    result = orchestrator.run_all(SAMPLE_CONTEXT)

    assert result["scope"] is not None
    assert result["risks"] is not None
    assert result["blockers"] is not None


def test_groq_and_gemini_provider_json_extraction_edge_cases():
    from llm.groq_provider import GroqProvider
    from llm.gemini_provider import GeminiProvider
    from agents.schemas import BlockerActionOutput

    groq_provider = GroqProvider(api_key="mock_key")
    gemini_provider = GeminiProvider(api_key="mock_key")

    # Case 1: Reasoning <think> tags + markdown code fence
    case1 = """<think>
    Thinking about the blockers in the text...
    Found 1 blocker.
    </think>
    ```json
    {
      "blockers": [{"description": "DevOps delay", "source": "Notes.txt"}],
      "pending_decisions": [],
      "unresolved_issues": [],
      "action_items": [],
      "sources": ["Notes.txt"]
    }
    ```"""

    res1_groq = groq_provider._parse_and_validate_json(case1, BlockerActionOutput)
    res1_gemini = gemini_provider._parse_and_validate_json(case1, BlockerActionOutput)
    assert len(res1_groq["blockers"]) == 1
    assert len(res1_gemini["blockers"]) == 1

    # Case 2: Preamble + trailing commas
    case2 = """Here is the extracted blocker JSON data:
    {
      "blockers": [],
      "pending_decisions": [],
      "unresolved_issues": [],
      "action_items": [],
      "sources": ["doc1.pdf",],
    }
    Hope this helps!"""

    res2_groq = groq_provider._parse_and_validate_json(case2, BlockerActionOutput)
    res2_gemini = gemini_provider._parse_and_validate_json(case2, BlockerActionOutput)
    assert res2_groq["sources"] == ["doc1.pdf"]
    assert res2_gemini["sources"] == ["doc1.pdf"]


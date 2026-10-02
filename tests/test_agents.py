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

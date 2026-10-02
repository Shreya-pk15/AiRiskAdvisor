"""
Unit Tests for Project Health Scoring Module (Milestone 3 — Part 2).

Tests cover:
1. Healthy project evaluation (score >= 80, classification 'Healthy')
2. High-risk project evaluation (high-severity risks reducing delivery score)
3. Many-blocker project evaluation (active blockers reducing blocker score)
4. Delayed project evaluation (delayed delivery status reducing timeline score)
5. Insufficient data handling (low confidence indicator and explanation)
6. Bounded scoring (scores strictly clamped between 0 and 100 under extreme values)
7. Correct classification mapping and configurable thresholds
8. Project isolation verification
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.health_scorer import ProjectHealthScorer, DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS
from agents.schemas import ProjectHealthOutput


@pytest.fixture
def healthy_scope_data():
    return {
        "project_goals": [
            {"goal": "Build Intelligent Risk Platform", "source": "Spec.pdf", "evidence": "Primary goal"},
            {"goal": "Automate Delivery Tracking", "source": "Spec.pdf", "evidence": "Secondary goal"},
        ],
        "deliverables": [
            {"name": "Ingestion Pipeline", "source": "Spec.pdf", "evidence": "Deliverable 1"},
            {"name": "Vector Retrieval", "source": "Spec.pdf", "evidence": "Deliverable 2"},
            {"name": "Health Dashboard", "source": "Spec.pdf", "evidence": "Deliverable 3"},
        ],
        "milestones": [
            {"name": "Milestone 1", "target_date": "2026-10-01", "status": "Completed", "source": "Spec.pdf"},
            {"name": "Milestone 2", "target_date": "2026-10-15", "status": "In Progress", "source": "Spec.pdf"},
        ],
        "timeline": [
            {"label": "Start Date", "value": "2026-09-01", "source": "Spec.pdf"},
            {"label": "End Date", "value": "2026-11-01", "source": "Spec.pdf"},
        ],
        "sources": ["Spec.pdf"],
    }


@pytest.fixture
def healthy_risk_data():
    return {
        "risks": [
            {
                "risk_id": "R-01",
                "category": "Technical",
                "description": "Minor cache warm-up delay.",
                "severity": "Low",
                "source": "Spec.pdf",
            }
        ],
        "delivery_status": "On Track",
        "delivery_forecast": {
            "status": "On Track",
            "reason": "Development is progressing ahead of schedule.",
            "concerns": [],
        },
        "sources": ["Spec.pdf"],
    }


@pytest.fixture
def healthy_blocker_data():
    return {
        "blockers": [],
        "pending_decisions": [],
        "unresolved_issues": [],
        "action_items": [
            {"action": "Review documentation", "status": "Open", "source": "Spec.pdf"}
        ],
        "sources": ["Spec.pdf"],
    }


# =============================================================================
# 1. Healthy Project Test
# =============================================================================

def test_healthy_project_scoring(healthy_scope_data, healthy_risk_data, healthy_blocker_data):
    """Verify that a project with clear scope, no blockers, low risk, and on-track timeline scores >= 80 and is Healthy."""
    scorer = ProjectHealthScorer()
    health = scorer.evaluate_health(
        scope_data=healthy_scope_data,
        risk_data=healthy_risk_data,
        blocker_data=healthy_blocker_data,
    )

    assert isinstance(health, ProjectHealthOutput)
    assert health.overall_score >= 80.0
    assert health.classification == "Healthy"
    assert health.confidence == "High"
    assert health.dimensions["Scope Clarity"].score == 100.0
    assert health.dimensions["Timeline Risk"].score == 95.0
    assert health.dimensions["Blocker Status"].score == 100.0
    assert health.dimensions["Delivery Risk"].score == 96.0  # 100 - 4 (1 low risk)
    assert any("project deliverable" in f.lower() for f in health.key_factors)
    assert "Spec.pdf" in health.sources


# =============================================================================
# 2. High-Risk Project Test
# =============================================================================

def test_high_risk_project_scoring(healthy_scope_data, healthy_blocker_data):
    """Verify that multiple high-severity risks depress the Delivery Risk dimension and lower overall health."""
    high_risk_data = {
        "risks": [
            {"risk_id": "R-01", "severity": "High", "description": "Database outage risk"},
            {"risk_id": "R-02", "severity": "High", "description": "Security vulnerability"},
            {"risk_id": "R-03", "severity": "Medium", "description": "Vendor SLA breach"},
        ],
        "delivery_status": "At Risk",
        "delivery_forecast": {"status": "At Risk", "concerns": ["Security", "Database"]},
        "sources": ["RiskLog.pdf"],
    }

    scorer = ProjectHealthScorer()
    health = scorer.evaluate_health(
        scope_data=healthy_scope_data,
        risk_data=high_risk_data,
        blocker_data=healthy_blocker_data,
    )

    # 100 - (2 * 20 + 1 * 10) = 50.0
    assert health.dimensions["Delivery Risk"].score == 50.0
    assert health.overall_score < 80.0
    assert any("high-severity risk" in f.lower() for f in health.key_factors)


# =============================================================================
# 3. Many-Blocker Project Test
# =============================================================================

def test_many_blocker_project_scoring(healthy_scope_data, healthy_risk_data):
    """Verify that multiple active blockers and unresolved issues sharply reduce the Blocker Status score."""
    blocker_data = {
        "blockers": [
            {"description": "API Gateway down", "status": "Active"},
            {"description": "Credential delay", "status": "Active"},
            {"description": "Test environment crashed", "status": "Active"},
        ],
        "pending_decisions": [
            {"decision": "Framework selection", "status": "Pending"}
        ],
        "unresolved_issues": [
            {"issue": "Memory leak on file upload", "status": "Open"}
        ],
        "action_items": [],
        "sources": ["MeetingNotes.txt"],
    }

    scorer = ProjectHealthScorer()
    health = scorer.evaluate_health(
        scope_data=healthy_scope_data,
        risk_data=healthy_risk_data,
        blocker_data=blocker_data,
    )

    assert health.dimensions["Blocker Status"].score == 32.0
    assert health.dimensions["Blocker Status"].status == "Critical"
    assert health.overall_score < 85.0
    assert any("3 active blocker(s)" in f for f in health.key_factors)


# =============================================================================
# 4. Delayed Project Test
# =============================================================================

def test_delayed_project_scoring(healthy_scope_data, healthy_blocker_data):
    """Verify that a 'Delayed' delivery status directly assigns a low timeline score (<= 25)."""
    delayed_risk_data = {
        "risks": [],
        "delivery_status": "Delayed",
        "delivery_forecast": {
            "status": "Delayed",
            "reason": "Milestone 1 deadline was missed by 2 weeks.",
            "concerns": ["Overdue milestones", "Resource crunch"],
        },
        "sources": ["SprintReport.docx"],
    }

    scorer = ProjectHealthScorer()
    health = scorer.evaluate_health(
        scope_data=healthy_scope_data,
        risk_data=delayed_risk_data,
        blocker_data=healthy_blocker_data,
    )

    # Base delayed = 25.0 - (2 concerns * 5) = 15.0
    assert health.dimensions["Timeline Risk"].score <= 25.0
    assert health.dimensions["Timeline Risk"].status == "Critical"
    assert any("timeline currently classified as 'delayed'" in f.lower() for f in health.key_factors)


# =============================================================================
# 5. Insufficient Data Handling Test
# =============================================================================

def test_insufficient_data_scoring():
    """Verify that when project documents are missing or sparse, confidence is Low and sufficiency is explained."""
    scorer = ProjectHealthScorer()
    health = scorer.evaluate_health(
        scope_data={},
        risk_data={},
        blocker_data={},
    )

    assert health.confidence == "Low"
    assert "sparse" in health.data_sufficiency.lower()
    assert 0.0 <= health.overall_score <= 100.0
    assert any("additional project artifacts" in r.lower() for r in health.recommendations)


# =============================================================================
# 6. Score Bounded Between 0 and 100 Under Extreme Penalties
# =============================================================================

def test_score_bounded_between_zero_and_hundred():
    """Verify that extreme penalties (e.g. 50 blockers, 50 high risks) never produce negative numbers or > 100."""
    extreme_blockers = {
        "blockers": [{"description": f"Blocker {i}", "status": "Active"} for i in range(50)],
        "unresolved_issues": [{"issue": f"Issue {i}", "status": "Open"} for i in range(50)],
    }
    extreme_risks = {
        "risks": [{"risk_id": f"R-{i}", "severity": "High"} for i in range(50)],
        "delivery_status": "Delayed",
        "delivery_forecast": {"concerns": ["C1", "C2", "C3", "C4", "C5"]},
    }
    empty_scope = {}

    scorer = ProjectHealthScorer()
    health = scorer.evaluate_health(
        scope_data=empty_scope,
        risk_data=extreme_risks,
        blocker_data=extreme_blockers,
    )

    assert health.overall_score >= 0.0
    assert health.overall_score <= 100.0
    for dim in health.dimensions.values():
        assert 0.0 <= dim.score <= 100.0
        assert 0.0 <= dim.weighted_score <= 100.0

    assert health.classification == "Critical"


# =============================================================================
# 7. Classification Thresholds & Configurable Thresholds Test
# =============================================================================

def test_classification_thresholds():
    """Verify that threshold mappings accurately categorize scores into Healthy, Moderate, At Risk, Critical."""
    scorer = ProjectHealthScorer()

    assert scorer._classify_score(100.0) == "Healthy"
    assert scorer._classify_score(80.0) == "Healthy"
    assert scorer._classify_score(79.9) == "Moderate"
    assert scorer._classify_score(60.0) == "Moderate"
    assert scorer._classify_score(59.9) == "At Risk"
    assert scorer._classify_score(40.0) == "At Risk"
    assert scorer._classify_score(39.9) == "Critical"
    assert scorer._classify_score(0.0) == "Critical"

    # Test custom configurable thresholds
    custom_thresholds = {
        "Healthy": 90.0,
        "Moderate": 75.0,
        "At Risk": 50.0,
        "Critical": 0.0,
    }
    custom_scorer = ProjectHealthScorer(thresholds=custom_thresholds)
    assert custom_scorer._classify_score(85.0) == "Moderate"
    assert custom_scorer._classify_score(92.0) == "Healthy"


# =============================================================================
# 8. Project Isolation Verification
# =============================================================================

def test_project_isolation_in_health_scorer():
    """Verify that when project_id is passed, agent runs are scoped strictly to that project_id."""
    with patch("agents.scope_agent.ScopeExtractionAgent") as MockScope, \
         patch("agents.risk_agent.RiskDetectionAgent") as MockRisk, \
         patch("agents.blocker_agent.BlockerActionAgent") as MockBlocker:

        mock_scope_inst = MagicMock()
        mock_scope_inst.run.return_value = {"data": {"project_goals": [{"goal": "G1", "source": "doc.pdf"}]}}
        MockScope.return_value = mock_scope_inst

        mock_risk_inst = MagicMock()
        mock_risk_inst.run.return_value = {"data": {"risks": [], "delivery_status": "On Track"}}
        MockRisk.return_value = mock_risk_inst

        mock_blocker_inst = MagicMock()
        mock_blocker_inst.run.return_value = {"data": {"blockers": [], "action_items": []}}
        MockBlocker.return_value = mock_blocker_inst

        scorer = ProjectHealthScorer()
        health = scorer.evaluate_health(
            scope_data=None,
            risk_data=None,
            blocker_data=None,
            project_id="Tenant_Workspace_99",
        )

        mock_scope_inst.run.assert_called_once_with(project_id="Tenant_Workspace_99", top_k=5)
        mock_risk_inst.run.assert_called_once_with(project_id="Tenant_Workspace_99", top_k=5)
        mock_blocker_inst.run.assert_called_once_with(project_id="Tenant_Workspace_99", top_k=5)
        assert health.overall_score >= 0.0

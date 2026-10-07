"""
Unit Tests for Risk Detection and Delivery Forecasting Agent (Groq).

Tests cover:
1. Risk extraction
2. Severity classification (Low, Medium, High)
3. Delivery status (On Track, At Risk, Delayed, Insufficient Data)
4. Insufficient evidence handling
5. Source/evidence preservation
6. Groq API failure handling
7. Project isolation verification
8. Invalid structured output handling
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.risk_agent import RiskDetectionAgent, FALLBACK_NOT_SPECIFIED
from agents.schemas import DeliveryForecast, RiskDetectionOutput, RiskItem
from llm.base_provider import BaseLLMProvider
from llm.groq_provider import GroqProvider


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    chunks = [
        {
            "chunk_id": "chunk_defect_1",
            "text": "Payment gateway integration has 3 unresolved critical defects. Sprint release scheduled for Friday is delayed due to API authentication failures.",
            "metadata": {
                "source": "Defect_Tracker.xlsx",
                "document_id": "doc_202",
                "page": 2,
                "chunk_id": "chunk_defect_1"
            }
        }
    ]
    retriever.retrieve.return_value = chunks
    retriever.retrieve_all.return_value = chunks
    return retriever


@pytest.fixture
def mock_groq_provider():
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.return_value = {
        "risks": [
            {
                "risk_id": "RISK-001",
                "category": "Technical",
                "description": "Unresolved critical defects in payment gateway integration",
                "severity": "High",
                "evidence": "Payment gateway integration has 3 unresolved critical defects.",
                "source": "Defect_Tracker.xlsx",
                "document_name": "Defect_Tracker.xlsx",
                "document_id": "doc_202",
                "page": 2,
                "chunk_id": "chunk_defect_1",
                "recommended_action": "Fix authentication bugs before release"
            },
            {
                "risk_id": "RISK-002",
                "category": "Schedule",
                "description": "Sprint release delay caused by API failures",
                "severity": "Medium",
                "evidence": "Sprint release scheduled for Friday is delayed due to API authentication failures.",
                "source": "Defect_Tracker.xlsx",
                "document_name": "Defect_Tracker.xlsx",
                "document_id": "doc_202",
                "page": 2,
                "chunk_id": "chunk_defect_1",
                "recommended_action": "Reschedule sprint release deadline"
            }
        ],
        "delivery_status": "Delayed",
        "delivery_forecast": {
            "status": "Delayed",
            "reason": "Documented critical defects and API authentication failures.",
            "forecast_analysis": "The release is at significant risk of missing Friday target due to unresolved payment API defects.",
            "evidence": ["Payment gateway integration has 3 unresolved critical defects."],
            "concerns": ["API authentication failures", "Overdue testing"],
            "recommended_actions": ["Focus senior dev effort on API auth fix"]
        },
        "sources": ["Defect_Tracker.xlsx"]
    }
    return provider


def test_risk_extraction(mock_groq_provider, mock_retriever):
    """Test 1: Valid Risk Extraction from Groq API mock response."""
    agent = RiskDetectionAgent(provider=mock_groq_provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_Risk_Test")

    assert res["metadata"]["status"] == "Success"
    assert res["metadata"]["agent_name"] == "Risk & Delivery Agent"
    assert res["metadata"]["provider"] == "Groq"
    assert res["metadata"]["execution_time_seconds"] >= 0.0

    data = res["data"]
    assert len(data["risks"]) == 2
    assert data["risks"][0]["category"] == "Technical"
    assert data["risks"][0]["severity"] == "High"
    assert data["risks"][1]["category"] == "Schedule"


def test_severity_classification(mock_groq_provider, mock_retriever):
    """Test 2: Evidence-based Severity Classification (Low, Medium, High)."""
    agent = RiskDetectionAgent(provider=mock_groq_provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_Severity_Test")

    risks = res["data"]["risks"]
    severities = [r["severity"] for r in risks]

    assert "High" in severities
    assert "Medium" in severities
    for r in risks:
        assert r["severity"] in ["Low", "Medium", "High"]


def test_delivery_status(mock_groq_provider, mock_retriever):
    """Test 3: Delivery Status determination (Delayed)."""
    agent = RiskDetectionAgent(provider=mock_groq_provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_Status_Test")

    assert res["data"]["delivery_status"] == "Delayed"
    assert res["data"]["delivery_forecast"]["status"] == "Delayed"
    assert "Friday target" in res["data"]["delivery_forecast"]["forecast_analysis"]


def test_insufficient_evidence_handling(mock_retriever):
    """Test 4: Insufficient Evidence handling returns empty risks and Insufficient Data status."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.return_value = {
        "risks": [],
        "delivery_status": "Insufficient Data",
        "delivery_forecast": {
            "status": "Insufficient Data",
            "reason": "Not enough risk evidence documented.",
            "forecast_analysis": "No forward-looking risk analysis possible due to missing data.",
            "evidence": [],
            "concerns": [],
            "recommended_actions": []
        },
        "sources": []
    }

    agent = RiskDetectionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_NoData_Test")

    assert res["metadata"]["status"] == "Success"
    assert res["data"]["delivery_status"] == "Insufficient Data"
    assert len(res["data"]["risks"]) == 0


def test_source_and_evidence_preservation(mock_groq_provider, mock_retriever):
    """Test 5: Source and evidence preservation in detected risk objects."""
    agent = RiskDetectionAgent(provider=mock_groq_provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_Traceability_Test")

    risk = res["data"]["risks"][0]
    assert risk["source"] == "Defect_Tracker.xlsx"
    assert risk["document_name"] == "Defect_Tracker.xlsx"
    assert risk["document_id"] == "doc_202"
    assert risk["page"] == 2
    assert risk["chunk_id"] == "chunk_defect_1"
    assert "3 unresolved critical defects" in risk["evidence"]


def test_groq_api_failure_handling(mock_retriever):
    """Test 6: Groq API failure returns status Failed without crashing."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.side_effect = RuntimeError("Groq API rate limit exceeded (429)")

    agent = RiskDetectionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_GroqFail_Test")

    assert res["metadata"]["status"] == "Failed"
    assert "Groq API rate limit" in res["metadata"]["error"]
    assert isinstance(res["data"], dict)
    assert res["data"]["risks"] == []


def test_groq_provider_uses_fallback_for_unavailable_model(monkeypatch):
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "fallback/model")
    provider = GroqProvider(api_key="test-key", model_name="missing/model")

    with patch.object(
        provider,
        "_generate_structured_with_model",
        side_effect=[RuntimeError("model_not_found"), {"risks": []}],
    ) as generate:
        result = provider.generate_structured("prompt", RiskDetectionOutput)

    assert result == {"risks": []}
    assert [call.kwargs["model_id"] for call in generate.call_args_list] == [
        "missing/model",
        "fallback/model",
    ]
    assert provider.model_name == "fallback/model"


def test_project_isolation_verification(mock_groq_provider):
    """Test 7: Project isolation - retriever query calls filter by project_id."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.retrieve_all.return_value = []

    agent = RiskDetectionAgent(provider=mock_groq_provider, retriever=mock_retriever)
    agent.run(project_id="Isolated_Risk_Workspace_999")

    mock_retriever.retrieve.assert_any_call(
        query=RiskDetectionAgent.TARGET_QUERIES[0],
        project_name="Isolated_Risk_Workspace_999",
        top_k=5,
    )
    mock_retriever.retrieve_all.assert_not_called()


def test_risk_extraction_respects_top_k_and_similarity(mock_groq_provider, mock_retriever):
    """Only the most similar chunks up to top_k reach the risk model."""
    late_chunk = {
        "chunk_id": "chunk_late",
        "text": "The payment API credentials remain blocked pending vendor approval.",
        "distance": 0.1,
        "metadata": {"source": "Notes.txt", "chunk_id": "chunk_late"},
    }
    mock_retriever.retrieve.return_value = mock_retriever.retrieve.return_value + [late_chunk]

    agent = RiskDetectionAgent(provider=mock_groq_provider, retriever=mock_retriever)
    result = agent.run(project_id="Risk_Project", top_k=1)

    assert result["metadata"]["status"] == "Success"
    prompt = mock_groq_provider.generate_structured.call_args.kwargs["prompt"]
    assert "payment API credentials remain blocked" in prompt
    assert "primary objective is to provide" not in prompt
    mock_retriever.retrieve.assert_any_call(
        query=RiskDetectionAgent.TARGET_QUERIES[0], project_name="Risk_Project", top_k=1
    )


def test_invalid_structured_output_handling(mock_retriever):
    """Test 8: Invalid structured output from LLM recovery."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.side_effect = ValueError("Groq returned malformed JSON schema")

    agent = RiskDetectionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_InvalidJSON_Test")

    assert res["metadata"]["status"] == "Failed"
    assert "malformed JSON" in res["metadata"]["error"]

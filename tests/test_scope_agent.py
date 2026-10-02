"""
Unit Tests for Scope and Deliverable Extraction Agent.

Tests cover:
1. Valid scope extraction
2. Missing information handling ("Not specified in the available project documents.")
3. Source/evidence preservation
4. Invalid LLM output handling
5. Gemini API failure handling
6. Project isolation verification
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.schemas import (
    DeliverableItem,
    MilestoneItem,
    ProjectGoal,
    ResponsibilityItem,
    ScopeExtractionOutput,
    TimelineItem,
)
from agents.scope_agent import ScopeExtractionAgent, FALLBACK_NOT_SPECIFIED
from llm.base_provider import BaseLLMProvider


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "The main project goal is to build an Automated Risk Intelligence Platform. Deliverables include Notification System and Analytics API. Milestone 1 deadline is 2026-10-01. Lead developer: Alice.",
            "metadata": {
                "source": "Project_Spec.pdf",
                "document_id": "doc_101",
                "page": 1,
                "chunk_id": "chunk_1"
            }
        }
    ]
    retriever.retrieve.return_value = chunks
    retriever.retrieve_all.return_value = chunks
    return retriever


@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.return_value = {
        "project_goals": [
            {
                "goal": "Build Automated Risk Intelligence Platform",
                "description": "High-level platform objective",
                "source": "Project_Spec.pdf",
                "document_name": "Project_Spec.pdf",
                "document_id": "doc_101",
                "page": 1,
                "chunk_id": "chunk_1",
                "evidence": "The main project goal is to build an Automated Risk Intelligence Platform."
            }
        ],
        "deliverables": [
            {
                "name": "Notification System",
                "description": "Send alert notifications",
                "source": "Project_Spec.pdf",
                "document_name": "Project_Spec.pdf",
                "document_id": "doc_101",
                "page": 1,
                "chunk_id": "chunk_1",
                "evidence": "Deliverables include Notification System"
            }
        ],
        "milestones": [
            {
                "name": "Milestone 1",
                "description": "Phase 1 completion",
                "target_date": "2026-10-01",
                "status": "Planned",
                "source": "Project_Spec.pdf",
                "evidence": "Milestone 1 deadline is 2026-10-01."
            }
        ],
        "timeline": [
            {
                "label": "Milestone 1 Deadline",
                "value": "2026-10-01",
                "source": "Project_Spec.pdf",
                "evidence": "Milestone 1 deadline is 2026-10-01."
            }
        ],
        "responsibilities": [
            {
                "person": "Alice",
                "responsibility": "Lead Developer",
                "related_deliverable": "Notification System",
                "source": "Project_Spec.pdf",
                "evidence": "Lead developer: Alice."
            }
        ],
        "sources": ["Project_Spec.pdf"]
    }
    return provider


def test_valid_scope_extraction(mock_provider, mock_retriever):
    """Test 1: Valid Scope Extraction from mock LLM response."""
    agent = ScopeExtractionAgent(provider=mock_provider, retriever=mock_retriever)
    res = agent.run(project_id="Test_Project_A")

    assert res["metadata"]["status"] == "Success"
    assert res["metadata"]["agent_name"] == "Scope & Deliverable Agent"
    assert res["metadata"]["provider"] == "Gemini"
    assert res["metadata"]["execution_time_seconds"] >= 0.0

    data = res["data"]
    assert len(data["project_goals"]) == 1
    assert data["project_goals"][0]["goal"] == "Build Automated Risk Intelligence Platform"

    assert len(data["deliverables"]) == 1
    assert data["deliverables"][0]["name"] == "Notification System"

    assert len(data["milestones"]) == 1
    assert data["milestones"][0]["target_date"] == "2026-10-01"

    assert len(data["responsibilities"]) == 1
    assert data["responsibilities"][0]["person"] == "Alice"

    assert "Project_Spec.pdf" in data["sources"]


def test_missing_information_handling(mock_retriever):
    """Test 2: Missing information grounding fallback."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.return_value = {
        "project_goals": [
            {
                "goal": "Build System",
                "source": "",
                "evidence": ""
            }
        ],
        "deliverables": [],
        "milestones": [],
        "timeline": [],
        "responsibilities": [],
        "sources": []
    }

    agent = ScopeExtractionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Test_Project_Missing")

    assert res["metadata"]["status"] == "Success"
    data = res["data"]
    goal = data["project_goals"][0]

    assert goal["source"] == FALLBACK_NOT_SPECIFIED
    assert goal["evidence"] == FALLBACK_NOT_SPECIFIED


def test_source_and_evidence_preservation(mock_provider, mock_retriever):
    """Test 3: Source and evidence attribution preservation."""
    agent = ScopeExtractionAgent(provider=mock_provider, retriever=mock_retriever)
    res = agent.run(project_id="Test_Project_Sources")

    data = res["data"]
    deliverable = data["deliverables"][0]

    assert deliverable["source"] == "Project_Spec.pdf"
    assert deliverable["document_id"] == "doc_101"
    assert deliverable["page"] == 1
    assert deliverable["chunk_id"] == "chunk_1"
    assert "Notification System" in deliverable["evidence"]


def test_invalid_llm_output_handling(mock_retriever):
    """Test 4: Invalid LLM Output error recovery."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.side_effect = ValueError("Invalid JSON schema output from LLM")

    agent = ScopeExtractionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Test_Project_Err")

    assert res["metadata"]["status"] == "Failed"
    assert "Invalid JSON schema" in res["metadata"]["error"]
    assert res["data"]["project_goals"] == []


def test_gemini_api_failure_handling(mock_retriever):
    """Test 5: Gemini API Failure does not crash application."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.side_effect = RuntimeError("API Quota Exceeded (429)")

    agent = ScopeExtractionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Test_Project_Quota")

    assert res["metadata"]["status"] == "Failed"
    assert "API Quota Exceeded" in res["metadata"]["error"]
    assert isinstance(res["data"], dict)


def test_project_isolation_verification(mock_provider):
    """Test 6: Project isolation - retriever called with project_id filter."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.retrieve_all.return_value = []

    agent = ScopeExtractionAgent(provider=mock_provider, retriever=mock_retriever)
    agent.run(project_id="Isolated_Workspace_123")

    mock_retriever.retrieve.assert_any_call(
        query=ScopeExtractionAgent.TARGET_QUERIES[0],
        project_name="Isolated_Workspace_123",
        top_k=5,
    )
    mock_retriever.retrieve_all.assert_not_called()


def test_scope_extraction_respects_top_k_and_similarity(mock_provider, mock_retriever):
    """Only the most similar chunks up to top_k reach the extraction model."""
    late_chunk = {
        "chunk_id": "chunk_2",
        "text": "The final release review is due 2026-12-15 and is owned by the QA team.",
        "distance": 0.1,
        "metadata": {"source": "Project_Plan.docx", "chunk_id": "chunk_2", "page": 4},
    }
    mock_retriever.retrieve.return_value = mock_retriever.retrieve.return_value + [late_chunk]

    agent = ScopeExtractionAgent(provider=mock_provider, retriever=mock_retriever)
    result = agent.run(project_id="Test_Project_A", top_k=1)

    assert result["metadata"]["status"] == "Success"
    prompt = mock_provider.generate_structured.call_args.kwargs["prompt"]
    assert "final release review is due 2026-12-15" in prompt
    assert "main project goal is to build" not in prompt
    mock_retriever.retrieve.assert_any_call(
        query=ScopeExtractionAgent.TARGET_QUERIES[0], project_name="Test_Project_A", top_k=1
    )

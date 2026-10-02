"""
Unit Tests for Blocker and Action Item Identification Agent (Gemini).

Tests cover:
1. Blocker extraction
2. Pending decision extraction
3. Action item extraction
4. Missing assignee handling
5. Missing deadline handling
6. Source/evidence preservation
7. Gemini API failure handling
8. Invalid structured output handling
9. Project isolation verification
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.blocker_agent import BlockerActionAgent, FALLBACK_NOT_SPECIFIED
from agents.schemas import (
    ActionItem,
    BlockerActionOutput,
    BlockerItem,
    PendingDecision,
    UnresolvedIssue,
)
from llm.base_provider import BaseLLMProvider


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    chunks = [
        {
            "chunk_id": "chunk_meeting_1",
            "text": "Sprint Review Notes: Payment gateway deployment is blocked waiting for security team approval. Decision pending on database migration tool. Action: Rahul to update unit test suite by Friday.",
            "metadata": {
                "source": "Sprint_Notes.txt",
                "document_id": "doc_303",
                "page": 1,
                "chunk_id": "chunk_meeting_1"
            }
        }
    ]
    retriever.retrieve.return_value = chunks
    retriever.retrieve_all.return_value = chunks
    return retriever


@pytest.fixture
def mock_gemini_provider():
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.return_value = {
        "blockers": [
            {
                "title": "Security Approval Blocker",
                "description": "Payment gateway deployment is blocked waiting for security team approval",
                "impact": "Delays production release",
                "status": "Active",
                "source": "Sprint_Notes.txt",
                "evidence": "Payment gateway deployment is blocked waiting for security team approval.",
                "document_name": "Sprint_Notes.txt",
                "document_id": "doc_303",
                "page": 1,
                "chunk_id": "chunk_meeting_1",
                "owner": "Security Team"
            }
        ],
        "pending_decisions": [
            {
                "title": "Database Migration Tool",
                "decision": "Selection of database migration tool",
                "owner": "Architecture Team",
                "status": "Pending",
                "source": "Sprint_Notes.txt",
                "evidence": "Decision pending on database migration tool.",
                "document_name": "Sprint_Notes.txt",
                "document_id": "doc_303",
                "page": 1,
                "chunk_id": "chunk_meeting_1"
            }
        ],
        "unresolved_issues": [
            {
                "title": "Database Migration Delay",
                "issue": "Unresolved DB migration pipeline compatibility",
                "status": "Open",
                "source": "Sprint_Notes.txt",
                "evidence": "Decision pending on database migration tool.",
                "document_name": "Sprint_Notes.txt",
                "document_id": "doc_303",
                "page": 1,
                "chunk_id": "chunk_meeting_1"
            }
        ],
        "action_items": [
            {
                "action": "Update unit test suite",
                "assignee": "Rahul",
                "deadline": "Friday",
                "status": "Open",
                "priority": "High",
                "source": "Sprint_Notes.txt",
                "evidence": "Rahul to update unit test suite by Friday.",
                "document_name": "Sprint_Notes.txt",
                "document_id": "doc_303",
                "page": 1,
                "chunk_id": "chunk_meeting_1"
            }
        ],
        "sources": ["Sprint_Notes.txt"]
    }
    return provider


def test_blocker_extraction(mock_gemini_provider, mock_retriever):
    """Test 1: Valid Blocker extraction from mock LLM output."""
    agent = BlockerActionAgent(provider=mock_gemini_provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_Blocker_Test")

    assert res["metadata"]["status"] == "Success"
    assert res["metadata"]["agent_name"] == "Blocker & Action Agent"
    assert res["metadata"]["provider"] == "Gemini"

    data = res["data"]
    assert len(data["blockers"]) == 1
    assert "security team approval" in data["blockers"][0]["description"]
    assert data["blockers"][0]["owner"] == "Security Team"


def test_pending_decision_extraction(mock_gemini_provider, mock_retriever):
    """Test 2: Pending Decision extraction."""
    agent = BlockerActionAgent(provider=mock_gemini_provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_Decision_Test")

    data = res["data"]
    assert len(data["pending_decisions"]) == 1
    assert "database migration" in data["pending_decisions"][0]["decision"]
    assert data["pending_decisions"][0]["status"] == "Pending"


def test_action_item_extraction(mock_gemini_provider, mock_retriever):
    """Test 3: Action Item extraction with assignee and deadline."""
    agent = BlockerActionAgent(provider=mock_gemini_provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_Action_Test")

    data = res["data"]
    assert len(data["action_items"]) == 1
    act = data["action_items"][0]

    assert act["action"] == "Update unit test suite"
    assert act["assignee"] == "Rahul"
    assert act["deadline"] == "Friday"
    assert act["priority"] == "High"


def test_missing_assignee_handling(mock_retriever):
    """Test 4: Missing assignee fallback to grounding text."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.return_value = {
        "blockers": [],
        "pending_decisions": [],
        "unresolved_issues": [],
        "action_items": [
            {
                "action": "Fix login bug",
                "assignee": "",
                "deadline": "Tomorrow",
                "status": "Open",
                "priority": "Medium",
                "source": "Sprint_Notes.txt",
                "evidence": "Fix login bug by Tomorrow."
            }
        ],
        "sources": ["Sprint_Notes.txt"]
    }

    agent = BlockerActionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_NoAssignee_Test")

    assert res["metadata"]["status"] == "Success"
    act = res["data"]["action_items"][0]
    assert act["assignee"] == FALLBACK_NOT_SPECIFIED


def test_missing_deadline_handling(mock_retriever):
    """Test 5: Missing deadline fallback to grounding text."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.return_value = {
        "blockers": [],
        "pending_decisions": [],
        "unresolved_issues": [],
        "action_items": [
            {
                "action": "Refactor auth pipeline",
                "assignee": "Alice",
                "deadline": "",
                "status": "Open",
                "priority": "Low",
                "source": "Sprint_Notes.txt",
                "evidence": "Alice to refactor auth pipeline."
            }
        ],
        "sources": ["Sprint_Notes.txt"]
    }

    agent = BlockerActionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_NoDeadline_Test")

    assert res["metadata"]["status"] == "Success"
    act = res["data"]["action_items"][0]
    assert act["deadline"] == FALLBACK_NOT_SPECIFIED


def test_source_and_evidence_preservation(mock_gemini_provider, mock_retriever):
    """Test 6: Source and evidence attribution preservation across items."""
    agent = BlockerActionAgent(provider=mock_gemini_provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_Preserve_Test")

    data = res["data"]
    blocker = data["blockers"][0]
    assert blocker["source"] == "Sprint_Notes.txt"
    assert blocker["document_id"] == "doc_303"
    assert blocker["page"] == 1
    assert blocker["chunk_id"] == "chunk_meeting_1"
    assert "security team approval" in blocker["evidence"]


def test_gemini_api_failure_handling(mock_retriever):
    """Test 7: Gemini API failure returns status Failed without crashing."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.side_effect = RuntimeError("Gemini API connection error")

    agent = BlockerActionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_GeminiFail_Test")

    assert res["metadata"]["status"] == "Failed"
    assert "Gemini API connection error" in res["metadata"]["error"]
    assert isinstance(res["data"], dict)
    assert res["data"]["blockers"] == []


def test_invalid_structured_output_handling(mock_retriever):
    """Test 8: Invalid structured output from LLM recovery."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_structured.side_effect = ValueError("Malformed JSON schema response")

    agent = BlockerActionAgent(provider=provider, retriever=mock_retriever)
    res = agent.run(project_id="Project_BadJSON_Test")

    assert res["metadata"]["status"] == "Failed"
    assert "Malformed JSON schema" in res["metadata"]["error"]


def test_project_isolation_verification(mock_gemini_provider):
    """Test 9: Project isolation - retriever queries filter by project_id."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.retrieve_all.return_value = []

    agent = BlockerActionAgent(provider=mock_gemini_provider, retriever=mock_retriever)
    agent.run(project_id="Isolated_Blocker_Workspace_777")

    mock_retriever.retrieve.assert_any_call(
        query=BlockerActionAgent.TARGET_QUERIES[0],
        project_name="Isolated_Blocker_Workspace_777",
        top_k=5,
    )
    mock_retriever.retrieve_all.assert_not_called()


def test_blocker_extraction_respects_top_k_and_similarity(mock_gemini_provider, mock_retriever):
    """Only the most similar chunks up to top_k reach the blocker model."""
    late_chunk = {
        "chunk_id": "chunk_late",
        "text": "Action: Priya will finish API validation by 2026-10-15.",
        "distance": 0.1,
        "metadata": {"source": "Tasks.csv", "chunk_id": "chunk_late"},
    }
    mock_retriever.retrieve.return_value = mock_retriever.retrieve.return_value + [late_chunk]

    agent = BlockerActionAgent(provider=mock_gemini_provider, retriever=mock_retriever)
    result = agent.run(project_id="Blocker_Project", top_k=1)

    assert result["metadata"]["status"] == "Success"
    prompt = mock_gemini_provider.generate_structured.call_args.kwargs["prompt"]
    assert "Priya will finish API validation by 2026-10-15" in prompt
    assert "Sprint Review Notes" not in prompt
    mock_retriever.retrieve.assert_any_call(
        query=BlockerActionAgent.TARGET_QUERIES[0], project_name="Blocker_Project", top_k=1
    )

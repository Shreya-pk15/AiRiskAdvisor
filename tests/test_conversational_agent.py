"""
Unit Tests for Conversational Project Intelligence Assistant (Milestone 3 — Part 3).

Tests cover:
1. Basic project question
2. Risk question
3. Blocker question
4. Health question
5. Follow-up question (conversation history & pronoun resolution)
6. Question with no supporting evidence (strict grounding fallback)
7. Project isolation (workspace boundary enforcement)
8. LLM failure handling (graceful fallback)
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.conversational_agent import ConversationalProjectAssistant, FALLBACK_NO_INFO
from agents.schemas import ConversationalResponse
from llm.base_provider import BaseLLMProvider


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    # Default mock chunks
    retriever.retrieve.return_value = [
        {
            "id": "chunk_1",
            "text": "The project aim is to build an AI Risk Advisor platform with automated risk detection.",
            "source": "Project_Charter.pdf",
            "metadata": {"source": "Project_Charter.pdf", "page": 1, "project_name": "TestProject"},
            "distance": 0.12,
        },
        {
            "id": "chunk_2",
            "text": "Backend API development is assigned to Alice, while frontend UI is led by Bob.",
            "source": "Team_Roster.docx",
            "metadata": {"source": "Team_Roster.docx", "page": 2, "project_name": "TestProject"},
            "distance": 0.18,
        }
    ]
    return retriever


@pytest.fixture
def mock_llm_provider():
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_text.return_value = (
        "Alice is responsible for the backend API development, as specified in the team roster.\n\n"
        "[Source: Team_Roster.docx, Page 2]"
    )
    return provider


@pytest.fixture
def sample_intelligence():
    return {
        "scope_data": {
            "project_goals": [{"goal": "Build AI Risk Advisor platform", "source": "Project_Charter.pdf"}],
            "deliverables": [{"name": "Ingestion Pipeline"}, {"name": "Health Dashboard"}],
            "milestones": [{"name": "Milestone 1", "target_date": "2026-10-01"}],
        },
        "risk_data": {
            "delivery_status": "At Risk",
            "risks": [
                {
                    "risk_id": "R-01",
                    "description": "Third-party payment API deprecation",
                    "severity": "High",
                    "recommended_action": "Upgrade to v2 API before end of month",
                },
                {
                    "risk_id": "R-02",
                    "description": "Database memory leak under peak load",
                    "severity": "Medium",
                    "recommended_action": "Optimize connection pooling",
                }
            ],
        },
        "blocker_data": {
            "blockers": [
                {
                    "description": "Pending OAuth2 security approval from compliance team",
                    "status": "Active",
                }
            ],
            "unresolved_issues": [
                {"issue": "Flaky integration tests in CI pipeline"}
            ],
            "action_items": [
                {"action": "Submit compliance ticket", "assignee": "Charlie"}
            ],
            "pending_decisions": [],
        },
        "health_data": {
            "overall_score": 68.0,
            "classification": "Moderate",
            "confidence": "High",
            "key_factors": [
                "1 high-severity risk identified",
                "1 active blocker recorded",
            ],
            "recommendations": [
                "Expedite OAuth2 security compliance approval",
            ],
        }
    }


# -----------------------------------------------------------------------------
# Test 1: Basic Project Question
# -----------------------------------------------------------------------------
def test_basic_project_question(mock_retriever, mock_llm_provider):
    """Verify basic grounded Q&A retrieves chunks, calls LLM, and formats citations."""
    assistant = ConversationalProjectAssistant(provider=mock_llm_provider, retriever=mock_retriever)
    
    response = assistant.ask(
        question="Who is responsible for the backend?",
        project_id="TestProject",
    )

    assert isinstance(response, ConversationalResponse)
    assert response.status == "Success"
    assert "Alice" in response.answer
    assert len(response.sources) > 0
    assert any("Team_Roster.docx" in s for s in response.sources)
    mock_retriever.retrieve.assert_called_once()
    mock_llm_provider.generate_text.assert_called_once()


# -----------------------------------------------------------------------------
# Test 2: Risk Question
# -----------------------------------------------------------------------------
def test_risk_question(mock_retriever, sample_intelligence):
    """Verify asking about risks integrates existing Risk Agent intelligence and document context."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_text.return_value = (
        "Our biggest risk is the third-party payment API deprecation (High severity). "
        "Additionally, there is a database memory leak under peak load.\n\n"
        "[Source: Risk Register]"
    )
    assistant = ConversationalProjectAssistant(provider=provider, retriever=mock_retriever)

    response = assistant.ask(
        question="What are our biggest risks?",
        project_id="TestProject",
        risk_data=sample_intelligence["risk_data"],
    )

    assert response.status == "Success"
    assert "payment API deprecation" in response.answer or "High severity" in response.answer
    # Check that prompt sent to LLM included the structured risk intelligence
    call_args = provider.generate_text.call_args
    prompt_sent = call_args[1].get("prompt") or call_args[0][0]
    assert "DELIVERY STATUS: At Risk" in prompt_sent
    assert "Third-party payment API deprecation" in prompt_sent


# -----------------------------------------------------------------------------
# Test 3: Blocker Question
# -----------------------------------------------------------------------------
def test_blocker_question(mock_retriever, sample_intelligence):
    """Verify asking about blockers incorporates Blocker Agent intelligence."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_text.return_value = (
        "The project is currently blocked by pending OAuth2 security approval from the compliance team.\n\n"
        "[Source: Blocker Tracker]"
    )
    assistant = ConversationalProjectAssistant(provider=provider, retriever=mock_retriever)

    response = assistant.ask(
        question="What is currently blocking the project?",
        project_id="TestProject",
        blocker_data=sample_intelligence["blocker_data"],
    )

    assert response.status == "Success"
    assert "OAuth2" in response.answer
    call_args = provider.generate_text.call_args
    prompt_sent = call_args[1].get("prompt") or call_args[0][0]
    assert "ACTIVE BLOCKERS (1):" in prompt_sent
    assert "Pending OAuth2 security approval" in prompt_sent


# -----------------------------------------------------------------------------
# Test 4: Health Question
# -----------------------------------------------------------------------------
def test_health_question(mock_retriever, sample_intelligence):
    """Verify asking 'Are we on track?' includes Project Health score and status without recalculating."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_text.return_value = (
        "Current Health: 68/100\n"
        "Status: Moderate\n"
        "Delivery Status: At Risk\n\n"
        "Main factors:\n"
        "- 1 high-severity risk\n"
        "- 1 active blocker\n\n"
        "[Source: Project Health Assessment]"
    )
    assistant = ConversationalProjectAssistant(provider=provider, retriever=mock_retriever)

    response = assistant.ask(
        question="Are we on track?",
        project_id="TestProject",
        risk_data=sample_intelligence["risk_data"],
        health_data=sample_intelligence["health_data"],
    )

    assert response.status == "Success"
    assert "68/100" in response.answer or "Moderate" in response.answer
    call_args = provider.generate_text.call_args
    prompt_sent = call_args[1].get("prompt") or call_args[0][0]
    assert "PROJECT HEALTH SCORE: 68.0/100" in prompt_sent
    assert "Classification: Moderate" in prompt_sent


# -----------------------------------------------------------------------------
# Test 5: Follow-Up Question with Pronoun Resolution
# -----------------------------------------------------------------------------
def test_followup_question(mock_retriever):
    """Verify assistant maintains conversation history and passes previous context on follow-up."""
    provider = MagicMock(spec=BaseLLMProvider)
    
    # Turn 1
    provider.generate_text.return_value = "The major risks identified are payment API deprecation and memory leaks."
    assistant = ConversationalProjectAssistant(provider=provider, retriever=mock_retriever)
    
    resp1 = assistant.ask(
        question="What are our biggest risks?",
        project_id="TestProject",
    )
    assert resp1.status == "Success"
    assert len(assistant.get_history()) == 2

    # Turn 2: Follow-up using pronoun "them"
    provider.generate_text.return_value = "Alice is assigned to investigate and fix the memory leak, and Charlie owns the API migration."
    resp2 = assistant.ask(
        question="Who is responsible for fixing them?",
        project_id="TestProject",
    )
    assert resp2.status == "Success"
    assert len(assistant.get_history()) == 4

    # Verify that conversation history was included in the LLM prompt
    call_args = provider.generate_text.call_args
    prompt_sent = call_args[1].get("prompt") or call_args[0][0]
    assert "What are our biggest risks?" in prompt_sent
    assert "Who is responsible for fixing them?" in prompt_sent
    # Verify that search query was augmented to resolve pronoun
    retriever_calls = mock_retriever.retrieve.call_args_list
    assert len(retriever_calls) == 2
    second_query = retriever_calls[1][1].get("query") or retriever_calls[1][0][0]
    assert "What are our biggest risks?" in second_query or "risks" in second_query.lower()


# -----------------------------------------------------------------------------
# Test 6: Question with No Supporting Evidence
# -----------------------------------------------------------------------------
def test_question_with_no_supporting_evidence():
    """Verify asking about completely unsupported facts yields grounded fallback without hallucination."""
    empty_retriever = MagicMock()
    empty_retriever.retrieve.return_value = []
    
    assistant = ConversationalProjectAssistant(retriever=empty_retriever)

    # When no chunks and no prior intelligence are present
    response = assistant.ask(
        question="What is the CEO's favorite car brand?",
        project_id="TestProject",
        scope_data=None,
        risk_data=None,
        blocker_data=None,
        health_data=None,
    )

    assert response.status == "Success"
    assert FALLBACK_NO_INFO in response.answer


def test_unrelated_question_with_weak_retrieval_is_refused():
    """Nearest-neighbor chunks must not make an unrelated question answerable."""
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "text": "Project goal: build an auction platform.",
            "metadata": {"source": "Project_Charter.pdf"},
            "distance": 1.98,
        }
    ]
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_text.return_value = "Paris is the capital of France."
    assistant = ConversationalProjectAssistant(provider=provider, retriever=retriever)

    response = assistant.ask(
        question="What is the capital of France?",
        project_id="TestProject",
    )

    assert response.status == "Success"
    assert response.answer == FALLBACK_NO_INFO
    assert response.sources == []
    provider.generate_text.assert_not_called()


def test_relevant_question_within_retrieval_cutoff_is_answered():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "text": "The frontend UI is led by Bob.",
            "metadata": {"source": "Team_Roster.docx"},
            "distance": 1.54,
        }
    ]
    provider = MagicMock(spec=BaseLLMProvider)
    provider.generate_text.return_value = "Bob leads the frontend UI."
    assistant = ConversationalProjectAssistant(provider=provider, retriever=retriever)

    response = assistant.ask(
        question="Who is assigned to frontend UI?",
        project_id="TestProject",
    )

    assert response.answer == "Bob leads the frontend UI."
    provider.generate_text.assert_called_once()


# -----------------------------------------------------------------------------
# Test 7: Project Isolation
# -----------------------------------------------------------------------------
def test_project_isolation():
    """Verify that retrieval strictly queries ChromaDB using the designated project_id."""
    isolated_retriever = MagicMock()
    isolated_retriever.retrieve.return_value = []
    
    assistant = ConversationalProjectAssistant(retriever=isolated_retriever)

    assistant.ask(question="What are our goals?", project_id="Project_Alpha")
    isolated_retriever.retrieve.assert_called_with(
        query="What are our goals?",
        project_name="Project_Alpha",
        top_k=5,
    )

    assistant.ask(question="What are our goals?", project_id="Project_Beta")
    isolated_retriever.retrieve.assert_called_with(
        query="What are our goals?",
        project_name="Project_Beta",
        top_k=5,
    )


# -----------------------------------------------------------------------------
# Test 8: LLM Failure Handling
# -----------------------------------------------------------------------------
def test_llm_failure(mock_retriever, sample_intelligence):
    """Verify assistant handles LLM exception gracefully without crashing."""
    failing_provider = MagicMock(spec=BaseLLMProvider)
    failing_provider.generate_text.side_effect = RuntimeError("API rate limit exceeded")

    assistant = ConversationalProjectAssistant(
        provider=failing_provider,
        retriever=mock_retriever,
    )

    # Should not throw exception; should fallback safely
    response = assistant.ask(
        question="Are we on track?",
        project_id="TestProject",
        health_data=sample_intelligence["health_data"],
        risk_data=sample_intelligence["risk_data"],
    )

    assert isinstance(response, ConversationalResponse)
    assert response.answer is not None
    # Offline fallback should capture health score or fallback message
    assert "HEALTH" in response.answer or FALLBACK_NO_INFO in response.answer

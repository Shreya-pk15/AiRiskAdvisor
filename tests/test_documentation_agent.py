"""
Unit Tests for Documentation Generation Agent (Milestone 3 — Part 1).

Tests cover:
1. User story generation from actual scope & deliverables
2. Risk register generation from risk detection outputs
3. Action item generation from blocker outputs
4. Missing fields handling ("Not specified in the available project documents.")
5. Unsupported / invented information prevention (strict grounding)
6. Pydantic validation and schema error handling
7. LLM failure handling and graceful recovery
8. Project isolation verification
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.documentation_agent import DocumentationAgent, FALLBACK_NOT_SPECIFIED
from agents.schemas import (
    ActionItem,
    ActionItemsOutput,
    DocumentationGenerationResult,
    RiskRegisterItem,
    RiskRegisterOutput,
    UserStoriesOutput,
    UserStory,
)
from llm.base_provider import BaseLLMProvider


@pytest.fixture
def sample_scope_data():
    return {
        "project_goals": [
            {
                "goal": "Build an Automated Project Intelligence Platform",
                "description": "Unified platform to analyze risks and sprint health",
                "source": "Project_Proposal.pdf",
                "evidence": "Primary goal: Build an Automated Project Intelligence Platform.",
            }
        ],
        "deliverables": [
            {
                "name": "Document Ingestion Module",
                "description": "Ingest PDF, DOCX, CSV, TXT, and XLSX files",
                "source": "SRS_Document.docx",
                "evidence": "Deliverable: Ingest PDF, DOCX, CSV, TXT, and XLSX files.",
            },
            {
                "name": "Vector Retrieval Architecture",
                "description": "ChromaDB vector store with semantic embeddings",
                "source": "SRS_Document.docx",
                "evidence": "Deliverable: Vector Retrieval Architecture.",
            },
        ],
        "milestones": [
            {
                "name": "Milestone 1 - Ingestion & RAG",
                "target_date": "2026-10-01",
                "status": "Completed",
                "source": "Project_Proposal.pdf",
                "evidence": "Milestone 1 deadline: 2026-10-01.",
            }
        ],
        "timeline": [
            {
                "label": "Sprint 1",
                "value": "2026-09-15 to 2026-10-01",
                "source": "Project_Proposal.pdf",
                "evidence": "Sprint 1 timeframe.",
            }
        ],
        "sources": ["Project_Proposal.pdf", "SRS_Document.docx"],
    }


@pytest.fixture
def sample_risk_data():
    return {
        "risks": [
            {
                "risk_id": "RISK-01",
                "category": "Dependency",
                "description": "Payment integration gateway credentials are not issued.",
                "severity": "High",
                "evidence": "Payment gateway provider has not issued credentials.",
                "source": "Sprint_Meeting_Notes.txt",
                "recommended_action": "Escalate to gateway vendor account manager.",
            },
            {
                "risk_id": "RISK-02",
                "category": "Schedule",
                "description": "Testing phase compressed due to API delays.",
                "severity": "Medium",
                "evidence": "Testing schedule may be shortened by 3 days.",
                "source": "Sprint_Meeting_Notes.txt",
                "recommended_action": "Begin test automation scripts ahead of time.",
            },
        ],
        "delivery_status": "At Risk",
        "delivery_forecast": {
            "status": "At Risk",
            "reason": "Payment credentials dependency pending.",
            "forecast_analysis": "Integration testing will be delayed if credentials not received.",
            "concerns": ["Vendor responsiveness", "Tight sprint deadline"],
            "recommended_actions": ["Escalate credential request"],
        },
        "sources": ["Sprint_Meeting_Notes.txt"],
    }


@pytest.fixture
def sample_blocker_data():
    return {
        "blockers": [
            {
                "description": "Production API credentials not received from payment provider.",
                "impact": "Blocks payment module end-to-end testing.",
                "status": "Active",
                "source": "Sprint_Meeting_Notes.txt",
                "evidence": "Cannot proceed until credentials arrive.",
                "owner": "John Doe",
            }
        ],
        "pending_decisions": [
            {
                "decision": "Choose between ChromaDB local vs hosted vector database.",
                "owner": "Sarah",
                "status": "Pending",
                "source": "Architecture_Notes.docx",
                "evidence": "Decision on vector DB deployment topology pending.",
            }
        ],
        "unresolved_issues": [
            {
                "issue": "Excel XLSX file parser memory spike with large sheets.",
                "status": "Open",
                "source": "Defect_Tracker.csv",
                "evidence": "Defect #102: XLSX memory usage > 500MB.",
            }
        ],
        "action_items": [
            {
                "action": "Follow up with payment gateway vendor support.",
                "assignee": "John Doe",
                "deadline": "2026-09-30",
                "priority": "High",
                "status": "Open",
                "source": "Sprint_Meeting_Notes.txt",
                "evidence": "John to follow up with vendor by Wednesday.",
            }
        ],
        "sources": ["Sprint_Meeting_Notes.txt", "Architecture_Notes.docx", "Defect_Tracker.csv"],
    }


@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=BaseLLMProvider)
    return provider


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "text": "Project deliverables include Document Ingestion Pipeline and ChromaDB store.",
            "metadata": {"source": "Proposal.pdf"},
        }
    ]
    return retriever


# =============================================================================
# 1. User Story Generation Tests
# =============================================================================

def test_generate_user_stories_success(sample_scope_data, mock_provider, mock_retriever):
    """Verify user story generation creates valid UserStory models with MoSCoW priorities and acceptance criteria."""
    mock_provider.generate_structured.return_value = {
        "user_stories": [
            {
                "story_id": "US-01",
                "user_story": "As a project user, I want to upload project documents so that the system can analyze my project information.",
                "description": "Enables multi-format ingestion of project artifacts.",
                "priority": "Must Have",
                "acceptance_criteria": [
                    "User can upload PDF",
                    "User can upload DOCX",
                    "Uploaded document is processed successfully",
                ],
                "dependencies": "Document Ingestion Module",
                "source": "SRS_Document.docx",
                "evidence": "Deliverable: Ingest PDF, DOCX, CSV, TXT, and XLSX files.",
            }
        ],
        "sources": ["SRS_Document.docx"],
    }

    agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
    result = agent.generate_user_stories(project_id="TestProject", scope_data=sample_scope_data)

    assert result["metadata"]["status"] == "Success"
    assert result["metadata"]["document_type"] == "User Stories"
    data = result["data"]
    assert len(data["user_stories"]) == 1

    story = data["user_stories"][0]
    assert story["story_id"] == "US-01"
    assert story["priority"] == "Must Have"
    assert len(story["acceptance_criteria"]) == 3
    assert story["dependencies"] == "Document Ingestion Module"
    assert "SRS_Document.docx" in story["source"]
    assert "Deliverable" in story["evidence"]


# =============================================================================
# 2. Risk Register Generation Tests
# =============================================================================

def test_generate_risk_register_success(sample_risk_data, mock_provider, mock_retriever):
    """Verify risk register generation creates valid RiskRegisterItem models with grounded mitigations and ratings."""
    mock_provider.generate_structured.return_value = {
        "risk_register": [
            {
                "risk_id": "R-01",
                "risk_description": "Payment integration gateway credentials are not issued delaying testing.",
                "category": "Schedule / Technical",
                "probability": "Medium",
                "impact": "High",
                "severity": "High",
                "mitigation": "Escalate to gateway vendor account manager.",
                "owner": "Not specified in the available project documents.",
                "status": "Open",
                "source": "Sprint_Meeting_Notes.txt",
                "evidence": "Payment gateway provider has not issued credentials.",
            }
        ],
        "sources": ["Sprint_Meeting_Notes.txt"],
    }

    agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
    result = agent.generate_risk_register(project_id="TestProject", risk_data=sample_risk_data)

    assert result["metadata"]["status"] == "Success"
    assert result["metadata"]["document_type"] == "Risk Register"
    data = result["data"]
    assert len(data["risk_register"]) == 1

    risk = data["risk_register"][0]
    assert risk["risk_id"] == "R-01"
    assert risk["category"] == "Schedule / Technical"
    assert risk["severity"] == "High"
    assert risk["owner"] == FALLBACK_NOT_SPECIFIED
    assert risk["status"] == "Open"
    assert "Sprint_Meeting_Notes.txt" in risk["source"]


# =============================================================================
# 3. Action Item Generation Tests
# =============================================================================

def test_generate_action_items_success(sample_blocker_data, mock_provider, mock_retriever):
    """Verify action item generation creates valid ActionItem models with assignees, priorities, and related blockers."""
    mock_provider.generate_structured.return_value = {
        "action_items": [
            {
                "action_id": "ACT-01",
                "action": "Follow up with payment gateway vendor support to obtain production credentials.",
                "assignee": "John Doe",
                "priority": "High",
                "deadline": "2026-09-30",
                "status": "Open",
                "related_blocker_risk": "Production API credentials not received.",
                "source": "Sprint_Meeting_Notes.txt",
                "evidence": "John to follow up with vendor by Wednesday.",
            }
        ],
        "sources": ["Sprint_Meeting_Notes.txt"],
    }

    agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
    result = agent.generate_action_items(project_id="TestProject", blocker_data=sample_blocker_data)

    assert result["metadata"]["status"] == "Success"
    assert result["metadata"]["document_type"] == "Action Items"
    data = result["data"]
    assert len(data["action_items"]) == 1

    item = data["action_items"][0]
    assert item["action_id"] == "ACT-01"
    assert item["assignee"] == "John Doe"
    assert item["priority"] == "High"
    assert item["deadline"] == "2026-09-30"
    assert item["related_blocker_risk"] == "Production API credentials not received."


def test_generate_action_items_refreshes_empty_cached_blockers(
    sample_blocker_data, mock_provider, mock_retriever
):
    """An empty cached result should trigger fresh blocker extraction before generation."""
    mock_provider.generate_structured.return_value = {
        "action_items": [
            {
                "action_id": "ACT-01",
                "action": "Follow up with payment gateway vendor support.",
                "assignee": "John Doe",
                "priority": "High",
                "deadline": "2026-09-30",
                "status": "Open",
                "related_blocker_risk": "Payment API credentials pending.",
                "source": "Sprint_Meeting_Notes.txt",
                "evidence": "John to follow up with vendor by Wednesday.",
            }
        ],
        "sources": [],
    }
    cached_empty = {
        "action_items": [],
        "blockers": [],
        "pending_decisions": [],
        "unresolved_issues": [],
    }

    with patch("agents.documentation_agent.BlockerActionAgent") as mock_blocker_agent:
        mock_blocker_agent.return_value.run.return_value = {"data": sample_blocker_data}
        agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
        result = agent.generate_action_items(
            project_id="TestProject", blocker_data=cached_empty
        )

    mock_blocker_agent.return_value.run.assert_called_once_with(project_id="TestProject", top_k=5)
    assert result["metadata"]["status"] == "Success"
    assert result["data"]["action_items"][0]["assignee"] == "John Doe"


# =============================================================================
# 4. Missing Fields Handling Tests
# =============================================================================

def test_missing_fields_handling(sample_risk_data, sample_blocker_data, mock_provider, mock_retriever):
    """Verify that unmentioned fields strictly default to 'Not specified in the available project documents.'"""
    mock_provider.generate_structured.return_value = {
        "risk_register": [
            {
                "risk_id": "R-01",
                "risk_description": "General latency risk.",
                "category": "Technical",
                "probability": "",
                "impact": "",
                "severity": "Medium",
                "mitigation": "",
                "owner": "",
                "status": "",
                "source": "",
                "evidence": "",
            }
        ],
        "sources": [],
    }

    agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
    result = agent.generate_risk_register(project_id="TestProject", risk_data=sample_risk_data)

    risk = result["data"]["risk_register"][0]
    assert risk["owner"] == FALLBACK_NOT_SPECIFIED
    assert risk["probability"] == FALLBACK_NOT_SPECIFIED
    assert risk["impact"] == FALLBACK_NOT_SPECIFIED
    assert risk["mitigation"] == FALLBACK_NOT_SPECIFIED
    assert risk["source"] == FALLBACK_NOT_SPECIFIED
    assert risk["evidence"] == FALLBACK_NOT_SPECIFIED


# =============================================================================
# 5. Unsupported / Invented Information Prevention (Strict Grounding)
# =============================================================================

def test_unsupported_invented_info_grounding(mock_provider, mock_retriever):
    """Verify that when no deliverables or risks exist, the agent returns empty data and does not invent documents."""
    empty_scope = {"project_goals": [], "deliverables": [], "milestones": []}
    agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
    res_stories = agent.generate_user_stories(project_id="TestProject", scope_data=empty_scope)

    assert res_stories["metadata"]["status"] == "Success"
    assert res_stories["data"]["user_stories"] == []
    assert "No scope goals" in res_stories["metadata"].get("note", "")

    empty_risks = {"risks": []}
    res_risks = agent.generate_risk_register(project_id="TestProject", risk_data=empty_risks)
    assert res_risks["metadata"]["status"] == "Success"
    assert res_risks["data"]["risk_register"] == []
    assert "No risks identified" in res_risks["metadata"].get("note", "")


# =============================================================================
# 6. Pydantic Validation Error Handling
# =============================================================================

def test_pydantic_validation_error(sample_scope_data, mock_provider, mock_retriever):
    """Verify that malformed LLM response is caught by Pydantic validation without crashing."""
    # Invalid structure missing required fields or having wrong types
    mock_provider.generate_structured.return_value = {
        "user_stories": "this-should-be-a-list-not-a-string"
    }

    agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
    result = agent.generate_user_stories(project_id="TestProject", scope_data=sample_scope_data)

    assert result["metadata"]["status"] == "Failed"
    assert "Invalid structured output" in result["metadata"]["error"] or "validation" in result["metadata"]["error"].lower()
    assert result["data"]["user_stories"] == []


# =============================================================================
# 7. LLM Failure Handling
# =============================================================================

def test_llm_failure_handling(sample_risk_data, mock_provider, mock_retriever):
    """Verify that provider exception (e.g. rate limit, timeout) is caught gracefully with status='Failed'."""
    mock_provider.generate_structured.side_effect = RuntimeError("Gemini API connection timed out after 30 seconds")

    agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
    result = agent.generate_risk_register(project_id="TestProject", risk_data=sample_risk_data)

    assert result["metadata"]["status"] == "Failed"
    assert "Gemini API connection timed out" in result["metadata"]["error"]
    assert result["data"]["risk_register"] == []


# =============================================================================
# 8. Project Isolation Verification
# =============================================================================

def test_project_isolation(mock_provider, mock_retriever):
    """Verify that when scope_data is not provided, ScopeExtractionAgent is invoked strictly with the project_id."""
    with patch("agents.documentation_agent.ScopeExtractionAgent") as MockScopeAgent:
        mock_scope_instance = MagicMock()
        mock_scope_instance.run.return_value = {
            "data": {
                "project_goals": [{"goal": "Goal A", "source": "A.pdf", "evidence": "evidence A"}],
                "deliverables": [{"name": "Deliverable A", "source": "A.pdf", "evidence": "evidence A"}],
                "milestones": [],
                "timeline": [],
                "sources": ["A.pdf"],
            }
        }
        MockScopeAgent.return_value = mock_scope_instance

        mock_provider.generate_structured.return_value = {
            "user_stories": [
                {
                    "story_id": "US-01",
                    "user_story": "As a user, I want Deliverable A so that Goal A is achieved.",
                    "priority": "Must Have",
                    "acceptance_criteria": ["Criteria 1"],
                    "source": "A.pdf",
                    "evidence": "evidence A",
                }
            ],
            "sources": ["A.pdf"],
        }

        agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
        res = agent.generate_user_stories(project_id="Workspace_Tenant_99", scope_data=None)

        # Assert ScopeExtractionAgent was run specifically with project_id="Workspace_Tenant_99"
        mock_scope_instance.run.assert_called_once_with(project_id="Workspace_Tenant_99", top_k=5)
        assert res["metadata"]["status"] == "Success"


# =============================================================================
# 9. Generate All Documents Test
# =============================================================================

def test_generate_all_combines_outputs(
    sample_scope_data, sample_risk_data, sample_blocker_data, mock_provider, mock_retriever
):
    """Verify that generate_all aggregates User Stories, Risk Register, and Action Items into DocumentationGenerationResult."""
    def fake_generate_structured(prompt, response_schema, **kwargs):
        if response_schema == UserStoriesOutput:
            return {
                "user_stories": [
                    {
                        "story_id": "US-01",
                        "user_story": "As a user, I want ingestion so that files are indexed.",
                        "priority": "Must Have",
                        "acceptance_criteria": ["Test AC"],
                        "dependencies": "None",
                        "source": "SRS.docx",
                        "evidence": "Evidence text",
                    }
                ],
                "sources": ["SRS.docx"],
            }
        elif response_schema == RiskRegisterOutput:
            return {
                "risk_register": [
                    {
                        "risk_id": "R-01",
                        "risk_description": "API latency risk",
                        "category": "Technical",
                        "probability": "Low",
                        "impact": "Medium",
                        "severity": "Medium",
                        "mitigation": "Caching",
                        "owner": "Not specified in the available project documents.",
                        "status": "Open",
                        "source": "SRS.docx",
                        "evidence": "Evidence text",
                    }
                ],
                "sources": ["SRS.docx"],
            }
        elif response_schema == ActionItemsOutput:
            return {
                "action_items": [
                    {
                        "action_id": "ACT-01",
                        "action": "Implement redis cache",
                        "assignee": "Alice",
                        "priority": "High",
                        "deadline": "2026-10-05",
                        "status": "Open",
                        "related_blocker_risk": "API latency",
                        "source": "SRS.docx",
                        "evidence": "Evidence text",
                    }
                ],
                "sources": ["SRS.docx"],
            }
        return {}

    mock_provider.generate_structured.side_effect = fake_generate_structured

    agent = DocumentationAgent(provider=mock_provider, retriever=mock_retriever)
    all_res = agent.generate_all(
        project_id="CombinedWorkspace",
        scope_data=sample_scope_data,
        risk_data=sample_risk_data,
        blocker_data=sample_blocker_data,
    )

    assert all_res["metadata"]["status"] == "Success"
    assert all_res["metadata"]["document_type"] == "All Documents"
    data = all_res["data"]
    assert len(data["user_stories"]) == 1
    assert len(data["risk_register"]) == 1
    assert len(data["action_items"]) == 1
    assert "SRS.docx" in data["sources"]

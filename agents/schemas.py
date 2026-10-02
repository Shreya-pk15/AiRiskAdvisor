"""
Pydantic Schemas for Project Intelligence Agents.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field



# -----------------------------------------------------------------------------
# Milestone 2 — Agent 1: Scope & Deliverable Extraction Schemas
# -----------------------------------------------------------------------------

class ProjectGoal(BaseModel):
    goal: str = Field(description="Objective, business/technical goal, or expected outcome")
    description: Optional[str] = Field(default=None, description="Detailed explanation of the goal if available")
    source: str = Field(default="Not specified in the available project documents.", description="Document source name or file path")
    document_name: Optional[str] = Field(default=None, description="Name of source document")
    document_id: Optional[str] = Field(default=None, description="ID of source document")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")
    evidence: str = Field(default="Not specified in the available project documents.", description="Direct snippet or quote from document")


class DeliverableItem(BaseModel):
    name: str = Field(description="Name of feature, module, system, report, API, or document output")
    description: Optional[str] = Field(default=None, description="Description of the deliverable output")
    source: str = Field(default="Not specified in the available project documents.", description="Document source name")
    document_name: Optional[str] = Field(default=None, description="Name of source document")
    document_id: Optional[str] = Field(default=None, description="ID of source document")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")
    evidence: str = Field(default="Not specified in the available project documents.", description="Direct snippet or quote from document")


class MilestoneItem(BaseModel):
    name: str = Field(description="Milestone name, phase, or iteration")
    description: Optional[str] = Field(default=None, description="Description of milestone objectives")
    target_date: Optional[str] = Field(default=None, description="Target date or deadline")
    status: Optional[str] = Field(default=None, description="Status if available e.g. Planned, In Progress, Completed")
    source: str = Field(default="Not specified in the available project documents.", description="Document source name")
    document_name: Optional[str] = Field(default=None, description="Name of source document")
    document_id: Optional[str] = Field(default=None, description="ID of source document")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")
    evidence: str = Field(default="Not specified in the available project documents.", description="Direct snippet or quote from document")


class TimelineItem(BaseModel):
    label: str = Field(description="Timeline reference e.g., Start Date, End Date, Deadline, Sprint 1")
    value: str = Field(description="Date, duration, or timeframe string")
    start_date: Optional[str] = Field(default=None, description="Start date if explicitly mentioned")
    end_date: Optional[str] = Field(default=None, description="End date if explicitly mentioned")
    deadline: Optional[str] = Field(default=None, description="Deadline if explicitly mentioned")
    source: str = Field(default="Not specified in the available project documents.", description="Document source name")
    document_name: Optional[str] = Field(default=None, description="Name of source document")
    document_id: Optional[str] = Field(default=None, description="ID of source document")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")
    evidence: str = Field(default="Not specified in the available project documents.", description="Direct snippet or quote from document")


class ResponsibilityItem(BaseModel):
    person: str = Field(description="Person, team, or role assigned")
    responsibility: str = Field(description="Assigned task, role, or deliverable responsibility")
    related_deliverable: Optional[str] = Field(default=None, description="Associated feature, module, or deliverable name")
    source: str = Field(default="Not specified in the available project documents.", description="Document source name")
    document_name: Optional[str] = Field(default=None, description="Name of source document")
    document_id: Optional[str] = Field(default=None, description="ID of source document")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")
    evidence: str = Field(default="Not specified in the available project documents.", description="Direct snippet or quote from document")


class ScopeExtractionOutput(BaseModel):
    project_goals: List[ProjectGoal] = Field(default_factory=list, description="Extracted project goals")
    deliverables: List[DeliverableItem] = Field(default_factory=list, description="Extracted project deliverables")
    milestones: List[MilestoneItem] = Field(default_factory=list, description="Extracted project milestones")
    timeline: List[TimelineItem] = Field(default_factory=list, description="Extracted timeline details")
    responsibilities: List[ResponsibilityItem] = Field(default_factory=list, description="Extracted team responsibilities")
    sources: List[str] = Field(default_factory=list, description="List of document sources referenced")


ScopeResult = ScopeExtractionOutput


# -----------------------------------------------------------------------------
# Milestone 2 — Agent 2: Risk Detection & Delivery Forecasting Schemas
# -----------------------------------------------------------------------------

RiskCategory = Literal["Schedule", "Dependency", "Resource", "Technical", "Quality", "Planning", "Delivery"]
RiskSeverity = Literal["Low", "Medium", "High"]
DeliveryStatusType = Literal["On Track", "At Risk", "Delayed", "Insufficient Data"]


class RiskItem(BaseModel):
    risk_id: str = Field(default_factory=lambda: f"RISK-{uuid.uuid4().hex[:6].upper()}", description="Unique identifier for the risk")
    category: RiskCategory = Field(description="Risk category classification")
    description: str = Field(description="Clear explanation of the potential risk")
    severity: RiskSeverity = Field(description="Evidence-justified severity rating")
    evidence: str = Field(default="Not specified in the available project documents.", description="Exact document snippet supporting risk")
    source: str = Field(default="Not specified in the available project documents.", description="Source document filename")
    document_name: Optional[str] = Field(default=None, description="Source document filename")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")
    recommended_action: Optional[str] = Field(default="Not specified", description="Recommended mitigation action")


class DeliveryForecast(BaseModel):
    status: DeliveryStatusType = Field(default="Insufficient Data", description="Evidence-based delivery status")
    reason: str = Field(default="No documented evidence available.", description="Summary justification for status")
    forecast_analysis: str = Field(default="No forward-looking analysis available due to lack of evidence.", description="Forward-looking analysis of delivery challenges")
    evidence: List[str] = Field(default_factory=list, description="Documented evidence points")
    concerns: List[str] = Field(default_factory=list, description="List of documented concerns or blockers")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended next steps")


class RiskDetectionOutput(BaseModel):
    risks: List[RiskItem] = Field(default_factory=list, description="List of evidence-backed risks")
    delivery_status: DeliveryStatusType = Field(default="Insufficient Data", description="Overall evidence-based status")
    delivery_forecast: DeliveryForecast = Field(default_factory=DeliveryForecast, description="Forward-looking delivery forecast")
    sources: List[str] = Field(default_factory=list, description="Referenced document sources")


RiskResult = RiskDetectionOutput


# -----------------------------------------------------------------------------
# Milestone 2 — Agent 3: Blocker & Action Item Identification Schemas
# -----------------------------------------------------------------------------

class BlockerItem(BaseModel):
    title: Optional[str] = Field(default=None, description="Title of the blocker")
    description: str = Field(description="Detailed explanation of the blocker activity, dependency, or issue")
    impact: str = Field(default="Not specified in the available project documents.", description="Impact on project or sprint delivery")
    status: str = Field(default="Active", description="Status of blocker e.g., Active, Open, In Progress, Resolved")
    source: str = Field(default="Not specified in the available project documents.", description="Source document filename")
    evidence: str = Field(default="Not specified in the available project documents.", description="Exact document snippet supporting blocker")
    document_name: Optional[str] = Field(default=None, description="Document filename")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")
    owner: Optional[str] = Field(default="Not specified", description="Person/team responsible for resolution")


class PendingDecision(BaseModel):
    title: Optional[str] = Field(default=None, description="Title or topic of pending decision")
    decision: str = Field(description="Description of decision that still needs to be made")
    owner: str = Field(default="Not specified in the available project documents.", description="Owner or decision-maker if mentioned")
    status: str = Field(default="Pending", description="Status e.g., Pending, Under Review, Escalated")
    source: str = Field(default="Not specified in the available project documents.", description="Source document filename")
    evidence: str = Field(default="Not specified in the available project documents.", description="Exact document snippet supporting decision")
    document_name: Optional[str] = Field(default=None, description="Document filename")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")


class UnresolvedIssue(BaseModel):
    title: Optional[str] = Field(default=None, description="Title of unresolved issue")
    issue: str = Field(description="Description of technical, operational, or testing issue mentioned")
    status: str = Field(default="Open", description="Status e.g., Open, In Investigation, Unresolved")
    source: str = Field(default="Not specified in the available project documents.", description="Source document filename")
    evidence: str = Field(default="Not specified in the available project documents.", description="Exact document snippet supporting issue")
    document_name: Optional[str] = Field(default=None, description="Document filename")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")


class ActionItem(BaseModel):
    action_id: Optional[str] = Field(default=None, description="Unique identifier for the action item e.g. ACT-01")
    action: str = Field(description="Description of specific action item or task to be completed")
    assignee: str = Field(default="Not specified in the available project documents.", description="Assigned person or team member")
    priority: str = Field(default="Not specified in the available project documents.", description="Priority e.g., High, Medium, Low")
    deadline: str = Field(default="Not specified in the available project documents.", description="Target deadline or completion date")
    status: str = Field(default="Open", description="Status e.g., Open, In Progress, Pending Review")
    related_blocker_risk: str = Field(default="Not specified in the available project documents.", description="Related blocker or risk")
    source: str = Field(default="Not specified in the available project documents.", description="Source document filename")
    evidence: str = Field(default="Not specified in the available project documents.", description="Exact document snippet supporting action item")
    document_name: Optional[str] = Field(default=None, description="Document filename")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID if applicable")


class BlockerActionOutput(BaseModel):
    blockers: List[BlockerItem] = Field(default_factory=list, description="Extracted project blockers")
    pending_decisions: List[PendingDecision] = Field(default_factory=list, description="Extracted pending decisions")
    unresolved_issues: List[UnresolvedIssue] = Field(default_factory=list, description="Extracted unresolved issues")
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted action items")
    sources: List[str] = Field(default_factory=list, description="Referenced document sources")


# Alias for BlockerActionResult
BlockerActionResult = BlockerActionOutput


# -----------------------------------------------------------------------------
# Milestone 3 — Part 1: Documentation Generation Agent Schemas
# -----------------------------------------------------------------------------

class UserStory(BaseModel):
    story_id: str = Field(description="Unique story identifier e.g. US-01")
    user_story: str = Field(description="User story in standard format: As a [role], I want [feature] so that [benefit]")
    description: str = Field(default="Not specified in the available project documents.", description="Detailed explanation of the user story")
    priority: str = Field(default="Must Have", description="MoSCoW priority: Must Have, Should Have, Could Have, or Won't Have")
    acceptance_criteria: List[str] = Field(default_factory=list, description="List of acceptance criteria")
    dependencies: str = Field(default="Not specified in the available project documents.", description="Prerequisite modules, APIs, or stories")
    source: str = Field(default="Not specified in the available project documents.", description="Source document name")
    evidence: str = Field(default="Not specified in the available project documents.", description="Direct snippet or quote from document")


class UserStoriesOutput(BaseModel):
    user_stories: List[UserStory] = Field(default_factory=list, description="List of generated user stories")
    sources: List[str] = Field(default_factory=list, description="Referenced source documents")


class RiskRegisterItem(BaseModel):
    risk_id: str = Field(description="Unique risk identifier e.g. R-01")
    risk_description: str = Field(description="Detailed explanation of the risk")
    category: str = Field(description="Risk category classification e.g. Schedule / Technical, Resource, Dependency")
    probability: str = Field(default="Not specified in the available project documents.", description="Probability rating (Low, Medium, High, or Not specified)")
    impact: str = Field(default="Not specified in the available project documents.", description="Impact rating (Low, Medium, High, or Not specified)")
    severity: str = Field(default="Medium", description="Severity rating (Low, Medium, High)")
    mitigation: str = Field(default="Not specified in the available project documents.", description="Mitigation strategy or action")
    owner: str = Field(default="Not specified in the available project documents.", description="Owner of the risk or person responsible")
    status: str = Field(default="Open", description="Risk status e.g. Open, In Progress, Mitigated, Closed")
    source: str = Field(default="Not specified in the available project documents.", description="Source document name")
    evidence: str = Field(default="Not specified in the available project documents.", description="Direct snippet or quote from document")


class RiskRegisterOutput(BaseModel):
    risk_register: List[RiskRegisterItem] = Field(default_factory=list, description="List of risk register items")
    sources: List[str] = Field(default_factory=list, description="Referenced source documents")


class ActionItemsOutput(BaseModel):
    action_items: List[ActionItem] = Field(default_factory=list, description="List of generated action items")
    sources: List[str] = Field(default_factory=list, description="Referenced source documents")


class DocumentationGenerationResult(BaseModel):
    user_stories: List[UserStory] = Field(default_factory=list, description="Generated user stories")
    risk_register: List[RiskRegisterItem] = Field(default_factory=list, description="Generated risk register")
    action_items: List[ActionItem] = Field(default_factory=list, description="Generated action items")
    sources: List[str] = Field(default_factory=list, description="Referenced source documents")


# -----------------------------------------------------------------------------
# Milestone 3 — Part 2: Project Health Scoring Schemas
# -----------------------------------------------------------------------------

HealthClassification = Literal["Healthy", "Moderate", "At Risk", "Critical"]
HealthConfidence = Literal["High", "Medium", "Low"]


class HealthDimensionScore(BaseModel):
    dimension_name: str = Field(description="Name of health dimension")
    score: float = Field(description="Score between 0 and 100 for this dimension")
    weight: float = Field(description="Relative weight in overall calculation (0.0 to 1.0)")
    weighted_score: float = Field(description="Score multiplied by weight")
    status: str = Field(description="Qualitative status for this dimension e.g. Healthy, Moderate, At Risk, Critical")
    factors: List[str] = Field(default_factory=list, description="Evidence-backed factors that affected this score")


class ProjectHealthOutput(BaseModel):
    overall_score: float = Field(description="Overall project health score (0 to 100)")
    classification: HealthClassification = Field(description="Classification: Healthy, Moderate, At Risk, or Critical")
    confidence: HealthConfidence = Field(description="Data confidence rating: High, Medium, or Low")
    data_sufficiency: str = Field(description="Explanation of data sufficiency")
    dimensions: Dict[str, HealthDimensionScore] = Field(default_factory=dict, description="Detailed score per dimension")
    key_factors: List[str] = Field(default_factory=list, description="Key project evidence factors determining score")
    recommendations: List[str] = Field(default_factory=list, description="Actionable next steps to improve project health")
    sources: List[str] = Field(default_factory=list, description="Referenced source documents")


# Alias for backwards compatibility
ProjectHealthResult = ProjectHealthOutput


# -----------------------------------------------------------------------------
# Milestone 3 — Part 3: Conversational Project Intelligence Assistant Schemas
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    content: str = Field(description="Text content of the message")
    sources: List[str] = Field(default_factory=list, description="Referenced document sources if assistant message")
    timestamp: Optional[float] = Field(default=None, description="Unix timestamp of message")


class ConversationalResponse(BaseModel):
    answer: str = Field(description="Grounded response to user question")
    sources: List[str] = Field(default_factory=list, description="Referenced source citations")
    chunks_used: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved context chunks utilized")
    project_id: str = Field(description="Workspace identifier")
    status: str = Field(default="Success", description="Execution status: Success or Failed")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")

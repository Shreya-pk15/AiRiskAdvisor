"""
Project Health Scoring Module.

Milestone 3 — Part 2:
Calculates a deterministic, evidence-grounded Project Health Score (0-100)
based on existing Milestone 2 Agent outputs (Scope, Risk, Blocker).

Health Dimensions:
1. Scope Clarity (25% weight)
2. Timeline Risk (30% weight)
3. Blocker Status (25% weight)
4. Delivery Risk (20% weight)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.schemas import (
    HealthClassification,
    HealthConfidence,
    HealthDimensionScore,
    ProjectHealthOutput,
)

logger = logging.getLogger(__name__)

# Default Thresholds
DEFAULT_THRESHOLDS = {
    "Healthy": 80.0,
    "Moderate": 60.0,
    "At Risk": 40.0,
    "Critical": 0.0,
}

# Dimension Weights
DEFAULT_WEIGHTS = {
    "scope_clarity": 0.25,
    "timeline_risk": 0.30,
    "blocker_status": 0.25,
    "delivery_risk": 0.20,
}


class ProjectHealthScorer:
    """
    Deterministic Project Health Scoring Engine.

    Calculates an overall project health score between 0 and 100 based strictly
    on evidence extracted by Scope, Risk, and Blocker agents without LLM calls.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

    def evaluate_health(
        self,
        scope_data: Optional[Dict[str, Any]] = None,
        risk_data: Optional[Dict[str, Any]] = None,
        blocker_data: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        retriever: Optional[Any] = None,
        top_k: int = 5,
    ) -> ProjectHealthOutput:
        """
        Evaluate project health using provided agent outputs or by invoking agents if missing.

        Args:
            scope_data: Output dictionary from ScopeExtractionAgent
            risk_data: Output dictionary from RiskDetectionAgent
            blocker_data: Output dictionary from BlockerActionAgent
            project_id: Workspace identifier if agents need to be queried
            retriever: Retriever instance if agents need to be queried
            top_k: Retrieval depth if agents need to be queried

        Returns:
            ProjectHealthOutput: Structured health score with dimension breakdowns and explanations.
        """
        # 1. Fallback to agents if data not supplied and project_id provided
        scope = scope_data or {}
        risk = risk_data or {}
        blocker = blocker_data or {}

        if not scope and project_id:
            try:
                from agents.scope_agent import ScopeExtractionAgent
                scope = ScopeExtractionAgent(retriever=retriever).run(project_id=project_id, top_k=top_k).get("data", {})
            except Exception as e:
                logger.warning("Could not auto-fetch scope data for health scorer: %s", e)

        if not risk and project_id:
            try:
                from agents.risk_agent import RiskDetectionAgent
                risk = RiskDetectionAgent(retriever=retriever).run(project_id=project_id, top_k=top_k).get("data", {})
            except Exception as e:
                logger.warning("Could not auto-fetch risk data for health scorer: %s", e)

        if not blocker and project_id:
            try:
                from agents.blocker_agent import BlockerActionAgent
                blocker = BlockerActionAgent(retriever=retriever).run(project_id=project_id, top_k=top_k).get("data", {})
            except Exception as e:
                logger.warning("Could not auto-fetch blocker data for health scorer: %s", e)

        # 2. Compute individual dimensions
        confidence, sufficiency_note = self._assess_data_sufficiency(scope, risk, blocker)

        scope_dim = self._calculate_scope_clarity(scope, confidence)
        timeline_dim = self._calculate_timeline_risk(risk, confidence)
        blocker_dim = self._calculate_blocker_status(blocker, confidence)
        delivery_dim = self._calculate_delivery_risk(risk, confidence)

        # 3. Overall Weighted Score
        w_scope = self.weights.get("scope_clarity", 0.25)
        w_time = self.weights.get("timeline_risk", 0.30)
        w_block = self.weights.get("blocker_status", 0.25)
        w_deliv = self.weights.get("delivery_risk", 0.20)

        overall_raw = (
            scope_dim.score * w_scope
            + timeline_dim.score * w_time
            + blocker_dim.score * w_block
            + delivery_dim.score * w_deliv
        )
        overall_score = round(max(0.0, min(100.0, overall_raw)), 1)

        # 4. Classification
        classification = self._classify_score(overall_score)

        # 5. Aggregate Key Factors & Recommendations
        key_factors = []
        key_factors.extend(scope_dim.factors)
        key_factors.extend(timeline_dim.factors)
        key_factors.extend(blocker_dim.factors)
        key_factors.extend(delivery_dim.factors)

        recommendations = self._generate_recommendations(
            scope_dim, timeline_dim, blocker_dim, delivery_dim, classification, confidence
        )

        # 6. Aggregate Sources
        all_sources = set()
        all_sources.update(scope.get("sources", []))
        all_sources.update(risk.get("sources", []))
        all_sources.update(blocker.get("sources", []))

        dimensions = {
            "Scope Clarity": scope_dim,
            "Timeline Risk": timeline_dim,
            "Blocker Status": blocker_dim,
            "Delivery Risk": delivery_dim,
        }

        return ProjectHealthOutput(
            overall_score=overall_score,
            classification=classification,
            confidence=confidence,
            data_sufficiency=sufficiency_note,
            dimensions=dimensions,
            key_factors=key_factors,
            recommendations=recommendations,
            sources=sorted(list(all_sources)),
        )

    # -------------------------------------------------------------------------
    # Dimension 1: Scope Clarity (25% Weight)
    # -------------------------------------------------------------------------

    def _calculate_scope_clarity(self, scope: Dict[str, Any], confidence: HealthConfidence) -> HealthDimensionScore:
        """
        Evaluate scope clarity based on goals, deliverables, milestones, and timeline.
        Max score: 100.
        """
        weight = self.weights.get("scope_clarity", 0.25)
        goals = scope.get("project_goals", [])
        deliverables = scope.get("deliverables", [])
        milestones = scope.get("milestones", [])
        timeline = scope.get("timeline", [])

        # Sub-score components
        # Goals: up to 25 pts
        goal_pts = 25.0 if len(goals) >= 2 else (15.0 if len(goals) == 1 else 0.0)

        # Deliverables: up to 35 pts
        deliv_count = len(deliverables)
        if deliv_count >= 3:
            deliv_pts = 35.0
        elif deliv_count == 2:
            deliv_pts = 25.0
        elif deliv_count == 1:
            deliv_pts = 15.0
        else:
            deliv_pts = 0.0

        # Milestones: up to 20 pts
        mile_pts = 20.0 if len(milestones) >= 2 else (12.0 if len(milestones) == 1 else 0.0)

        # Timeline: up to 20 pts
        time_pts = 20.0 if len(timeline) >= 2 else (12.0 if len(timeline) == 1 else 0.0)

        raw_score = goal_pts + deliv_pts + mile_pts + time_pts

        if not goals and not deliverables and not milestones and not timeline:
            raw_score = 15.0 if confidence == "Low" else 0.0

        score = round(max(0.0, min(100.0, raw_score)), 1)
        status = self._classify_score(score)

        factors = []
        if goals:
            factors.append(f"{len(goals)} project goal(s) clearly documented.")
        else:
            factors.append("No explicit project goals identified in project documents.")

        if deliverables:
            factors.append(f"{len(deliverables)} project deliverable(s) specified.")
        else:
            factors.append("Deliverables are missing or not explicitly itemized.")

        if milestones:
            factors.append(f"{len(milestones)} milestone(s) documented.")
        else:
            factors.append("Milestone breakdown not defined in available documents.")

        if timeline:
            factors.append(f"{len(timeline)} timeline/deadline reference(s) identified.")
        else:
            factors.append("No clear timeline or sprint schedule documented.")

        return HealthDimensionScore(
            dimension_name="Scope Clarity",
            score=score,
            weight=weight,
            weighted_score=round(score * weight, 1),
            status=status,
            factors=factors,
        )

    # -------------------------------------------------------------------------
    # Dimension 2: Timeline Risk (30% Weight)
    # -------------------------------------------------------------------------

    def _calculate_timeline_risk(self, risk: Dict[str, Any], confidence: HealthConfidence) -> HealthDimensionScore:
        """
        Evaluate timeline health based on Risk Agent's delivery status & forecast.
        Max score: 100.
        """
        weight = self.weights.get("timeline_risk", 0.30)
        delivery_status = risk.get("delivery_status", "Insufficient Data")
        forecast = risk.get("delivery_forecast", {})
        concerns = forecast.get("concerns", [])
        reason = forecast.get("reason", "")

        status_scores = {
            "On Track": 95.0,
            "At Risk": 55.0,
            "Delayed": 25.0,
            "Insufficient Data": 50.0,
        }

        base_score = status_scores.get(delivery_status, 50.0)

        # Deduct if multiple timeline concerns exist
        concern_penalty = min(15.0, len(concerns) * 5.0)
        score = max(0.0, min(100.0, base_score - concern_penalty))
        score = round(score, 1)

        status = self._classify_score(score)

        factors = [f"Timeline currently classified as '{delivery_status}'."]
        if reason and reason != "No documented evidence available.":
            factors.append(f"Forecast analysis notes: {reason}")
        if concerns:
            factors.append(f"{len(concerns)} schedule concern(s) recorded: {', '.join(concerns[:2])}.")

        return HealthDimensionScore(
            dimension_name="Timeline Risk",
            score=score,
            weight=weight,
            weighted_score=round(score * weight, 1),
            status=status,
            factors=factors,
        )

    # -------------------------------------------------------------------------
    # Dimension 3: Blocker Status (25% Weight)
    # -------------------------------------------------------------------------

    def _calculate_blocker_status(self, blocker: Dict[str, Any], confidence: HealthConfidence) -> HealthDimensionScore:
        """
        Evaluate impact of active blockers, unresolved issues, and pending decisions.
        Starts at 100 pts.
        """
        weight = self.weights.get("blocker_status", 0.25)
        blockers = blocker.get("blockers", [])
        decisions = blocker.get("pending_decisions", [])
        issues = blocker.get("unresolved_issues", [])

        # Filter active blockers
        active_blockers = [
            b for b in blockers
            if str(b.get("status", "Active")).lower() not in {"resolved", "closed"}
        ]

        # Penalties:
        # Each active blocker: -18 pts
        # Each unresolved issue: -8 pts
        # Each pending decision: -6 pts
        blocker_penalty = len(active_blockers) * 18.0
        issue_penalty = len(issues) * 8.0
        decision_penalty = len(decisions) * 6.0

        total_penalty = blocker_penalty + issue_penalty + decision_penalty
        raw_score = 100.0 - total_penalty

        # If completely empty and low confidence, default to 65 rather than pure 100
        if not blockers and not decisions and not issues and confidence == "Low":
            raw_score = 65.0

        score = round(max(0.0, min(100.0, raw_score)), 1)
        status = self._classify_score(score)

        factors = []
        if active_blockers:
            factors.append(f"{len(active_blockers)} active blocker(s) impacting progress.")
        else:
            factors.append("No active blockers identified.")

        if issues:
            factors.append(f"{len(issues)} unresolved defect/impediment(s) reported.")
        if decisions:
            factors.append(f"{len(decisions)} pending architectural or project decision(s).")

        return HealthDimensionScore(
            dimension_name="Blocker Status",
            score=score,
            weight=weight,
            weighted_score=round(score * weight, 1),
            status=status,
            factors=factors,
        )

    # -------------------------------------------------------------------------
    # Dimension 4: Delivery Risk (20% Weight)
    # -------------------------------------------------------------------------

    def _calculate_delivery_risk(self, risk: Dict[str, Any], confidence: HealthConfidence) -> HealthDimensionScore:
        """
        Evaluate delivery risk severity (high, medium, low risk items).
        Starts at 100 pts.
        """
        weight = self.weights.get("delivery_risk", 0.20)
        risks = risk.get("risks", [])

        high_risks = [r for r in risks if str(r.get("severity", "")).lower() == "high"]
        med_risks = [r for r in risks if str(r.get("severity", "")).lower() == "medium"]
        low_risks = [r for r in risks if str(r.get("severity", "")).lower() == "low"]

        # Penalties:
        # High: -20 pts each
        # Med:  -10 pts each
        # Low:  -4 pts each
        risk_penalty = (len(high_risks) * 20.0) + (len(med_risks) * 10.0) + (len(low_risks) * 4.0)
        raw_score = 100.0 - risk_penalty

        if not risks and confidence == "Low":
            raw_score = 60.0

        score = round(max(0.0, min(100.0, raw_score)), 1)
        status = self._classify_score(score)

        factors = []
        if high_risks:
            factors.append(f"{len(high_risks)} high-severity risk(s) identified.")
        if med_risks:
            factors.append(f"{len(med_risks)} medium-severity risk(s) identified.")
        if low_risks:
            factors.append(f"{len(low_risks)} low-severity risk(s) identified.")
        if not risks:
            factors.append("No significant delivery risks identified from available documents.")

        return HealthDimensionScore(
            dimension_name="Delivery Risk",
            score=score,
            weight=weight,
            weighted_score=round(score * weight, 1),
            status=status,
            factors=factors,
        )

    # -------------------------------------------------------------------------
    # Helpers: Data Sufficiency, Classification & Recommendations
    # -------------------------------------------------------------------------

    def _assess_data_sufficiency(
        self,
        scope: Dict[str, Any],
        risk: Dict[str, Any],
        blocker: Dict[str, Any],
    ) -> Tuple[HealthConfidence, str]:
        """Determine whether documentation contains sufficient evidence for confident scoring."""
        count = 0
        count += len(scope.get("project_goals", []))
        count += len(scope.get("deliverables", []))
        count += len(scope.get("milestones", []))
        count += len(scope.get("timeline", []))
        count += len(risk.get("risks", []))
        count += len(blocker.get("blockers", []))
        count += len(blocker.get("action_items", []))

        deliv_status = risk.get("delivery_status", "")

        if count < 3 or deliv_status == "Insufficient Data" or not scope and not risk and not blocker:
            return "Low", "Low — sparse project documentation available in knowledge base."
        elif count <= 7:
            return "Medium", "Medium — partial project artifacts available."
        else:
            return "High", "High — comprehensive documentation available across scope, risks, and blockers."

    def _classify_score(self, score: float) -> HealthClassification:
        """Map score between 0 and 100 to health classification."""
        if score >= self.thresholds.get("Healthy", 80.0):
            return "Healthy"
        elif score >= self.thresholds.get("Moderate", 60.0):
            return "Moderate"
        elif score >= self.thresholds.get("At Risk", 40.0):
            return "At Risk"
        else:
            return "Critical"

    def _generate_recommendations(
        self,
        scope_dim: HealthDimensionScore,
        timeline_dim: HealthDimensionScore,
        blocker_dim: HealthDimensionScore,
        delivery_dim: HealthDimensionScore,
        classification: HealthClassification,
        confidence: HealthConfidence,
    ) -> List[str]:
        """Produce concrete, actionable next steps based on dimension bottlenecks."""
        recs = []
        if confidence == "Low":
            recs.append("Ingest additional project artifacts (SRS, Sprint Tracker, Meeting Notes) to improve score confidence.")

        if scope_dim.score < 70.0:
            recs.append("Clarify undocumented project deliverables and milestone deadlines to increase scope clarity.")

        if timeline_dim.score < 60.0:
            recs.append("Conduct an urgent timeline review to re-baseline sprint deadlines or reduce scope.")

        if blocker_dim.score < 70.0:
            recs.append("Triage active blockers and assign dedicated owners to resolve pending technical impediments.")

        if delivery_dim.score < 70.0:
            recs.append("Implement mitigation strategies for documented high/medium severity delivery risks.")

        if not recs and classification == "Healthy":
            recs.append("Maintain sprint velocity and continue weekly risk and milestone tracking.")

        return recs

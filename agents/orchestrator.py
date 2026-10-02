from typing import Any, Dict, List, Sequence

from agents.blocker_agent import BlockerActionAgent
from agents.risk_agent import RiskDetectionAgent
from agents.scope_agent import ScopeExtractionAgent


class AgentOrchestrator:
    """Runs the three Milestone 2 agents and returns a unified intelligence payload."""

    def __init__(self, project_name: str = "Main Workspace"):
        self.project_name = project_name
        self.scope_agent = ScopeExtractionAgent()
        self.risk_agent = RiskDetectionAgent()
        self.blocker_agent = BlockerActionAgent()

    def run_all(self, retrieved_chunks: Sequence[Dict[str, Any]] | None) -> Dict[str, Any]:
        scope = self.scope_agent.analyze_context(retrieved_chunks, project_name=self.project_name)
        risks = self.risk_agent.analyze_context(retrieved_chunks, project_name=self.project_name)
        blockers = self.blocker_agent.analyze_context(retrieved_chunks, project_name=self.project_name)

        return {
            "scope": scope.model_dump(),
            "risks": risks.model_dump(),
            "blockers": blockers.model_dump(),
        }

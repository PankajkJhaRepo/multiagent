from typing import Any, Dict
from agents.researcher.deep_researcher.memory.deep_researcher_state import ResearchState


class HumanAgent:
    """Agent for human interaction"""

    def __init__(self):
        print("Human Agent initialized.")
    
    def review_feedback(self, state: ResearchState) -> Dict[str, Any]:
        """Review feedback from the human agent."""
        print("Reviewing feedback...")
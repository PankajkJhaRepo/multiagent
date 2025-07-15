from request import RequestedTask
from memory.agent_state import AgentState
from typing import Dict, Any


class HumanAgent:
    """A simple human agent that can interact with the system."""
    
    def __init__(self):
        print("Init HumanAgent")

    def review_plan(self, state: AgentState) -> Dict[str, Any]:
        """
        Review the plan and provide feedback.
        """
        print(f"Reviewing plan from state")
        
        # Extract relevant information from state
        task = state.get("task", {})
        initial_research = state.get("initial_research")
        deep_research = state.get("deep_research", {})
        
        print(f"Reviewing task: {task.get('query', 'Unknown')}")
        print(f"Initial research available: {initial_research is not None}")
        print(f"Deep research available: {bool(deep_research)}")
        
        # TODO: Implement actual human review logic here
        # For now, we'll simulate automatic approval
        human_feedback = {
            "status": "approved",
            "feedback": "Research looks comprehensive and well-structured.",
            "reviewer": "HumanAgent",
            "reviewed_at": "2025-07-15"
        }
        
        # Update state with human feedback
        updated_state = state.copy()
        updated_state["human_feedback"] = human_feedback
        updated_state["agent_state"] = "HumanReviewComplete"
        
        print("Plan reviewed successfully")
        return updated_state

from typing import Any, Dict
from agents.researcher.deep_researcher.memory.deep_researcher_state import ResearchState


class ReturnBack_Agent:
    """
    Agent to return back to the previous state in the research workflow.
    This agent is used when the user wants to go back to a previous step.
    """
    
    def __init__(self):
        print("ReturnBack_Agent initialized.")

    def invoke(self, state: ResearchState) -> Dict[str, Any]:
        """
        Invoke the agent to return back to the previous state.
        """
        print("Returning back research...")  
        research_result = state.get("research_result")
        print(f"Returning back with research result: {research_result}")
        return {
            "research_result": research_result,
        }
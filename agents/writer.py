from request import RequestedTask
from memory.agent_state import AgentState
from typing import Dict, Any


class WriterAgent:
    """Todo A simple writer agent that can create and format content based on requests."""
    
    def __init__(self):
        print("Init WriterAgent")
    
    def run(self, state: AgentState) -> Dict[str, Any]:
        """Run the writer agent with the given state."""
        print(f"Running WriterAgent with state:")
        
        # Extract relevant information from state
        task = state.get("task", {})
        initial_research = state.get("initial_research")
        deep_research = state.get("deep_research", {})
        
        print(f"Task: {task}")
        print(f"Initial research available: {initial_research is not None}")
        print(f"Deep research available: {bool(deep_research)}")
        
        # TODO: Implement actual writing logic here
        # For now, create a basic final report structure
        final_report = {
            "title": f"Research Report: {task.get('query', 'Unknown Topic')}",
            "summary": "This is a placeholder summary of the research findings.",
            "content": "Detailed content will be generated based on the research findings.",
            "sources": [],
            "created_at": "2025-07-15",
            "agent": "WriterAgent"
        }
        
        # Update state with the final report
        updated_state = state.copy()
        updated_state["final_report"] = final_report
        updated_state["agent_state"] = "WritingComplete"
        
        print("Writing completed successfully")
        return updated_state

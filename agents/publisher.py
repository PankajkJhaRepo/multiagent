from request import RequestedTask
from memory.agent_state import AgentState
from typing import Dict, Any


class PublisherAgent:
    """A simple publisher agent that can format and publish content."""
    
    def __init__(self):
        print("Init PublisherAgent")

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Format and publish the content based on the specified format.
        """
        print(f"Publishing content from state")
        
        # Extract final report from state
        final_report = state.get("final_report", {})
        task = state.get("task", {})
        
        if not final_report:
            print("Warning: No final report found in state")
            final_report = {"title": "Empty Report", "content": "No content available"}
        
        print(f"Publishing report: {final_report.get('title', 'Untitled')}")
        
        # TODO: Implement actual publishing logic here (save to file, send to API, etc.)
        # For now, we'll just simulate successful publishing
        
        published_content = {
            "status": "published",
            "publication_id": f"pub_{task.get('task_id', 'unknown')}",
            "title": final_report.get("title", "Untitled Report"),
            "published_at": "2025-07-15",
            "format": "text",  # could be "pdf", "html", etc.
        }
        
        # Update state with publication info
        updated_state = state.copy()
        updated_state["publication_result"] = published_content
        updated_state["agent_state"] = "PublishingComplete"
        
        print("Content published successfully")
        return updated_state

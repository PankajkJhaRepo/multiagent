from typing import List, Dict, Any, TypedDict, Optional

class AgentState(TypedDict, total=False):
    """
    Represents the state of a research workflow.
    """
    task: Dict[str, Any]
    agent_state: str  # e.g., "InitialResearch", "WritingComplete", etc.
    initial_research: Any  # Will be RelatedTopics but using Any to avoid circular imports
    deep_research: Dict[str, Any]  # Placeholder for deep research results
    final_report: Dict[str, Any]  # The final research report
    human_feedback: Optional[Dict[str, Any]]  # Human review feedback
    publication_result: Optional[Dict[str, Any]]  # Publication details
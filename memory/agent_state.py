from typing import List, Dict, Any, TypedDict, Optional
from agents.researcher.initial_researcher.chains.initial_research_chain import RelatedTopics

class AgentState(TypedDict, total=False):
    """
    Represents the state of a research workflow.
    """
    task: Dict[str, Any]
    agent_state: str  # e.g., "InitialResearch", "WritingComplete", etc.
    initial_research: RelatedTopics  # Changed from dict[str, Any] to RelatedTopics
    deep_research: Dict[str, Any]  # Placeholder for deep research results
    final_report: Dict[str, Any]  # The final research report
    human_feedback: Optional[Dict[str, Any]]  # Human review feedback
    publication_result: Optional[Dict[str, Any]]  # Publication details
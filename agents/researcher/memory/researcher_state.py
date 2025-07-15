from typing import List, Dict, Any, TypedDict

from agents.researcher.memory.research_topics import RelatedTopics



class ResearchState(TypedDict):
    """
    Represents the state of a research .
    """
    task: dict
    initial_research: RelatedTopics  # Changed from dict[str, Any] to RelatedTopics
    deep_research: RelatedTopics  # Placeholder for deep research results
    execution_status: str  # e.g., "InitialResearch", "DeepResearch", etc.
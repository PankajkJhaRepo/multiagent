from typing import Any, List, Literal, TypedDict

from agents.researcher.memory.research_topics import RelatedTopics


class ResearchState(TypedDict):
    """
    Represents the state of a research .
    """
    task:dict[str,Any]
    research_from: Literal['WebSearch', 'KnowledgeBase' ]
    research_state: str
    human_feedback: str
    research_result: RelatedTopics  # Changed from dict[str, Any] to RelatedTopics
    is_hallucinationed: bool
    revise_research: bool
    response_grader_score: bool

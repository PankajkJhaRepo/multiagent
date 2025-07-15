    
    # research-state could be Literals as number of state can be fixed eg Started->Running{Name}->...->Stopped
    # For now string is fine

from typing import List, Literal, TypedDict

from agents.researcher.memory.research_topics import RelatedTopics


class InitialResearchState(TypedDict):
    query:str
    ## todo research_from: List[str]  # List of sources like 'WebSearch', 'KnowledgeBase'
    research_from: Literal['WebSearch', 'KnowledgeBase' ]

    research_state: str
    human_feedback: str
    research_result: RelatedTopics  # Changed from dict[str, Any] to RelatedTopics
    hallucination_score: bool
    research_reviewer_score: bool
    response_grader_score: bool


from typing import TypedDict

class DraftState(TypedDict):
    """
    Represents the state of a draft.
    """
    task: dict
    topic: str
    draft: dict
    review: str
    revision_notes: str
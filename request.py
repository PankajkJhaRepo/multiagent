from typing import List, Literal, TypedDict

class RequestedTask(TypedDict):
    """A task requested by the user."""
    task_id: str
    query: str
    max_sections: int
    publish_format: Literal["markdown", "pdf", "docx"] = "markdown"
    include_human_feedback: bool = False
    follow_guidelines: bool = False
    model: Literal["gpt-3.5-turbo", "gpt-4"] = "gpt-4"
    guidelines: List[str] = []
    verbose: bool = True
    
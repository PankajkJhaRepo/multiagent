from pydantic import BaseModel


class Topic(BaseModel):
    """A class representing a related topic in the research workflow."""
    topic: str
    description: str
    source: str

class RelatedTopics(BaseModel):
    """A class representing a list of related topics."""
    topics: list[Topic]
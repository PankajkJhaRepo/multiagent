from typing import Any, Dict
from agents.researcher.initial_researcher.memory.initial_research_state import InitialResearchState
from agents.researcher.initial_researcher.chains.research_reviewer_chain import research_reviewer_chain

class ResearchReviewerAgent:
    """Agent responsible for reviewing research findings."""
    
    def __init__(self):
        # Initialize any necessary attributes or dependencies here
        print("Research Reviewer Agent initialized.")

    def review_research(self,state: InitialResearchState) -> Dict[str, Any]:
        # Implementation of research review logic
        print("Running research review...")
        query = state.get("query")
        research_result= state.get("research_result")

        # Create a list to store topics that pass the review
        reviewed_topics = []

        for topic in research_result.topics:
            print(f"Topic: {topic.topic}")
            print(f"Description: {topic.description}")
            print(f"Source: {topic.source}")
            score = research_reviewer_chain.invoke({
                "query": query,
                "topic": topic.topic,
                "description": topic.description,
                "source": topic.source
            })
            print(f"Review Score: {score.binary_score}")
              # Only keep complete topic objects that scored True
            if score.binary_score is True:
                reviewed_topics.append(topic)  # Keep the entire topic object
                print(f"✓ Topic object '{topic.topic}' passed review and retained")
            else:
                print(f"✗ Topic '{topic.topic}' failed review and will be removed")
        

        
        print(f"Review completed. {len(reviewed_topics)} topics passed out of {len(research_result.topics)} original topics")
        
        # Update the research_result with filtered topics
        research_result.topics = reviewed_topics

        # Return updated state
        return {
            "research_result": research_result,
            "research_reviewer_score": len(reviewed_topics) > 0  # True if at least one topic passed
        }
            
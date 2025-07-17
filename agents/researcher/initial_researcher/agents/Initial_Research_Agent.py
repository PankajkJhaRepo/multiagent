from typing import Any, Dict
from agents.researcher.initial_researcher.memory.initial_research_state import InitialResearchState
from agents.researcher.initial_researcher.chains.initial_research_chain import research_chain

class InitialResearchAgent:
    """Agent responsible for conducting initial research."""
    def __init__(self):
        # Initialize any necessary attributes or dependencies here
        print("Initial Research Agent initialized.")
    def run_initial_research(self,state: InitialResearchState) -> Dict[str, Any]:
        # Implementation of initial research logic
        print("Running initial research...")
        query = state.get("query")
        research_result = state.get("research_result")
        
        try:
            # Check if research_result exists and prepare the request accordingly
            if research_result is not None:
                print("Previous research found, including history in prompt")
                response = research_chain.invoke({
                    "topic": query,
                    "history": research_result
                })
            else:
                print("No previous research found, proceeding without history")
                response = research_chain.invoke({
                    "topic": query
                })
        except Exception as e:
            print(f"Error in research chain: {e}")
            # Create a fallback response
            from agents.researcher.memory.research_topics import RelatedTopics, Topic
            response = RelatedTopics(topics=[
                Topic(
                    topic=f"Research on {query}",
                    description=f"Initial research topic: {query}. Error occurred during research process.",
                    source="System fallback"
                )
            ])
        
        return {
            "query": query,
            "research_result": response,
            "research_state": "InitialResearch",
            "human_feedback": None
        }

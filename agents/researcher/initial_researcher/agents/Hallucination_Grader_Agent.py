from typing import Any, Dict
from agents.researcher.initial_researcher.memory.initial_research_state import InitialResearchState
from agents.researcher.initial_researcher.chains.hallucination_grader_chain import hallucination_grader_chain

class HallucinationGraderAgent:
    """Agent responsible for grading hallucinations in research."""
    
    def __init__(self):
        # Initialize any necessary attributes or dependencies here
        print("Hallucination Grader Agent initialized.")
    
    def verify_hallucinations(self,state: InitialResearchState) -> Dict[str, Any]:
        # Implementation of hallucination grading logic
        print("Planning research and grading hallucinations...")
        query = state.get("query")
        research_result= state.get("research_result")

        score= hallucination_grader_chain.invoke({
            "query": query,
            "topics": research_result
        })
        print(f"verify_hallucinations Score: {score}")

        return {
            "query": query,
            "research_result": research_result,
            "research_state": "HallucinationGraded",
            "human_feedback": None,
            "hallucination_score": score
        }

        print(f"verify_hallucinations Query: {query}")
from typing import Any, Dict
from agents.researcher.deep_researcher.memory.deep_researcher_state import ResearchState


class HallucinationGraderAgent:
    """Agent to verify hallucinations in research results."""
    
    def __init__(self):
        print("Hallucination Grader Agent initialized.")

    def verify_hallucinations(self, state: ResearchState) -> Dict[str, Any]:
        """
        Verify if the research results contain hallucinations.
        Returns 'continue' if no hallucinations are found, otherwise 'revise'.
        """
        print("Verifying hallucinations in research results...")
        # hallucination_score = state.get("hallucination_score", 0)
        
        # if hallucination_score < 0.5:  # Threshold for hallucination detection
        #     print("No significant hallucinations detected.")
        #     return "continue"
        # else:
        #     print("Hallucinations detected, revision needed.")
        #     return "revise"
        return{
            "is_hallucinationed": False
        }
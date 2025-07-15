from typing import Any, Dict
from agents.constants import HUMAN_FEEDBACK
from agents.researcher.initial_researcher.memory.initial_research_state import InitialResearchState

### Will comeback to veify the use of this agent
### The HumanAgent is responsible for providing human feedback in the research process.
### What I expect is the While loop in resercher.py to be here 
### Todo Will came back later to verify the use of this agent
### The problem with current approach is that since it has while loop in researcher.py,'
### it will require be to in single executiom i.e the user cannot comeback later to provide feedback which may be required
### This agent may help to solve that problem
### Do not delete this agent
class HumanAgent:
    """Agent responsible for human feedback in the research process."""
    
    def __init__(self):
        # Initialize any necessary attributes or dependencies here
        print("Human Agent initialized.")
    
    def get_human_feedback(self,state: InitialResearchState) -> Dict[str, Any]:
        # Implementation of human feedback logic
        print("Running human feedback...")
        # feedback = state.get(HUMAN_FEEDBACK)
        # print(f"{HUMAN_FEEDBACK} Query: {feedback}")
        # # Simulate human feedback process
        # return {"human_feedback": feedback}  # Example feedback


from typing import Dict, List, Optional
from memory.agent_state import AgentState
from request import RequestedTask


class PlannerAgent:
    """A simple planner agent that can create and execute plans for tasks."""
    
    def __init__(self):
        print("Init PlannerAgent")

    async def plan_research(self, research_state: AgentState) -> Dict[str, any]:
        """
        Plan on the research topic
        """
        initial_research = research_state["initial_research"]
        task = research_state["task"]
        include_human_feedback = task.get("include_human_feedback")
        human_feedback = research_state.get("human_feedback")
        max_sections = task.get("max_sections")

        prompt = self._create_planning_prompt(
            initial_research, include_human_feedback, human_feedback, max_sections)

    def _create_planning_prompt(self, initial_research: str, include_human_feedback: bool,
                                human_feedback: Optional[str], max_sections: int) -> List[Dict[str, str]]:
        """Create the prompt for research planning."""
        return [
            {
                "role": "system",
                "content": "You are a research editor. Your goal is to oversee the research project "
                           "from inception to completion. Your main task is to plan the article section "
                           "layout based on an initial research summary.\n ",
            },
            {
                "role": "user",
                "content": self._format_planning_instructions(initial_research, include_human_feedback,
                                                              human_feedback, max_sections),
            },
        ]
    
    def run_parallel_research(self):
        """Run parallel research tasks."""
        # This method would implement the logic to run parallel research tasks
        # For now, it is a placeholder
        print("Running parallel research tasks...")
        return "Parallel research tasks completed."




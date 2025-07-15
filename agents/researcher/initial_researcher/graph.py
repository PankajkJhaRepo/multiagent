from dotenv import load_dotenv
import os
from langgraph.graph import END, StateGraph

from agents.constants import HALLUCINATION_GRADER, HUMAN_FEEDBACK, INITIAL_PLAN, INITIAL_RESEARCH, RESEARCH_REVIEWER, RESPONSE_GRADER

from .memory.initial_research_state import InitialResearchState
from .agents.Initial_Research_Agent import InitialResearchAgent
from .agents.Hallucination_Grader_Agent import HallucinationGraderAgent
from .agents.Research_Reviewer_Agent import ResearchReviewerAgent
from .agents.Initial_Planner_Agent import InitialPlannerAgent
from .agents.Response_Grader_Agent import Response_Grader_Agent
from .agents.Human_Agent import HumanAgent

def _initialize_agents():
    return {
        INITIAL_RESEARCH: InitialResearchAgent(),
        HALLUCINATION_GRADER: HallucinationGraderAgent(),
        RESEARCH_REVIEWER: ResearchReviewerAgent(),
        RESPONSE_GRADER: Response_Grader_Agent(),
        HUMAN_FEEDBACK: HumanAgent(),
        INITIAL_PLAN: InitialPlannerAgent()
    }

def _create_workflow(agents):
    workflow = StateGraph(InitialResearchState)

    # Add nodes for each agent
    workflow.add_node(INITIAL_RESEARCH, agents[INITIAL_RESEARCH].run_initial_research)
    workflow.add_node(HALLUCINATION_GRADER, agents[HALLUCINATION_GRADER].verify_hallucinations)
    workflow.add_node(RESEARCH_REVIEWER, agents[RESEARCH_REVIEWER].review_research)
    workflow.add_node(RESPONSE_GRADER, agents[RESPONSE_GRADER].grade_response)
    workflow.add_node(HUMAN_FEEDBACK, agents[HUMAN_FEEDBACK].get_human_feedback)
    workflow.add_node(INITIAL_PLAN, agents[INITIAL_PLAN].plan_initial_research)

    # Add edges
    _add_workflow_edges(workflow)

    return workflow

def _decide_next_step(state: InitialResearchState) -> str:
    """Always continue to INITIAL_PLAN after human feedback."""
    return "continue"

def _decide_next_step(state: InitialResearchState) -> str:
    """
    Decision function to determine next step based on human feedback.
    Returns 'accept' to go to initial-plan, 'revise' to go back to initial-research.
    """
    print("Deciding next step based on human feedback...")
    feedback = state.get(HUMAN_FEEDBACK)
    print(f"Human feedback received: {feedback}")
    
    # Check if feedback indicates acceptance
    if feedback and feedback.lower() in ['accept', 'approved', 'good', 'ok', 'yes']:
        return "accept"
    else:
        return "revise"

def _add_workflow_edges(workflow):

    
    workflow.add_edge(INITIAL_RESEARCH, HALLUCINATION_GRADER)
    
    # workflow.add_edge(HALLUCINATION_GRADER, RESEARCH_REVIEWER)
    workflow.add_conditional_edges(
        HALLUCINATION_GRADER,
        lambda state: "continue" if state.get("hallucination_score") else "revise",
        {
            "continue": RESEARCH_REVIEWER,
            "revise": INITIAL_RESEARCH
        }
    )

    workflow.add_conditional_edges(
        RESEARCH_REVIEWER,
        lambda state: "continue" if state.get("research_reviewer_score") else "revise",
        {
            "continue": RESPONSE_GRADER,
            "revise": INITIAL_RESEARCH
        }
    )

    workflow.add_conditional_edges(
        RESPONSE_GRADER,
        lambda state: "continue" if state.get("response_grader_score") else "revise",
        {
            "continue": HUMAN_FEEDBACK,
            "revise": INITIAL_RESEARCH
        }
    )


    
    # workflow.add_edge(RESEARCH_REVIEWER, RESPONSE_GRADER)
    
    # workflow.add_edge(RESPONSE_GRADER, HUMAN_FEEDBACK)
    # workflow.add_edge(HUMAN_FEEDBACK, INITIAL_PLAN)
    
    # Replace simple edge with conditional edge
    # workflow.add_conditional_edges(
    #     HUMAN_FEEDBACK,
    #     _decide_next_step,  # Decision function
    #     {
    #         "continue": INITIAL_PLAN,  # Always continue to INITIAL_PLAN
    #     }
    # )
    
    workflow.set_entry_point(INITIAL_RESEARCH)
    workflow.add_edge(INITIAL_PLAN, END)

    # Add human in the loop
    workflow.add_conditional_edges(
            HUMAN_FEEDBACK,
            _decide_next_step,  # Decision function
            {
                "accept": INITIAL_PLAN,
                "revise": INITIAL_RESEARCH
            }
        )

def init_research_team():
    """Initialize the research team with various agents."""
    agents = _initialize_agents()
    return _create_workflow(agents)


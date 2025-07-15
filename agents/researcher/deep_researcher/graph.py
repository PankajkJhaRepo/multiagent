
from agents.researcher.deep_researcher.agents.ReturnBack_Agent import ReturnBack_Agent
from agents.researcher.deep_researcher.agents.Planner_Agent import Planner_Agent
from agents.researcher.deep_researcher.agents.HallucinationGrader_Agent import HallucinationGraderAgent
from agents.researcher.deep_researcher.agents.Human_Agent import HumanAgent
from agents.researcher.deep_researcher.agents.ResearchReviewer_Agent import ResearchReviewerAgent
from agents.researcher.deep_researcher.agents.Response_Grader_Agent import ResponseGraderAgent
from agents.researcher.deep_researcher.constants import PLAN, RESEARCH, HALLUCINATION_GRADER, HUMAN_FEEDBACK, RESEARCH_REVIEWER, RESPONSE_GRADER, RETURN_BACK
from agents.researcher.deep_researcher.memory.deep_researcher_state import ResearchState
from agents.researcher.deep_researcher.agents.Research_Agent import ResearchAgent
from langgraph.graph import StateGraph,END


def _initialize_agents():
    return {
        PLAN: Planner_Agent(),
        RESEARCH: ResearchAgent(),
        HALLUCINATION_GRADER: HallucinationGraderAgent(),
        RESEARCH_REVIEWER: ResearchReviewerAgent(),
        RESPONSE_GRADER: ResponseGraderAgent(),
        HUMAN_FEEDBACK: HumanAgent(),
        RETURN_BACK: ReturnBack_Agent()
    }

def _create_workflow(agents):
    workflow = StateGraph(ResearchState)

    # Add nodes for each agent
    workflow.add_node(PLAN, agents[PLAN].plan_research)
    workflow.add_node(RESEARCH, agents[RESEARCH].run_research)
    workflow.add_node(HALLUCINATION_GRADER, agents[HALLUCINATION_GRADER].verify_hallucinations)
    workflow.add_node(RESEARCH_REVIEWER, agents[RESEARCH_REVIEWER].review_research)
    workflow.add_node(RESPONSE_GRADER, agents[RESPONSE_GRADER].grade_response)
    workflow.add_node(HUMAN_FEEDBACK, agents[HUMAN_FEEDBACK].review_feedback)
    workflow.add_node(RETURN_BACK, agents[RETURN_BACK].invoke)

    # Add edges
    _add_workflow_edges(workflow)

    return workflow


def _add_workflow_edges(workflow):

    workflow.set_entry_point(PLAN)
    workflow.add_edge(PLAN, RESEARCH)
    workflow.add_edge(RESEARCH, HALLUCINATION_GRADER)
    
    # workflow.add_edge(HALLUCINATION_GRADER, RESEARCH_REVIEWER)
    workflow.add_conditional_edges(
        HALLUCINATION_GRADER,
        lambda state: "continue" if state.get("hallucination_score") else "revise",
        {
            "continue": RESEARCH_REVIEWER,
            "revise": RESEARCH
        }
    )

    workflow.add_conditional_edges(
        RESEARCH_REVIEWER,
        lambda state: "continue" if state.get("research_reviewer_score") else "revise",
        {
            "continue": RESPONSE_GRADER,
            "revise": RESEARCH
        }
    )

    workflow.add_conditional_edges(
        RESPONSE_GRADER,
        lambda state: "continue" if state.get("response_grader_score") else "revise",
        {
            "continue": HUMAN_FEEDBACK,
            "revise": RESEARCH
        }
    )

    # Add human in the loop
    workflow.add_conditional_edges(
            HUMAN_FEEDBACK,
            _decide_next_step,  # Decision function
            {
                "accept": RETURN_BACK,
                "revise": RESEARCH
            }
        )
    workflow.add_edge(RETURN_BACK, END)

def _decide_next_step(state: ResearchState) -> str:
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


def init_deep_research_team():
    """Initialize the research team with various agents."""
    agents = _initialize_agents()
    return _create_workflow(agents)
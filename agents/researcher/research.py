

from typing import Any, Dict
from agents.constants import HUMAN_FEEDBACK
from agents.researcher.deep_researcher.graph import init_deep_research_team
from agents.researcher.memory.researcher_state import ResearchState
from agents.researcher.initial_researcher.chains.initial_research_chain import RelatedTopics

from  .initial_researcher.graph import init_research_team
import datetime
from langgraph.checkpoint.memory import MemorySaver

class ResearchAgent:
    """A simple research agent that can gather information based on requests."""
    
    def __init__(self, websocket=None, stream_output=False, headers=None):
        print("Init ResearchAgent")
    
    async def run_initial_research(self,research_state: ResearchState):
        print("Running initial research...")
        task = research_state.get("task")
        query = task.get("query")
        ### todo source can be based on some plan
        source = task.get("source", "web")
        task_id = task.get("task_id", "default-task-id")
        print(f"Running initial research on query: {query} with source: {source} and task_id: {task_id}")
        # if self.websocket and self.stream_output:
        #     await self.stream_output("logs", "initial_research", f"Running initial research on the following query: {query}", self.websocket)
        # else:
        #     print_agent_output(f"Running initial research on the following query: {query}", agent="RESEARCHER")
        research_report =  await self.get_research_report(query=query, task_id=task_id)
        # print(f"Initial research completed with report: {research_report}")
        print(f"Type of research report: {type(research_report)}")
        print(f"Task {task}")
        return {"task": task, "initial_research": research_report,
                "execution_status": "InitialResearch"}
    
    async def get_research_report(self,query: str, task_id:str):
        # Initialize the researcher
        # researcher = GPTResearcher(query=query, report_type=research_report, parent_query=parent_query,
        #                            verbose=verbose, report_source=source, tone=tone, websocket=self.websocket, headers=self.headers)
        
        researcher_workflow  = init_research_team()
        ## Temporary in-memory checkpointing
        memory = MemorySaver()
        researcher = researcher_workflow.compile(checkpointer=memory,interrupt_before=[HUMAN_FEEDBACK])
        researcher.get_graph().draw_mermaid_png(output_file_path="initial_researcher.png")

        thread = {"configurable":{"thread_id": task_id}}
        initial_input = {"query": query}
        response= await researcher.ainvoke(initial_input,thread)
        print("Response:", response)
        print(researcher.get_state(thread).next)
        while True:
            user_input = input("Enter your feedback: ")
            feedback = {HUMAN_FEEDBACK: user_input}
            response = researcher.update_state(thread,feedback,as_node=HUMAN_FEEDBACK)
            print("Response after feedback:", response)
            print(researcher.get_state(thread).next)
            print("Final state:", researcher.get_state(thread))
            response= await researcher.ainvoke(None,thread,)
            print("Response after second input:", response)
            if user_input.lower() == "accept":
                print("Feedback accepted, proceeding to next step.")
                break
        
        # print(f"Research completed with response {response}")
        # print(f"Type of response: {type(response)}")
        research_result = response["research_result"]
        ### todo verify the response, it should be a dict with research_result
        return research_result;

    async def get_deep_research_report(self, task:dict,initial_research: RelatedTopics):
        # Placeholder for deep research logic
        query = task.get("query")
        source = task.get("source", "web")
        task_id = task.get("task_id", "default-task-id")
        print(f"Running deep research on query: {query} with source: {source} and task_id: {task_id}")
        # Implement the logic to get deep research report
        # This could involve more complex queries, multiple sources, etc.

        deep_researcher_workflow  = init_deep_research_team()

        ## Temporary in-memory checkpointing
        memory = MemorySaver()
        deep_researcher = deep_researcher_workflow.compile(checkpointer=memory,interrupt_before=[HUMAN_FEEDBACK])
        deep_researcher.get_graph().draw_mermaid_png(output_file_path="deep_researcher.png")
        print("Deep Researcher Workflow initialized.")
        researched_topics = []        # ### This can be parallelized and deep iterations should be possible based on users feedback
        thread = {"configurable":{"thread_id": task_id}}
        for topic in initial_research.topics:
 
            print(f"Related topic: {topic}")
            research_task: Dict[str, Any] = {
                "query": query,
                "source": source,
                "task_id": task_id,
                "topic": topic
            }            # Create properly structured input for deep researcher's ResearchState
            deep_research_input = {
                "task": research_task,
                "research_from": "WebSearch",  # Default value
                "research_state": "Planning",
                "human_feedback": "",
                "research_result": RelatedTopics(topics=[]),  # Empty initial result
                "hallucination_score": True,
                "research_reviewer_score": True,
                "response_grader_score": True
            }

            response= await deep_researcher.ainvoke(deep_research_input, thread)
            ### todo verify the response, it should be a dict with research_result
            # Add the researched topic to the list
            if hasattr(response, 'research_result') and response.research_result:
                researched_topics.extend(response.research_result.topics)
            else:
                # If no research result, keep the original topic
                researched_topics.append(topic)
        
        # Create and return RelatedTopics object
        related_topics = RelatedTopics(topics=researched_topics)

        print("Deep research completed.")
        return related_topics
        
    async def run_parallel_deep_research(self, research_state: ResearchState):
        # Placeholder for parallel deep research logic
        print("Running parallel deep research...")
        task = research_state.get("task")
        print(f"Task in deep_research {task}")
        
        # Handle nested task structure: task = {'task': {'query': '...', 'source': '...', ...}}
        if isinstance(task, dict) and "task" in task:
            actual_task = task["task"]  # Extract the inner task dictionary
        else:
            actual_task = task  # Use task directly if not nested
            
        query = actual_task.get("query") if actual_task else None
        ### todo source can be based on some plan
        source = actual_task.get("source", "web") if actual_task else "web"
        task_id = actual_task.get("task_id", "default-task-id") if actual_task else "default-task-id"
        initial_research = research_state.get("initial_research", None)
        
        deep_research_report = await self.get_deep_research_report(
            task=actual_task, initial_research=initial_research)
        
        for topic in deep_research_report.topics:
            print(f"Deep researched topic: {topic.topic} with description: {topic.description}")

        # Implement the logic to run deep research in parallel
        return {"task": actual_task, "deep_research": deep_research_report,
                "execution_status": "DeepResearch"}

if __name__ == "__main__":
    # Example usage

    agent = ResearchAgent()
    import asyncio
    asyncio.run(agent.run_initial_research({"task": {"query": "What is the advantage of AI and LLM in medical science ?", "source": "web", "verbose": True, "task_id": "task-123"}}))
    # asyncio.run(agent.run_parallel_deep_research({"task": {"query": "What is the advantage of AI and LLM in medical science ?", "source": "web", "verbose": True, "task_id": "task-123"}}))

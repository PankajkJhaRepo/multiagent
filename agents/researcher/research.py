

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
        
        try:
            # Implement the logic to get deep research report
            # This could involve more complex queries, multiple sources, etc.
            deep_researcher_workflow  = init_deep_research_team()

            ## Temporary in-memory checkpointing
            memory = MemorySaver()
            # deep_researcher = deep_researcher_workflow.compile(checkpointer=memory,interrupt_before=[HUMAN_FEEDBACK])
            
            ## without interrupting before HUMAN_FEEDBACK
            deep_researcher = deep_researcher_workflow.compile(checkpointer=memory)
            deep_researcher.get_graph().draw_mermaid_png(output_file_path="deep_researcher.png")
            print("Deep Researcher Workflow initialized.")
            
            researched_topics = []        
            # ### This can be parallelized and deep iterations should be possible based on users feedback
            thread = {"configurable":{"thread_id": f"{task_id}-deep"}}
            
            for i, topic in enumerate(initial_research.topics):
                print(f"Processing topic {i+1}/{len(initial_research.topics)}: {topic.topic}")
                # "task_id": f"{task_id}-topic-{i}",
                research_task: Dict[str, Any] = {
                    "query": query,
                    "source": source,
                    "task_id": f"{task_id}-topic-{i}",
                    "topic": topic
                }            
                
                # Create properly structured input for deep researcher's ResearchState
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

                try:
                    response = await deep_researcher.ainvoke(deep_research_input, thread)
                    print(f"Deep research response for topic {i+1}: {type(response)}")
                    print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
                    
                    # Handle response as dictionary (which it is based on console output)
                    if isinstance(response, dict) and 'research_result' in response and response['research_result']:
                        research_result = response['research_result']
                        if hasattr(research_result, 'topics') and research_result.topics:
                            researched_topics.extend(research_result.topics)
                            print(f"Added {len(research_result.topics)} topics from deep research")
                        else:
                            print(f"Research result has no topics, keeping original topic: {topic.topic}")
                            researched_topics.append(topic)
                    else:
                        # If no research result, keep the original topic
                        researched_topics.append(topic)
                        print(f"No deep research result in response, keeping original topic: {topic.topic}")
                        
                except Exception as topic_error:
                    print(f"Error processing topic {i+1}: {topic_error}")
                    import traceback
                    traceback.print_exc()
                    # Keep the original topic if deep research fails
                    researched_topics.append(topic)
            
            # Create and return RelatedTopics object
            related_topics = RelatedTopics(topics=researched_topics)
            print(f"Deep research completed with {len(researched_topics)} total topics.")
            return related_topics
            
        except Exception as e:
            print(f"Error in deep research workflow: {e}")
            print("Falling back to initial research results...")
            # Return the initial research if deep research fails
            return initial_research
        
    async def run_parallel_deep_research(self, research_state: ResearchState):
        # Placeholder for parallel deep research logic
        print("Running parallel deep research...")
        task = research_state.get("task")
        print(f"Task in deep_research: {task}")
        
        try:
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
            
            if not initial_research:
                print("❌ No initial research found in state!")
                return {"task": actual_task, "deep_research": RelatedTopics(topics=[]),
                        "execution_status": "DeepResearchFailed", "error": "No initial research data"}
            
            if not initial_research.topics:
                print("⚠️ No topics found in initial research!")
                return {"task": actual_task, "deep_research": RelatedTopics(topics=[]),
                        "execution_status": "DeepResearchSkipped", "error": "No topics to research"}
            
            print(f"📊 Starting deep research on {len(initial_research.topics)} topics...")
            
            deep_research_report = await self.get_deep_research_report(
                task=actual_task, initial_research=initial_research)
            
            print(f"📋 Deep research results:")
            for i, topic in enumerate(deep_research_report.topics, 1):
                print(f"  {i}. {topic.topic}")
                print(f"     Description: {topic.description[:100]}...")

            # Implement the logic to run deep research in parallel
            return {"task": actual_task, "deep_research": deep_research_report,
                    "execution_status": "DeepResearch"}
                    
        except Exception as e:
            print(f"❌ Error in deep research: {e}")
            import traceback
            traceback.print_exc()
            return {"task": actual_task if 'actual_task' in locals() else task, 
                    "deep_research": RelatedTopics(topics=[]),
                    "execution_status": "DeepResearchError", "error": str(e)}

if __name__ == "__main__":
    # Example usage - Complete research pipeline
    async def run_complete_research():
        agent = ResearchAgent()
        
        # Initial research state
        initial_state = {
            "task": {
                "query": "What is the advantage of AI and LLM in medical science ?", 
                "source": "web", 
                "verbose": True, 
                "task_id": "task-123"
            }
        }
        
        print("🔍 Starting Initial Research Phase...")
        initial_result = await agent.run_initial_research(initial_state)
        print(f"✅ Initial Research Completed!")
        print(f"Initial research status: {initial_result.get('execution_status')}")
        
        # Create state for deep research
        deep_research_state = {
            "task": initial_result["task"],
            "initial_research": initial_result["initial_research"],
            "execution_status": initial_result["execution_status"]
        }
        
        print("\n🔬 Starting Deep Research Phase...")
        deep_result = await agent.run_parallel_deep_research(deep_research_state)
        print(f"✅ Deep Research Completed!")
        print(f"Deep research status: {deep_result.get('execution_status')}")
        
        # Print final results summary
        print("\n📊 Research Pipeline Summary:")
        print(f"Initial topics found: {len(initial_result['initial_research'].topics)}")
        print(f"Deep research topics: {len(deep_result['deep_research'].topics)}")
        
        return {
            "initial_research": initial_result,
            "deep_research": deep_result
        }
    
    import asyncio
    final_results = asyncio.run(run_complete_research())

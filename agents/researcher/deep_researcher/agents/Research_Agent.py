from typing import Any, Dict
from agents.researcher.deep_researcher.memory.deep_researcher_state import ResearchState
from agents.researcher.memory.research_topics import Topic
from agents.researcher.deep_researcher.chains.deep_research_chain import research_chain

class ResearchAgent:
    
    def __init__(self):
        # Initialize the ResearchAgent
        print("Research Agent initialized.")

    async def run_research(self, state: ResearchState) -> Dict[str, Any]:
        print("Running research...")
        task = state.get("task")
        query = task.get("query")
        topic_obj = task.get("topic")  # This is a Topic object, not a string
        research_from = state.get("research_from")
        research_result = state.get("research_result")

        print(f"Query: {query}")
        print(f"Topic: {topic_obj}")
        print(f"Research From: {research_from}")
        print(f"Research Result: {research_result}")
        # Ensure topic_obj is a Topic instance

        
        topic_name = topic_obj.topic if isinstance(topic_obj, Topic) else str(topic_obj)
        existing_topic_details = self._search_existing_research(research_result, topic_name)

        # Check if research_result exists and prepare the request accordingly
        try:
            if existing_topic_details is not None:
                print("Previous research found, including history in prompt")
                response = research_chain.invoke({
                    "query": query,
                    "topic": topic_obj,
                    "history": existing_topic_details
                })
            else:
                print("No previous research found, proceeding without history")
                response = research_chain.invoke({
                    "query": query,
                    "topic": topic_obj,
                })
            
            # Merge the new response with existing research_result
            merged_research_result = self._merge_research_results(research_result, response)
            
            return {
                "query": query,
                "research_result": merged_research_result,
                "research_state": "DeepResearch",
            }
        except Exception as e:
            print(f"Error during research chain execution: {e}")
            # Return a minimal response to allow the system to continue
            from agents.researcher.memory.research_topics import RelatedTopics
            fallback_response = RelatedTopics(topics=[])
            return {
                "query": query,
                "research_result": fallback_response,
                "research_state": "DeepResearch",
            }

    def _search_existing_research(self, research_result, target_topic_name):
        """
        Search through research_result to find details for the specified topic name.
        
        Args:
            research_result: The existing research data (RelatedTopics object)
            target_topic_name: The topic name (string) to search for
            
        Returns:
            Topic object if found, None otherwise
        """
        if not research_result or not hasattr(research_result, 'topics'):
            print("No research_result or topics available to search")
            return None
        
        # Search through all topics in the research result
        for existing_topic in research_result.topics:
            # Case-insensitive comparison for better matching
            if existing_topic.topic.lower().strip() == target_topic_name.lower().strip():
                print(f"Exact match found for topic: {target_topic_name}")
                return existing_topic
        
        # If no exact match, try partial matching
        for existing_topic in research_result.topics:
            if (target_topic_name.lower() in existing_topic.topic.lower() or 
                existing_topic.topic.lower() in target_topic_name.lower()):
                print(f"Partial match found: '{existing_topic.topic}' for target '{target_topic_name}'")
                return existing_topic
        
        print(f"No match found for topic: {target_topic_name}")
        return None

    def _merge_research_results(self, existing_research_result, new_response):
        """
        Merge new research response with existing research_result by appending topics.
        
        Args:
            existing_research_result: RelatedTopics object with existing topics (can be None)
            new_response: RelatedTopics object from research_chain
            
        Returns:
            RelatedTopics object with merged topics
        """
        from agents.researcher.memory.research_topics import RelatedTopics, Topic
        
        # Initialize lists for merging
        existing_topics = []
        new_topics = []
        
        # Extract existing topics if available
        if existing_research_result and hasattr(existing_research_result, 'topics'):
            existing_topics = existing_research_result.topics or []
            print(f"Found {len(existing_topics)} existing topics")
        
        # Extract new topics from response
        if new_response and hasattr(new_response, 'topics'):
            new_topics = new_response.topics or []
            print(f"Found {len(new_topics)} new topics from research")
        
        # Merge topics, avoiding duplicates based on topic name
        merged_topics = list(existing_topics)  # Start with existing topics
        existing_topic_names = {topic.topic.lower().strip() for topic in existing_topics}
        
        for new_topic in new_topics:
            # Check for duplicates (case-insensitive)
            if new_topic.topic.lower().strip() not in existing_topic_names:
                merged_topics.append(new_topic)
                existing_topic_names.add(new_topic.topic.lower().strip())
                print(f"Added new topic: {new_topic.topic}")
            else:
                print(f"Skipped duplicate topic: {new_topic.topic}")
        
        print(f"Merged research result contains {len(merged_topics)} total topics")
        
        # Return new RelatedTopics object with merged topics
        return RelatedTopics(topics=merged_topics)


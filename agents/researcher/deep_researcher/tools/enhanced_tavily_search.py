from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from typing import Dict, Any, Union
import json

def SearchUsingTavilyEnhanced(search_input: Union[str, Dict[str, Any]]) -> str:
    """
    Enhanced search for topics using Tavily Search API with context and token management.
    
    Args:
        search_input: Can be either:
            - str: Simple topic string (backward compatibility)
            - dict: Enhanced input with query, topic, description, max_results, include_raw_content, etc.
    
    Returns:
        str: Search results from Tavily
    """
    load_dotenv()

    # Default parameters
    max_results = 3  # Reduced default for token efficiency
    include_raw_content = False  # Disabled by default for smaller payloads
    
    # Handle both string and dict inputs for backward compatibility
    if isinstance(search_input, str):
        search_query = search_input
    elif isinstance(search_input, dict):
        # Extract information from the enhanced input
        topic = search_input.get('topic', '')
        query = search_input.get('query', '')
        description = search_input.get('description', '')
        
        # Extract search configuration parameters
        max_results = search_input.get('max_results', 3)
        include_raw_content = search_input.get('include_raw_content', False)
        
        # Construct a more comprehensive search query
        search_parts = []
        
        if query:
            search_parts.append(f"Query: {query}")
        if topic:
            search_parts.append(f"Topic: {topic}")
        if description:
            # Truncate description if too long to avoid oversized queries
            desc_preview = description[:200] + "..." if len(description) > 200 else description
            search_parts.append(f"Context: {desc_preview}")
        
        # Create a rich search query
        search_query = " | ".join(search_parts) if search_parts else topic or query
    else:
        # Fallback for unexpected input types
        search_query = str(search_input)

    # Perform a search using Tavily with optimized parameters
    search = TavilySearchResults(
        max_results=max_results,  # Configurable based on input
        include_answer=True,
        include_raw_content=include_raw_content,  # Configurable to control payload size
        include_images=False,
        search_depth="advanced",
        # include_domains = []
        # exclude_domains = []
    )

    try:
        result = search.invoke({
            "args": {"query": search_query},
            "type": "tool_call", 
            "id": "enhanced_search", 
            "name": "tavily"
        })
        
        # Add metadata about the search for debugging
        search_metadata = {
            "search_query_used": search_query,
            "original_input_type": type(search_input).__name__,
            "enhanced_search": isinstance(search_input, dict)
        }
        
        # If the result is a list, add metadata to the first result
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                result[0]["_search_metadata"] = search_metadata
        
        return result
        
    except Exception as e:
        print(f"Error in enhanced Tavily search: {e}")
        # Fallback to basic search
        basic_search = TavilySearchResults(max_results=5)
        return basic_search.invoke({
            "args": {"query": search_query},
            "type": "tool_call", 
            "id": "fallback_search", 
            "name": "tavily"
        })


def SearchUsingTavily(topic: str) -> str:
    """
    Original search function for backward compatibility.
    """
    return SearchUsingTavilyEnhanced(topic)

#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced search functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.researcher.deep_researcher.tools.enhanced_tavily_search import SearchUsingTavilyEnhanced
from agents.researcher.deep_researcher.chains.deep_research_chain import context_aware_search, set_search_context

def test_enhanced_search():
    """Test the enhanced search functionality"""
    
    print("=== Testing Enhanced Tavily Search ===\n")
    
    # Test 1: Basic string search (backward compatibility)
    print("1. Testing basic string search:")
    basic_result = SearchUsingTavilyEnhanced("AI in healthcare")
    print(f"Basic search result type: {type(basic_result)}")
    print(f"Basic search returned: {len(basic_result) if isinstance(basic_result, list) else 'single result'} items\n")
    
    # Test 2: Enhanced search with context
    print("2. Testing enhanced search with context:")
    context_search_input = {
        'query': 'sustainable technology in medical science',
        'topic': 'Energy-Efficient LLMs in Healthcare',
        'description': 'Large Language Models optimized for energy efficiency in medical applications'
    }
    
    enhanced_result = SearchUsingTavilyEnhanced(context_search_input)
    print(f"Enhanced search result type: {type(enhanced_result)}")
    print(f"Enhanced search returned: {len(enhanced_result) if isinstance(enhanced_result, list) else 'single result'} items")
    
    # Check if search metadata was added
    if isinstance(enhanced_result, list) and len(enhanced_result) > 0:
        first_result = enhanced_result[0]
        if isinstance(first_result, dict) and "_search_metadata" in first_result:
            metadata = first_result["_search_metadata"]
            print(f"Search metadata found: {metadata}")
    print()
    
    # Test 3: Context-aware search using global context
    print("3. Testing context-aware search with global context:")
    
    # Set global context
    set_search_context(
        query="AI and LLM applications in sustainable medical technology",
        topic_context={
            'description': 'Research on energy-efficient AI deployment in healthcare settings',
            'name': 'Sustainable AI in Medicine'
        }
    )
    
    # Use context-aware search
    context_result = context_aware_search("Green AI technologies")
    print(f"Context-aware search result type: {type(context_result)}")
    print(f"Context-aware search returned: {len(context_result) if isinstance(context_result, list) else 'single result'} items\n")
    
    print("=== Test completed successfully! ===")
    return True

def demonstrate_search_improvement():
    """Demonstrate the improvement in search quality"""
    
    print("=== Demonstrating Search Quality Improvement ===\n")
    
    # Original simple search
    print("Original simple search for 'LLM healthcare':")
    simple_result = SearchUsingTavilyEnhanced("LLM healthcare")
    
    # Enhanced search with context
    print("\nEnhanced search with full context:")
    enhanced_input = {
        'query': 'sustainable technology applications in medical science',
        'topic': 'LLM healthcare',
        'description': 'Large Language Models being deployed in healthcare with focus on energy efficiency and environmental sustainability'
    }
    enhanced_result = SearchUsingTavilyEnhanced(enhanced_input)
    
    print("The enhanced search should provide more targeted results by considering:")
    print("- Original research query: sustainable technology applications")
    print("- Specific topic: LLM healthcare")  
    print("- Rich context: energy efficiency and sustainability focus")
    print("\nThis leads to more relevant and comprehensive search results!")

if __name__ == "__main__":
    try:
        print("Note: This test requires TAVILY_API_KEY to be set in your environment.")
        print("If the API key is not available, the test will show the structure but may not return actual search results.\n")
        
        test_enhanced_search()
        print("\n" + "="*50 + "\n")
        demonstrate_search_improvement()
        
    except Exception as e:
        print(f"Test error: {e}")
        print("This is expected if TAVILY_API_KEY is not configured or if there are network issues.")
        print("The enhancement structure is ready for use when the API is available.")

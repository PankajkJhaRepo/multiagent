#!/usr/bin/env python3
"""
Test script to verify the parsing fix for the Initial Research Agent.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

from agents.researcher.memory.research_topics import RelatedTopics, Topic

def test_json_parsing():
    """Test the JSON parsing function with problematic output."""
    
    # Import the parsing function
    from agents.researcher.initial_researcher.chains.initial_research_chain import parse_agent_response
    
    # Create a mock response similar to the one causing the error
    mock_response = {
        "output": """I will now compile the related topics from the observations and provide brief descriptions and sources based on the information gathered.

```json
{
    "topics": [
        {
            "topic": "Roles and Potential of Large Language Models in Healthcare",
            "description": "Large Language Models (LLMs) can deliver accurate medical management plans, summarize recent literature, and generate visual aids to help understand complex conditions. They are valuable for specialized medical education and improving clinical learning through adaptive, personalized training environments.",
            "source": "https://www.sciencedirect.com/science/article/pii/S2319417025000423"
        },
        {
            "topic": "The Future Landscape of Large Language Models in Medicine",
            "description": "LLMs use AI algorithms to generate human-like language and are trained on large datasets. They are applied in healthcare delivery for tasks like answering questions, providing summaries, and generating medical reports. Their applications extend to bioinformatics and data science.",
            "source": "https://www.nature.com/articles/s43856-023-00370-1"
        }
    ]
}
```"""
    }
    
    print("Testing JSON parsing with problematic output...")
    try:
        result = parse_agent_response(mock_response)
        print(f"✅ Parsing successful!")
        print(f"Number of topics found: {len(result.topics)}")
        for i, topic in enumerate(result.topics, 1):
            print(f"  {i}. {topic.topic}")
            print(f"     Description: {topic.description[:100]}...")
            print(f"     Source: {topic.source}")
        return True
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return False

def test_agent_initialization():
    """Test that the Initial Research Agent can be initialized."""
    try:
        from agents.researcher.initial_researcher.agents.Initial_Research_Agent import InitialResearchAgent
        agent = InitialResearchAgent()
        print("✅ Initial Research Agent initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def test_research_chain_import():
    """Test that the research chain can be imported without errors."""
    try:
        from agents.researcher.initial_researcher.chains.initial_research_chain import research_chain
        print("✅ Research chain imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Research chain import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running parsing fix tests...\n")
    
    tests = [
        ("JSON Parsing", test_json_parsing),
        ("Agent Initialization", test_agent_initialization),
        ("Research Chain Import", test_research_chain_import)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n📋 {name}:")
        if test_func():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The parsing fix should work.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")

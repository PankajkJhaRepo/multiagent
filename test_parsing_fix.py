#!/usr/bin/env python3
"""
Test script to verify the deep research chain parsing fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.researcher.memory.research_topics import Topic, RelatedTopics
from agents.researcher.deep_researcher.chains.deep_research_chain import research_chain

def test_parsing():
    """Test the parsing functionality with mock data"""
    
    # Create a test topic
    test_topic = Topic(
        topic="AI in Healthcare",
        description="Testing AI applications in medical field",
        source="test source"
    )
    
    # Test input data
    test_data = {
        "query": "Find related topics about AI and sustainable technology in medicine",
        "topic": test_topic,
        # No history for this test
    }
    
    print("Testing deep research chain...")
    print(f"Input: {test_data}")
    
    try:
        # This would normally call the LLM, but let's test our parsing logic directly
        from agents.researcher.deep_researcher.chains.deep_research_chain import parse_agent_response
        
        # Mock agent response similar to the error we saw
        mock_response = {
            "output": '''Based on the information gathered, I will now compile a list of related topics that connect AI and LLM in medical science with sustainable technology, along with detailed descriptions and sources.

```json
{
    "topics": [
        {
            "topic": "Energy-Efficient Deployment of LLMs in Healthcare",
            "description": "Large Language Models (LLMs) in healthcare, such as ChatGPT, are being optimized for energy efficiency through techniques like quantization, knowledge distillation, and pruning.",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12163604/"
        },
        {
            "topic": "Improved Healthcare Outcomes with AI and LLMs",
            "description": "AI and LLMs are transforming healthcare by improving data handling, process automation, and personalized care.",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11130776/"
        }
    ]
}
```'''
        }
        
        result = parse_agent_response(mock_response)
        print(f"Parsing successful! Result type: {type(result)}")
        print(f"Number of topics found: {len(result.topics)}")
        for i, topic in enumerate(result.topics):
            print(f"Topic {i+1}: {topic.topic}")
        
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_parsing()
    if success:
        print("✅ Test passed! The parsing fixes should work.")
    else:
        print("❌ Test failed. There may be issues with the parsing logic.")

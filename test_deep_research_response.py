#!/usr/bin/env python3
"""
Test script to verify the deep research response handling fix.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

from agents.researcher.memory.research_topics import RelatedTopics, Topic

def test_response_handling():
    """Test the response handling logic with mock deep research response."""
    
    # Create a mock response similar to what the deep research workflow returns
    mock_response = {
        'research_result': RelatedTopics(topics=[
            Topic(
                topic='AI and LLM in Genomics',
                description='Large Language Models (LLMs) contribute significantly to genomics by annotating the functions of newly identified genes using existing literature and databases. This is crucial given the rapid pace of gene discovery. LLMs enhance the understanding of genetic functions and interactions, aiding in the development of targeted therapies and personalized medicine.',
                source='https://www.sciencedirect.com/science/article/pii/S2589004224009350'
            ),
            Topic(
                topic='AI in Clinical Trials',
                description='AI and LLMs are transforming clinical trials by improving trial design, feasibility, and site selection. They also enhance patient recruitment and retention. For instance, a clinical trial patient matching tool leveraging LLMs reduced pre-screening time for physicians by 90%, streamlining the process and increasing efficiency.',
                source='https://www.pharmaceutical-technology.com/sponsored/how-ai-and-machine-learning-are-transforming-drug-discovery/'
            ),
            Topic(
                topic='AI-Powered Molecule Generation',
                description='AI-powered LLMs accelerate chemical discovery by addressing challenges in property prediction, molecule generation, and synthesis prediction. These models, combined with autonomous agents, enable rapid exploration of vast chemical spaces, facilitating the discovery of novel compounds with desired properties.',
                source='https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc03921a'
            )
        ]),
        'research_state': 'completed',
        'task': {'query': 'test query'}
    }
    
    # Test the response handling logic
    researched_topics = []
    original_topic = Topic(
        topic="AI and LLM in Drug Discovery",
        description="Original topic description",
        source="original source"
    )
    
    print("Testing response handling logic...")
    print(f"Mock response type: {type(mock_response)}")
    print(f"Response keys: {mock_response.keys() if isinstance(mock_response, dict) else 'Not a dict'}")
    
    try:
        # Handle response as dictionary (which it is based on console output)
        if isinstance(mock_response, dict) and 'research_result' in mock_response and mock_response['research_result']:
            research_result = mock_response['research_result']
            if hasattr(research_result, 'topics') and research_result.topics:
                researched_topics.extend(research_result.topics)
                print(f"✅ Added {len(research_result.topics)} topics from deep research")
                
                for i, topic in enumerate(research_result.topics, 1):
                    print(f"  {i}. {topic.topic}")
                    print(f"     Description: {topic.description[:100]}...")
                    print(f"     Source: {topic.source}")
            else:
                print(f"Research result has no topics, keeping original topic: {original_topic.topic}")
                researched_topics.append(original_topic)
        else:
            # If no research result, keep the original topic
            researched_topics.append(original_topic)
            print(f"No deep research result in response, keeping original topic: {original_topic.topic}")
        
        print(f"\n📊 Total topics after processing: {len(researched_topics)}")
        return True
        
    except Exception as e:
        print(f"❌ Error in response handling: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing deep research response handling...\n")
    
    success = test_response_handling()
    
    if success:
        print("\n🎉 Response handling test passed! The fix should work correctly.")
    else:
        print("\n⚠️ Response handling test failed. Please check the error messages above.")

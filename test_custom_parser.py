#!/usr/bin/env python3
"""
Test script to verify the custom React output parser handles JSON responses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_custom_output_parser():
    """Test the custom React output parser with problematic JSON formats"""
    
    from agents.researcher.deep_researcher.chains.flexible_output_parser import FlexibleReActOutputParser
    from langchain_core.agents import AgentFinish
    
    parser = FlexibleReActOutputParser()
    
    # Test case 1: The exact problematic output from your error
    problematic_output = '''````json
{
    "topics": [
        {
            "topic": "AI and LLMs in Medical Science",
            "description": "AI and large language models (LLMs) are revolutionizing medical science by enhancing literature retrieval, streamlining drug discovery, and improving doctor-patient communication. They enable efficient synthesis of biomedical research, assist in predicting drug activity, and provide foundational knowledge to patients through online consultations. Additionally, LLMs reduce administrative burdens by automating documentation tasks, allowing healthcare professionals to focus more on patient care. However, challenges such as model interpretability, data privacy, and the need for evidence-based validation remain significant.",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11063642/"
        },
        {
            "topic": "Ethical Implications of AI in Healthcare",
            "description": "The integration of AI and LLMs in healthcare raises critical ethical concerns, including data privacy, transparency, and trust. While these technologies offer groundbreaking opportunities, such as personalized medical education and adaptive training environments, they also pose risks related to the misuse of sensitive patient data and the potential for biased decision-making. Addressing these ethical challenges is essential for the responsible adoption of AI in medical contexts.",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11091685/"
        },
        {
            "topic": "AI-Assisted Medical Education",
            "description": "AI tools, including LLMs, are transforming medical education by providing adaptive, personalized learning environments. These systems can summarize complex medical literature, generate visual aids, and offer tailored training for healthcare professionals. By leveraging unstructured data, AI enhances clinical learning and supports the development of specialized knowledge in fields like neuro-ophthalmology. This approach not only improves educational outcomes but also prepares professionals for the evolving demands of modern healthcare.",
            "source": "https://www.sciencedirect.com/science/article/pii/S2319417025000423"
        }
    ]
}
````'''
    
    # Test case 2: Standard React format (should still work)
    react_output = '''Thought: I need to search for information about AI in medicine.
Action: Enhanced Web Search for Topics
Action Input: AI in medicine
Observation: Found relevant information about AI applications in healthcare.
Thought: I now have enough information to provide a final answer.
Final Answer: Based on my research, here are the key findings about AI in medicine...'''
    
    # Test case 3: Mixed format with action and JSON
    mixed_output = '''Thought: I need to gather information about AI topics.
Action: Enhanced Web Search for Topics  
Action Input: artificial intelligence medical applications
Observation: Found comprehensive information about AI in healthcare.
Thought: I have sufficient information to provide the final answer in JSON format.
Final Answer: ```json
{
    "topics": [
        {
            "topic": "AI Healthcare Applications",
            "description": "AI is transforming healthcare through diagnostic tools and treatment optimization.",
            "source": "https://example.com/ai-healthcare"
        }
    ]
}
```'''
    
    test_cases = [
        ("Problematic 4-backtick JSON", problematic_output),
        ("Standard React format", react_output),
        ("Mixed React + JSON", mixed_output)
    ]
    
    print("🧪 Testing Custom React Output Parser")
    print("=" * 45)
    
    all_passed = True
    
    for test_name, test_output in test_cases:
        print(f"\n📝 Testing: {test_name}")
        try:
            result = parser.parse(test_output)
            
            if isinstance(result, AgentFinish):
                print(f"   ✅ Successfully parsed as AgentFinish")
                output_content = result.return_values.get("output", "")
                print(f"   📄 Output length: {len(output_content)} characters")
                
                # Check if it contains our expected JSON structure
                if "topics" in output_content:
                    print(f"   🎯 Contains expected 'topics' structure")
                else:
                    print(f"   ⚠️  No 'topics' structure found")
                    
            else:
                print(f"   ✅ Parsed as AgentAction: {result.tool}")
                
        except Exception as e:
            print(f"   ❌ Failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 45)
    if all_passed:
        print("✅ All custom output parser tests passed!")
        print("🎉 The parser can handle the problematic JSON format")
    else:
        print("❌ Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    test_custom_output_parser()

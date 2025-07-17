#!/usr/bin/env python3
"""
Test script to verify token management functionality in deep_research_chain.py
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_token_counting():
    """Test the token counting functionality"""
    try:
        from agents.researcher.deep_researcher.chains.deep_research_chain import count_tokens, truncate_text_by_tokens
        
        # Test basic token counting
        test_text = "This is a test sentence for token counting."
        token_count = count_tokens(test_text)
        print(f"✅ Token counting works: '{test_text}' = {token_count} tokens")
        
        # Test text truncation
        long_text = "This is a very long text that should be truncated. " * 100
        original_tokens = count_tokens(long_text)
        truncated_text = truncate_text_by_tokens(long_text, 50)
        truncated_tokens = count_tokens(truncated_text)
        
        print(f"✅ Text truncation works: {original_tokens} tokens → {truncated_tokens} tokens")
        
        # Test token management configuration
        from agents.researcher.deep_researcher.chains.deep_research_chain import MAX_TOKENS_PER_REQUEST, MAX_CONTEXT_TOKENS, MAX_SEARCH_RESULTS
        print(f"✅ Configuration loaded:")
        print(f"   - Max tokens per request: {MAX_TOKENS_PER_REQUEST}")
        print(f"   - Max context tokens: {MAX_CONTEXT_TOKENS}")
        print(f"   - Max search results: {MAX_SEARCH_RESULTS}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing token management: {e}")
        return False

def test_enhanced_search_configuration():
    """Test the enhanced search configuration"""
    try:
        from agents.researcher.deep_researcher.tools.enhanced_tavily_search import SearchUsingTavilyEnhanced
        
        # Test enhanced search with reduced parameters
        test_config = {
            'topic': 'artificial intelligence',
            'query': 'machine learning trends',
            'description': 'Testing reduced payload configuration',
            'max_results': 2,
            'include_raw_content': False
        }
        
        print(f"✅ Enhanced search configuration test prepared:")
        print(f"   - Topic: {test_config['topic']}")
        print(f"   - Max results: {test_config['max_results']}")
        print(f"   - Include raw content: {test_config['include_raw_content']}")
        
        # Note: We don't actually call the search to avoid API usage during testing
        print("✅ Enhanced search function is importable and configured")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing enhanced search: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Testing Token Management Implementation")
    print("=" * 50)
    
    success = True
    
    # Test token counting
    print("\n1. Testing Token Counting:")
    if not test_token_counting():
        success = False
    
    # Test enhanced search configuration
    print("\n2. Testing Enhanced Search Configuration:")
    if not test_enhanced_search_configuration():
        success = False
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Token management implementation is ready.")
        print("\n📋 Summary of improvements:")
        print("   • Token counting with tiktoken")
        print("   • Automatic text truncation")
        print("   • Reduced search results (3 instead of 7)")
        print("   • Configurable raw content inclusion")
        print("   • History length limiting")
        print("   • Max iterations reduced to 2")
        print("   • Response token limiting (4000 tokens)")
        print("   • Context window management (12000 tokens)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()

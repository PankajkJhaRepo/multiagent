from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from agents.researcher.initial_researcher.tools.tavily_search import SearchUsingTavily
from agents.researcher.deep_researcher.tools.enhanced_tavily_search import SearchUsingTavilyEnhanced
from langchain.agents import (create_react_agent, Tool, AgentExecutor)
from langchain_core.runnables import RunnableLambda
from agents.researcher.memory.research_topics import RelatedTopics, Topic
from langchain import hub
from agents.researcher.deep_researcher.chains.flexible_output_parser import FlexibleReActOutputParser
import tiktoken
import logging

# Configure logging for token monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY');
OPENAI_API_BASEURL = os.getenv('OPENAI_API_BASE');
OPENAI_MODEL = os.getenv('LLM_MODEL');

# Token management configuration
MAX_TOKENS_PER_REQUEST = 4000  # Conservative limit for response generation
MAX_CONTEXT_TOKENS = 12000     # Leave room for response in 16k context window
MAX_SEARCH_RESULTS = 3         # Reduced from 7 to control payload size
MAX_HISTORY_LENGTH = 2000      # Limit history context

# Initialize tokenizer for the model
try:
    tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL if OPENAI_MODEL else "gpt-3.5-turbo")
except KeyError:
    # Fallback for unknown models
    tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in a text string"""
    if not text:
        return 0
    return len(tokenizer.encode(str(text)))

def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit"""
    if not text:
        return text
    
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    logger.info(f"Truncated text from {len(tokens)} to {len(truncated_tokens)} tokens")
    return truncated_text

llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    openai_api_key= OPENAI_API_KEY,
    openai_api_base= OPENAI_API_BASEURL,
    temperature=0.0,  # Set temperature to 0 for deterministic output
    max_tokens=MAX_TOKENS_PER_REQUEST,  # Limit response tokens
)

template = """Given the query {query} and related topic {topic}, generate a list of related topics with a detail description of around 150 words and source from where the detail is taken.

Previous research context (if available):
{history}

Use both web search tools and historical research tools to find comprehensive information.

CRITICAL: You must return ONLY a valid JSON object using EXACTLY three backticks. Do not use four or more backticks. Follow this exact format:

```json
{{
    "topics": [
        {{
            "topic": "<related topic>",
            "description": "<brief description of the related topic (max 150 words)>",
            "source": "<source of the information>"
        }}
    ]
}}
```

Do not include any text before or after the JSON block."""

prompt_template = PromptTemplate(
        input_variables=["query", "topic", "history"],
        template=template
        )

def search_historical_data(query: str) -> str:
    """Search through historical research data"""
    # This is a placeholder - you would implement actual historical search logic
    # You could search through:
    # - Previous research results stored in a database
    # - Cached search results
    # - Vector embeddings of past research
    
    # For now, return a placeholder response
    return f"Historical search for '{query}': No previous research found on this topic."

# Global variables to store context (this is a workaround for agent limitations)
_current_query = ""
_current_topic_context = {}

def set_search_context(query: str, topic_context: dict):
    """Set global context for searches"""
    global _current_query, _current_topic_context
    _current_query = query
    _current_topic_context = topic_context

def context_aware_search(topic_input: str) -> str:
    """
    Context-aware search that uses globally available context information
    Optimized for reduced payload size and token efficiency
    """
    global _current_query, _current_topic_context
    
    # Create enhanced search input with reduced scope
    search_context = {
        'topic': topic_input,
        'query': _current_query,
        'description': _current_topic_context.get('description', ''),
        'max_results': MAX_SEARCH_RESULTS,  # Reduced from default
        'include_raw_content': False,       # Reduce payload size
    }
    
    # Remove empty values
    search_context = {k: v for k, v in search_context.items() if v}
    
    try:
        result = SearchUsingTavilyEnhanced(search_context)
        
        # Count and log tokens for monitoring
        token_count = count_tokens(result)
        logger.info(f"Enhanced search result: {token_count} tokens for topic '{topic_input}'")
        
        # Truncate if result is too large
        if token_count > 2000:  # Reasonable limit per search result
            result = truncate_text_by_tokens(result, 2000)
            logger.info(f"Truncated search result to 2000 tokens")
        
        return result
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        # Fallback to basic search
        return SearchUsingTavily(topic_input)

def optimized_basic_search(topic_input: str) -> str:
    """
    Optimized basic search with token management
    """
    try:
        result = SearchUsingTavily(topic_input)
        
        # Count and log tokens
        token_count = count_tokens(result)
        logger.info(f"Basic search result: {token_count} tokens for topic '{topic_input}'")
        
        # Truncate if result is too large
        if token_count > 1500:  # Conservative limit for basic search
            result = truncate_text_by_tokens(result, 1500)
            logger.info(f"Truncated basic search result to 1500 tokens")
        
        return result
    except Exception as e:
        logger.error(f"Basic search failed: {e}")
        return f"Search failed for '{topic_input}': {str(e)}"

tools_for_agent = [
    Tool(
        name="Enhanced Web Search for Topics",
        func=context_aware_search,
        description="Comprehensive web search optimized for token efficiency. Considers research query, topic name, and description. Input should be the topic name. Returns focused, relevant results with controlled size.",
    ),
    Tool(
        name="Basic Web Search",
        func=optimized_basic_search,
        description="Basic web search optimized for token management. Input should be a topic string. Use for simple searches when context is not needed. Results are automatically truncated for efficiency.",
    ),
    Tool(
        name="Search Historical Research",
        func=search_historical_data,
        description="Search through previous research results and historical data. Input should be a search query string. Lightweight operation for context building.",
    )
]

# Use the original react prompt - it works reliably
react_prompt = hub.pull("hwchase17/react")

# Create custom output parser that can handle JSON responses
custom_output_parser = FlexibleReActOutputParser()

agent = create_react_agent(
    llm=llm,
    tools=tools_for_agent,
    prompt=react_prompt,
    output_parser=custom_output_parser)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools_for_agent, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2  # Reduced from 3 to control token usage
)

def format_prompt_for_agent(inputs):
    """Format the topic into the prompt template and prepare for agent executor with token management"""
    topic_input = inputs["topic"]
    query = inputs.get("query", "")
    history = inputs.get("history", None)
    
    # Handle Topic object or string
    if hasattr(topic_input, 'topic'):
        # It's a Topic object
        topic_name = topic_input.topic
        topic_description = getattr(topic_input, 'description', '')
        topic_source = getattr(topic_input, 'source', '')
        topic_str = f"{topic_name} - {topic_description}" if topic_description else topic_name
        
        # Set context for enhanced search
        topic_context = {
            'description': topic_description,
            'source': topic_source,
            'name': topic_name
        }
    else:
        # It's a string
        topic_str = str(topic_input)
        topic_context = {'name': topic_str}
    
    # Set the global search context for context-aware searches
    set_search_context(query, topic_context)
    
    # Process and truncate history if provided
    history_context = ""
    if history is not None:
        history_str = str(history)
        history_tokens = count_tokens(history_str)
        
        if history_tokens > MAX_HISTORY_LENGTH:
            history_context = truncate_text_by_tokens(history_str, MAX_HISTORY_LENGTH)
            logger.info(f"Truncated history from {history_tokens} to {MAX_HISTORY_LENGTH} tokens")
        else:
            history_context = history_str
    else:
        history_context = "No previous research history available."
    
    # Create the formatted prompt
    formatted_prompt = prompt_template.format_prompt(
        query=query, 
        topic=topic_str, 
        history=history_context
    ).to_string()
    
    # Count tokens in the complete prompt
    prompt_tokens = count_tokens(formatted_prompt)
    logger.info(f"Total prompt tokens: {prompt_tokens}")
    
    # Create optimized agent input with React format guidance
    if history is not None:
        agent_input = f"""Research Topic: {topic_str}

Previous Research Context:
{history_context}

Task: {formatted_prompt}

IMPORTANT: Please use the available tools efficiently to gather information, then provide your Final Answer in the exact JSON format specified above. 

Follow this pattern:
1. Use tools to search for information
2. When you have sufficient information, provide your Final Answer with the JSON response"""
    else:
        agent_input = f"""Research Topic: {topic_str}

Task: {formatted_prompt}

IMPORTANT: Please use the available tools efficiently to gather information, then provide your Final Answer in the exact JSON format specified above.

Follow this pattern:
1. Use tools to search for information  
2. When you have sufficient information, provide your Final Answer with the JSON response"""
    
    # Final token count check
    total_tokens = count_tokens(agent_input)
    if total_tokens > MAX_CONTEXT_TOKENS:
        logger.warning(f"Input tokens ({total_tokens}) exceed limit ({MAX_CONTEXT_TOKENS}). Truncating...")
        agent_input = truncate_text_by_tokens(agent_input, MAX_CONTEXT_TOKENS)
    
    logger.info(f"Final agent input tokens: {count_tokens(agent_input)}")
    return {"input": agent_input}

def parse_agent_response(response):
    """Parse the agent response and convert to RelatedTopics object with enhanced JSON extraction"""
    import json
    import re
    from agents.researcher.memory.research_topics import Topic
    
    # Get the output from agent_executor response
    output_text = response.get("output", "")
    
    try:
        # Try multiple extraction methods for JSON with enhanced backtick handling
        json_str = None
        
        # Method 1: Look for JSON in code blocks with any number of backticks
        json_match = re.search(r'`{3,}json\s*(\{.*?\})\s*`{3,}', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.info("Extracted JSON using method 1 (code blocks)")
        
        # Method 2: Look for JSON in code blocks without language specification
        if not json_str:
            json_match = re.search(r'`{3,}\s*(\{.*?"topics".*?\})\s*`{3,}', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("Extracted JSON using method 2 (code blocks without lang)")
        
        # Method 3: Look for JSON without code blocks
        if not json_str:
            json_match = re.search(r'\{[^{}]*"topics"[^{}]*\[[^\]]*\][^{}]*\}', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.info("Extracted JSON using method 3 (no code blocks)")
        
        # Method 4: Try to find any JSON-like structure with topics
        if not json_str:
            json_match = re.search(r'\{.*?"topics".*?\}', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.info("Extracted JSON using method 4 (loose match)")
        
        # Method 5: Extract everything between the first { and last } that contains "topics"
        if not json_str:
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and 'topics' in output_text[start_idx:end_idx+1]:
                json_str = output_text[start_idx:end_idx+1]
                logger.info("Extracted JSON using method 5 (full bracket match)")
        
        if json_str:
            # Clean up the JSON string
            json_str = json_str.strip()
            
            # Remove any trailing content after the JSON
            brace_count = 0
            clean_end = 0
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        clean_end = i + 1
                        break
            
            if clean_end > 0:
                json_str = json_str[:clean_end]
            
            logger.info(f"Attempting to parse JSON: {json_str[:100]}...")
            
            # Parse JSON and create RelatedTopics object
            parsed_data = json.loads(json_str)
            
            # Validate the structure
            if "topics" in parsed_data and isinstance(parsed_data["topics"], list):
                logger.info(f"Successfully parsed {len(parsed_data['topics'])} topics")
                return RelatedTopics(**parsed_data)
            else:
                logger.warning("Invalid JSON structure: missing 'topics' field or not a list")
                return RelatedTopics(topics=[])
        else:
            logger.warning("No valid JSON found in agent response")
            # Try to extract topic information from plain text as fallback
            topics = []
            lines = output_text.split('\n')
            current_topic = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('"topic":') or line.startswith('topic:'):
                    if current_topic:
                        topics.append(Topic(**current_topic))
                        current_topic = {}
                    topic_name = re.search(r'["\']([^"\']+)["\']', line)
                    if topic_name:
                        current_topic['topic'] = topic_name.group(1)
                elif line.startswith('"description":') or line.startswith('description:'):
                    desc = re.search(r'["\']([^"\']+)["\']', line)
                    if desc:
                        current_topic['description'] = desc.group(1)
                elif line.startswith('"source":') or line.startswith('source:'):
                    source = re.search(r'["\']([^"\']+)["\']', line)
                    if source:
                        current_topic['source'] = source.group(1)
            
            if current_topic:
                topics.append(Topic(**current_topic))
            
            logger.info(f"Fallback parsing extracted {len(topics)} topics")
            return RelatedTopics(topics=topics)
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Error parsing agent response: {e}")
        logger.error(f"Raw output: {output_text}")
        # Return empty RelatedTopics on error
        return RelatedTopics(topics=[])


# Create the research chain
research_chain = (
    RunnableLambda(format_prompt_for_agent) 
    | agent_executor
    | RunnableLambda(parse_agent_response)
)


from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from agents.researcher.initial_researcher.tools.tavily_search import SearchUsingTavily
from langchain.agents import (create_react_agent, Tool, AgentExecutor)
from langchain_core.runnables import RunnableLambda
from agents.researcher.memory.research_topics import RelatedTopics, Topic
from langchain import hub
from agents.researcher.memory.research_topics import RelatedTopics, Topic
from langchain import hub
from agents.researcher.deep_researcher.chains.custom_output_parser import CustomReActOutputParser


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY');
OPENAI_API_BASEURL = os.getenv('OPENAI_API_BASE');
OPENAI_MODEL = os.getenv('LLM_MODEL');

llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    openai_api_key= OPENAI_API_KEY,
    openai_api_base= OPENAI_API_BASEURL,
    temperature=0.0,  # Set temperature to 0 for deterministic output
)

template = """Given the query{query} and related topic {topic}, generate a list of related topics with a detail description of around 250 words and source from where the detail is taken.

Previous research context (if available):
{history}

Use both web search tools and historical research tools to find comprehensive information.

IMPORTANT: You must return ONLY a valid JSON object in the exact format below. Do not include any other text before or after the JSON.

```json
{{
    "topics": [
        {{
            "topic": "<related topic>",
            "description": "<brief description of the related topic>",
            "source": "<source of the information>"
        }}
    ]
}}
```"""

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

tools_for_agent = [
    Tool(
        name="Crawl Google for Related Topics",
        func=SearchUsingTavily,
        description="useful for when you need to find related topics on the web. Input should be a topic string.",
    ),
    Tool(
        name="Search Historical Research",
        func=search_historical_data,
        description="useful for searching through previous research results and historical data. Input should be a search query string.",
    )
]

# Use the original react prompt - it works reliably
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools_for_agent,
    prompt=react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools_for_agent, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

def format_prompt_for_agent(inputs):
    """Format the topic into the prompt template and prepare for agent executor"""
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
    else:
        # It's a string
        topic_str = str(topic_input)
    
    # Create the basic task prompt
    if history is not None:
        formatted_prompt = prompt_template.format_prompt(query=query, topic=topic_str, history=str(history)).to_string()
        # Include history context directly in the agent input
        agent_input = f"""Research Topic: {topic_str}

            Previous Research Context:
            {history}

            Task: {formatted_prompt}

            Please use the available tools to find comprehensive information, considering the previous research context."""
    else:
        formatted_prompt = prompt_template.format_prompt(query=query, topic=topic_str, history="No previous research history available.").to_string()
        # No history context
        agent_input = f"""Research Topic: {topic_str}

        Task: {formatted_prompt}

        Please use the available tools to find comprehensive information."""
            
    return {"input": agent_input}

def parse_agent_response(response):
    """Parse the agent response and convert to RelatedTopics object"""
    import json
    import re
    from agents.researcher.memory.research_topics import Topic
    
    # Get the output from agent_executor response
    output_text = response.get("output", "")
    
    try:
        # Try multiple extraction methods for JSON
        json_str = None
        
        # Method 1: Look for JSON in code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        
        # Method 2: Look for JSON without code blocks
        if not json_str:
            json_match = re.search(r'\{[^{}]*"topics"[^{}]*\[[^\]]*\][^{}]*\}', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
        
        # Method 3: Try to find any JSON-like structure
        if not json_str:
            json_match = re.search(r'\{.*?"topics".*?\}', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
        
        if json_str:
            # Clean up the JSON string
            json_str = json_str.strip()
            # Parse JSON and create RelatedTopics object
            parsed_data = json.loads(json_str)
            
            # Validate the structure
            if "topics" in parsed_data and isinstance(parsed_data["topics"], list):
                return RelatedTopics(**parsed_data)
            else:
                print("Invalid JSON structure: missing 'topics' field or not a list")
                return RelatedTopics(topics=[])
        else:
            print("No valid JSON found in agent response")
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
            
            return RelatedTopics(topics=topics)
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing agent response: {e}")
        print(f"Raw output: {output_text}")
        # Return empty RelatedTopics on error
        return RelatedTopics(topics=[])


# Create the research chain
research_chain = (
    RunnableLambda(format_prompt_for_agent) 
    | agent_executor
    | RunnableLambda(parse_agent_response)
)


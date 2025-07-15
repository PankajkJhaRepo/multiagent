from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


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

class GradeResearchTopics(BaseModel):
    """
    Binary score for hallucination present in generated answer.
    """
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'True' for grounded, 'False' for not grounded."
    )

structured_llm_grader = llm.with_structured_output(GradeResearchTopics)

system = """
You are and expert reviewer of the given research topic.
You will review topic, description and source. Based on the parameters  give a binary score 'True' or 'False'.
'True' means that the research topics are relevant and well-grounded, 'False' means they are not.
In case you find the topics not relevant or not well-grounded, then report 'False'.
"""


research_reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User query: {query}\n Related topics: {topic}, Description of topic: {description}, Source of the topic:{source}"),
        ("assistant", "Grade the topic, description and source with a binary score 'True' or 'False'.")
    ]
)

research_reviewer_chain = research_reviewer_prompt | structured_llm_grader
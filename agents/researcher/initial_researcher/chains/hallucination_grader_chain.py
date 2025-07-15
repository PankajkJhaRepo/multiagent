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

class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generated answer.
    """
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'True' for grounded, 'False' for not grounded."
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generated topics related to query is grounded in / supported by a set of retrieved topics. \n 
     Give a binary score 'True' or 'False'. 'True' means that the answer is grounded in / supported by the set of topics."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User query: {query}\n Related topics: {topics}"),
        ("assistant", "Grade the generation with a binary score 'True' or 'False'.")
    ]
)

hallucination_grader_chain = hallucination_prompt | structured_llm_grader
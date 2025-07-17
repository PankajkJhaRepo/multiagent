import sys
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.orchestrator import OrchestratorAgent
from request import RequestedTask

def test_foo() -> None:
    assert 1 == 1

@pytest.fixture
def orchestrator_request() -> RequestedTask:
    """Fixture to create a sample request"""
    return {
        "task_id": "test-123",
        "query": "Write a summary about AI",
        "max_sections": 3,
        "publish_format": "markdown",
        "include_human_feedback": False,
        "follow_guidelines": True,
        "model": "gpt-4",
        "guidelines": ["Keep it simple", "Be concise"],
        "verbose": True
    }

@pytest.fixture
def orchestrator(orchestrator_request):
    """Fixture to create an OrchestratorAgent instance"""
    return OrchestratorAgent(orchestrator_request)

def test_orchestrator_initialization(orchestrator, orchestrator_request):
    """Test if orchestrator is initialized correctly"""
    assert isinstance(orchestrator, OrchestratorAgent)
    assert hasattr(orchestrator, 'request')
    assert orchestrator.request == orchestrator_request

def test_generate_task_id(orchestrator):
    """Test if task_id is generated correctly"""
    task_id = orchestrator._generate_task_id()
    assert isinstance(task_id, int)
    assert task_id > 0

def test_initialize_agents(orchestrator):
    """Test if agents are initialized correctly"""
    agents = orchestrator._initialize_agents()
    expected_agents = ["writer", "planner", "research", "publisher", "human"]
    assert all(agent in agents for agent in expected_agents)
    assert len(agents) == len(expected_agents)

def test_init_research_team(orchestrator):
    """Test if research team workflow is created correctly"""
    workflow = orchestrator.init_research_team()
    assert workflow is not None
    # Check if the workflow has all required nodes
    expected_nodes = ["researcher", "deep_researcher", "writer", "publisher", "human"]
    actual_nodes = list(workflow.nodes.keys())
    print(f"Expected nodes: {expected_nodes}")
    print(f"Actual nodes: {actual_nodes}")
    assert all(node in workflow.nodes for node in expected_nodes)

# @pytest.mark.asyncio
# async def test_run_research_task(orchestrator):
#     """Test if research task runs correctly"""
#     with patch('langgraph.graph.StateGraph.compile') as mock_compile:
#         mock_chain = Mock()
#         mock_chain.ainvoke.return_value = {"status": "success", "result": "test result"}
#         mock_compile.return_value = mock_chain
        
#         result = await orchestrator.run_research_task()
#         assert result is not None
#         assert mock_chain.ainvoke.called




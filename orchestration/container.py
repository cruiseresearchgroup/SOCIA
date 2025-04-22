"""
Container: Dependency injection container for managing agent instances.
"""

import logging
import yaml
from dependency_injector import containers, providers
from typing import Dict, Any

from agents.task_understanding.agent import TaskUnderstandingAgent
from agents.data_analysis.agent import DataAnalysisAgent
from agents.model_planning.agent import ModelPlanningAgent
from agents.code_generation.agent import CodeGenerationAgent
from agents.code_verification.agent import CodeVerificationAgent
from agents.simulation_execution.agent import SimulationExecutionAgent
from agents.result_evaluation.agent import ResultEvaluationAgent
from agents.feedback_generation.agent import FeedbackGenerationAgent
from agents.iteration_control.agent import IterationControlAgent

class AgentContainer(containers.DeclarativeContainer):
    """
    Dependency injection container for managing agent instances.
    
    This container is responsible for:
    1. Loading the configuration
    2. Creating and configuring agent instances
    3. Managing agent lifecycles
    4. Providing dependency injection for the WorkflowManager
    """
    
    # Configuration provider
    config = providers.Configuration()
    
    # Configuration loader
    config_loader = providers.Resource(
        lambda path: yaml.safe_load(open(path, 'r')),
        providers.Callable(lambda: "config.yaml")
    )
    
    # Shared logger provider
    logger = providers.Singleton(
        lambda: logging.getLogger("SOCIA.Container")
    )
    
    # Agents factory providers
    task_understanding_agent = providers.Factory(
        TaskUnderstandingAgent,
        config=config.agents.task_understanding
    )
    
    data_analysis_agent = providers.Factory(
        DataAnalysisAgent,
        config=config.agents.data_analysis
    )
    
    model_planning_agent = providers.Factory(
        ModelPlanningAgent,
        config=config.agents.model_planning
    )
    
    code_generation_agent = providers.Factory(
        CodeGenerationAgent,
        config=config.agents.code_generation
    )
    
    code_verification_agent = providers.Factory(
        CodeVerificationAgent,
        config=config.agents.code_verification
    )
    
    simulation_execution_agent = providers.Factory(
        SimulationExecutionAgent,
        config=config.agents.simulation_execution
    )
    
    result_evaluation_agent = providers.Factory(
        ResultEvaluationAgent,
        config=config.agents.result_evaluation
    )
    
    feedback_generation_agent = providers.Factory(
        FeedbackGenerationAgent,
        config=config.agents.feedback_generation
    )
    
    iteration_control_agent = providers.Factory(
        IterationControlAgent,
        config=config.agents.iteration_control
    )
    
    # Agent provider dictionary for bulk access
    agent_providers = providers.Dict(
        {
            "task_understanding": task_understanding_agent,
            "data_analysis": data_analysis_agent,
            "model_planning": model_planning_agent,
            "code_generation": code_generation_agent,
            "code_verification": code_verification_agent,
            "simulation_execution": simulation_execution_agent,
            "result_evaluation": result_evaluation_agent,
            "feedback_generation": feedback_generation_agent,
            "iteration_control": iteration_control_agent
        }
    ) 
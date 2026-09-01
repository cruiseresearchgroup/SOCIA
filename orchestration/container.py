"""
Container: Dependency injection container for managing agent instances.
"""

import logging
import yaml
from dependency_injector import containers, providers
from typing import Dict, Any

from agents.task_understanding.agent import TaskUnderstandingAgent
from agents.task_understanding_odd.agent import TaskUnderstandingAgent as TaskUnderstandingOddAgent
from agents.data_analysis.agent import DataAnalysisAgent
from agents.data_analysis_odd.agent import DataAnalysisAgent as DataAnalysisOddAgent
from agents.data_analysis_ace.agent import DataAnalysisAgent as DataAnalysisAceAgent
from agents.model_planning.agent import ModelPlanningAgent
from agents.code_generation.agent import CodeGenerationAgent
from agents.code_generation_odd.agent import CodeGenerationAgent as CodeGenerationOddAgent
from agents.code_generation_ace.agent import CodeGenerationAgent as CodeGenerationAceAgent
from agents.code_generation_alpha.agent import CodeGenerationAgent as CodeGenerationAlphaAgent
from agents.code_generation_gsim.agent import CodeGenerationAgent as CodeGenerationGsimAgent
from agents.code_verification.agent import CodeVerificationAgent
from agents.simulation_execution.agent import SimulationExecutionAgent
from agents.simulation_execution_ace.agent import SimulationExecutionAgent as SimulationExecutionAceAgent
from agents.simulation_execution_alpha.agent import SimulationExecutionAgent as SimulationExecutionAlphaAgent
from agents.simulation_execution_gsim.agent import SimulationExecutionAgent as SimulationExecutionGsimAgent
from agents.result_evaluation.agent import ResultEvaluationAgent
from agents.feedback_generation.agent import FeedbackGenerationAgent
from agents.feedback_generation_odd.agent import FeedbackGenerationOddAgent
from agents.feedback_generation_ace.agent import FeedbackGenerationAgent as FeedbackGenerationAceAgent
from agents.feedback_generation_alpha.agent import FeedbackGenerationAgent as FeedbackGenerationAlphaAgent
from agents.feedback_generation_gsim.agent import FeedbackGenerationAgent as FeedbackGenerationGsimAgent
from agents.iteration_control.agent import IterationControlAgent
from agents.iteration_control_ace.agent import IterationControlAgent as IterationControlAceAgent

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
    
    # Shared output path provider
    output_path = providers.Callable(lambda: "output")
    
    # Helper function for providing default configs
    def get_default_config(agent_name):
        defaults = {
            "task_understanding": {"prompt_template": "templates/task_understanding_prompt.txt", "output_format": "json"},
            "task_understanding_odd": {"prompt_template": "templates/task_understanding_prompt.txt", "output_format": "json"},
            "data_analysis": {"prompt_template": "templates/data_analysis_prompt.txt", "output_format": "json"},
            "data_analysis_odd": {"prompt_template": "templates/data_analysis_odd_prompt.txt", "output_format": "json"},
            "data_analysis_ace": {"prompt_template": "templates/data_analysis_odd_prompt.txt", "output_format": "json"},
            "data_analysis_nabla": {"prompt_template": "templates/data_analysis_nabla_prompt.txt", "output_format": "json"},
            "model_planning": {"prompt_template": "templates/model_planning_prompt.txt", "output_format": "json"},
            "code_generation": {"prompt_template": "templates/code_generation_prompt.txt", "output_format": "python"},
            "code_generation_odd": {"prompt_template": "templates/code_generation_prompt.txt", "output_format": "python"},
            "code_generation_ace": {"prompt_template": "templates/code_generation_odd_prompt.txt", "output_format": "python"},
            "code_generation_nabla": {"prompt_template": "templates/code_generation_nabla_prompt.txt", "output_format": "python"},
            "code_generation_alpha": {"prompt_template": "templates/code_generation_ace_prompt.txt", "output_format": "python"},
            "code_generation_gsim": {"prompt_template": "templates/code_generation_ace_prompt.txt", "output_format": "python"},
            "data_analysis_gsim": {"prompt_template": "templates/data_analysis_alpha_prompt.txt", "output_format": "json"},
            "simulation_execution_gsim": {"prompt_template": "templates/simulation_execution_prompt.txt", "output_format": "json"},
            "feedback_generation_gsim": {"prompt_template": "templates/feedback_generation_gsim_prompt.txt", "output_format": "json"},
            "code_verification": {"prompt_template": "templates/code_verification_prompt.txt", "output_format": "json"},
            "simulation_execution": {"prompt_template": "templates/simulation_execution_prompt.txt", "output_format": "json"},
            "simulation_execution_ace": {"prompt_template": "templates/simulation_execution_prompt.txt", "output_format": "json"},
            "simulation_execution_nabla": {"prompt_template": "templates/simulation_execution_prompt.txt", "output_format": "json"},
            "result_evaluation": {"prompt_template": "templates/result_evaluation_prompt.txt", "output_format": "json"},
            "feedback_generation": {"prompt_template": "templates/feedback_generation_prompt.txt", "output_format": "json"},
            "feedback_generation_odd": {"prompt_template": "templates/feedback_generation_prompt.txt", "output_format": "json"},
            "feedback_generation_ace": {"prompt_template": "templates/feedback_generation_ace_prompt.txt", "output_format": "json"},
            "feedback_generation_nabla": {"prompt_template": "templates/feedback_generation_nabla_prompt.txt", "output_format": "json"},
            "feedback_generation_alpha": {"prompt_template": "templates/feedback_generation_alpha_prompt.txt", "output_format": "json"},
            "iteration_control": {"prompt_template": "templates/iteration_control_prompt.txt", "output_format": "json"},
            "iteration_control_ace": {"prompt_template": "templates/iteration_control_prompt.txt", "output_format": "json"},
            "iteration_control_nabla": {"prompt_template": "templates/iteration_control_prompt.txt", "output_format": "json"}
        }
        return defaults.get(agent_name, {})

    # Agents factory providers
    task_understanding_agent = providers.Factory(
        TaskUnderstandingAgent,
        config=config.agents.task_understanding
    )
    
    task_understanding_odd_agent = providers.Factory(
        TaskUnderstandingOddAgent,
        config=config.agents.task_understanding_odd
    )
    
    data_analysis_agent = providers.Factory(
        DataAnalysisAgent,
        config=config.agents.data_analysis,
        output_path=output_path
    )
    
    data_analysis_odd_agent = providers.Factory(
        DataAnalysisOddAgent,
        config=config.agents.data_analysis_odd,
        output_path=output_path
    )
    
    data_analysis_ace_agent = providers.Factory(
        DataAnalysisAceAgent,
        config=config.agents.data_analysis_ace,
        output_path=output_path
    )

    data_analysis_nabla_agent = providers.Factory(
        DataAnalysisAceAgent,
        config=config.agents.data_analysis_nabla,
        output_path=output_path
    )

    data_analysis_gsim_agent = providers.Factory(
        DataAnalysisAceAgent,
        config=config.agents.data_analysis_gsim,
        output_path=output_path
    )
    
    model_planning_agent = providers.Factory(
        ModelPlanningAgent,
        config=config.agents.model_planning
    )
    
    code_generation_agent = providers.Factory(
        CodeGenerationAgent,
        config=config.agents.code_generation
    )
    
    code_generation_odd_agent = providers.Factory(
        CodeGenerationOddAgent,
        config=config.agents.code_generation_odd
    )
    
    code_generation_ace_agent = providers.Factory(
        CodeGenerationAceAgent,
        config=config.agents.code_generation_ace
    )

    code_generation_nabla_agent = providers.Factory(
        CodeGenerationAceAgent,
        config=config.agents.code_generation_nabla
    )
    
    code_generation_alpha_agent = providers.Factory(
        CodeGenerationAlphaAgent,
        config=config.agents.code_generation_alpha
    )

    code_generation_gsim_agent = providers.Factory(
        CodeGenerationGsimAgent,
        config=config.agents.code_generation_gsim
    )
    
    code_verification_agent = providers.Factory(
        CodeVerificationAgent,
        output_dir=providers.Callable(lambda op: f"{op}/verification", output_path),
        config=config.agents.code_verification
    )
    
    simulation_execution_agent = providers.Factory(
        SimulationExecutionAgent,
        output_dir=providers.Callable(lambda op: f"{op}/execution", output_path),
        config=config.agents.simulation_execution
    )
    
    simulation_execution_ace_agent = providers.Factory(
        SimulationExecutionAceAgent,
        output_dir=providers.Callable(lambda op: f"{op}/execution", output_path),
        config=config.agents.simulation_execution_ace
    )

    simulation_execution_nabla_agent = providers.Factory(
        SimulationExecutionAceAgent,
        output_dir=providers.Callable(lambda op: f"{op}/execution", output_path),
        config=config.agents.simulation_execution_nabla
    )
    
    # Alpha simulation execution uses its own implementation/config
    simulation_execution_alpha_agent = providers.Factory(
        SimulationExecutionAlphaAgent,
        output_dir=providers.Callable(lambda op: f"{op}/execution", output_path),
        config=config.agents.simulation_execution_alpha
    )

    simulation_execution_gsim_agent = providers.Factory(
        SimulationExecutionGsimAgent,
        output_dir=providers.Callable(lambda op: f"{op}/execution", output_path),
        config=config.agents.simulation_execution_gsim
    )
    
    result_evaluation_agent = providers.Factory(
        ResultEvaluationAgent,
        config=config.agents.result_evaluation
    )
    
    feedback_generation_agent = providers.Factory(
        FeedbackGenerationAgent,
        config=config.agents.feedback_generation
    )
    
    feedback_generation_odd_agent = providers.Factory(
        FeedbackGenerationOddAgent,
        config=config.agents.feedback_generation_odd
    )
    
    feedback_generation_ace_agent = providers.Factory(
        FeedbackGenerationAceAgent,
        config=config.agents.feedback_generation_ace
    )

    feedback_generation_nabla_agent = providers.Factory(
        FeedbackGenerationAceAgent,
        config=config.agents.feedback_generation_nabla
    )
    
    feedback_generation_alpha_agent = providers.Factory(
        FeedbackGenerationAlphaAgent,
        config=config.agents.feedback_generation_alpha
    )

    feedback_generation_gsim_agent = providers.Factory(
        FeedbackGenerationGsimAgent,
        config=config.agents.feedback_generation_gsim
    )
    
    iteration_control_agent = providers.Factory(
        IterationControlAgent,
        config=config.agents.iteration_control
    )
    
    iteration_control_ace_agent = providers.Factory(
        IterationControlAceAgent,
        config=config.agents.iteration_control_ace
    )

    iteration_control_nabla_agent = providers.Factory(
        IterationControlAceAgent,
        config=config.agents.iteration_control_nabla
    )
    
    # Agent provider dictionary for bulk access
    agent_providers = providers.Dict(
        {
            "task_understanding": task_understanding_agent,
            "task_understanding_odd": task_understanding_odd_agent,
            "data_analysis": data_analysis_agent,
            "data_analysis_odd": data_analysis_odd_agent,
            "data_analysis_ace": data_analysis_ace_agent,
            "data_analysis_nabla": data_analysis_nabla_agent,
            "model_planning": model_planning_agent,
            "code_generation": code_generation_agent,
            "code_generation_odd": code_generation_odd_agent,
            "code_generation_ace": code_generation_ace_agent,
            "code_generation_nabla": code_generation_nabla_agent,
            "code_generation_alpha": code_generation_alpha_agent,
            "code_generation_gsim": code_generation_gsim_agent,
            "data_analysis_gsim": data_analysis_gsim_agent,
            "simulation_execution_gsim": simulation_execution_gsim_agent,
            "feedback_generation_gsim": feedback_generation_gsim_agent,
            "code_verification": code_verification_agent,
            "simulation_execution": simulation_execution_agent,
            "simulation_execution_ace": simulation_execution_ace_agent,
            "simulation_execution_nabla": simulation_execution_nabla_agent,
            "simulation_execution_alpha": simulation_execution_alpha_agent,
            "result_evaluation": result_evaluation_agent,
            "feedback_generation": feedback_generation_agent,
            "feedback_generation_odd": feedback_generation_odd_agent,
            "feedback_generation_ace": feedback_generation_ace_agent,
            "feedback_generation_nabla": feedback_generation_nabla_agent,
            "feedback_generation_alpha": feedback_generation_alpha_agent,
            "iteration_control": iteration_control_agent,
            "iteration_control_ace": iteration_control_ace_agent,
            "iteration_control_nabla": iteration_control_nabla_agent
        }
    ) 

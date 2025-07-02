#!/usr/bin/env python3
# SOCIA: Simulation Orchestration for City Intelligence and Agents

import argparse
import logging
import os
import sys
import yaml
from orchestration.workflow_manager import WorkflowManager
from orchestration.container import AgentContainer
import numpy as np
import matplotlib.pyplot as plt
from models.epidemic_model import create_epidemic_simulation
from utils.llm_utils import load_api_key
from dependency_injector.wiring import Provide, inject

def setup_logging(output_path=None):
    """Configure logging for the application."""
    # Try to read logging level from config file
    log_level = logging.INFO  # Default level is INFO
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            log_level_str = config.get("logging", {}).get("level", "INFO")
            log_level = getattr(logging, log_level_str)
    except Exception as e:
        # If there's an error reading the config, use default level
        print(f"Warning: Could not read logging level from config: {e}")
    
    # Create handlers list
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # If output path is provided, add a file handler
    if output_path:
        try:
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            # Create log file path
            log_file_path = os.path.join(output_path, "socia.log")
            
            # Add file handler to handlers list
            handlers.append(logging.FileHandler(log_file_path))
            print(f"Logging to file: {log_file_path}")
        except Exception as e:
            print(f"Warning: Could not set up logging to file: {e}")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('SOCIA')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SOCIA: LLM-driven multi-agent social simulation generator')
    parser.add_argument('--task', type=str, help='Description of the simulation task')
    parser.add_argument('--task-file', type=str, help='Path to task description JSON file containing task_objective, data_folder, data_files, and evaluation_metrics')
    parser.add_argument('--output', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to configuration file')
    parser.add_argument('--iterations', type=int, default=3, help='Hard maximum number of iterations (multiple of 3 recommended); soft window starts at 3.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--run-example', action='store_true', help='Run the example epidemic simulation')
    parser.add_argument('--setup-api-key', action='store_true', help='Setup OpenAI API key')
    parser.add_argument('--auto', action='store_true', default=False, help='Enable automatic mode; when False, user will be prompted to input feedback manually in each iteration')
    parser.add_argument('--mode', type=str, default='lite', choices=['lite', 'medium', 'full'], help='Workflow mode: lite, medium, or full.')
    parser.add_argument('--selfloop', type=int, default=0, help='Number of self-checking iterations for code generation; 0 disables self-checking.')
    
    args = parser.parse_args()
    
    # Validate that --task is provided when not running the example
    if not args.run_example and not args.setup_api_key and not args.task:
        parser.error("--task is required unless --run-example or --setup-api-key is specified")
    
    return args

def setup_container(config_path: str) -> AgentContainer:
    """
    Set up and configure the dependency injection container.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configured AgentContainer instance
    """
    # Create container instance
    container = AgentContainer()
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            container.config.from_dict(config)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        # Use minimal default configuration
        container.config.from_dict({
            "system": {"name": "SOCIA", "version": "0.1.0"},
            "agents": {
                "task_understanding": {"prompt_template": "templates/task_understanding_prompt.txt"},
                "data_analysis": {"prompt_template": "templates/data_analysis_prompt.txt"},
                "model_planning": {"prompt_template": "templates/model_planning_prompt.txt"},
                "code_generation": {"prompt_template": "templates/code_generation_prompt.txt"},
                "code_verification": {"prompt_template": "templates/code_verification_prompt.txt"},
                "simulation_execution": {"prompt_template": "templates/simulation_execution_prompt.txt"},
                "result_evaluation": {"prompt_template": "templates/result_evaluation_prompt.txt"},
                "feedback_generation": {"prompt_template": "templates/feedback_generation_prompt.txt"},
                "iteration_control": {"prompt_template": "templates/iteration_control_prompt.txt"}
            }
        })
    
    # Wire the container for dependency injection
    container.wire(modules=[
        sys.modules[__name__],                # main.py
        "orchestration.workflow_manager",     # WorkflowManager class
        "agents.task_understanding.agent",    # Individual agent modules
        "agents.data_analysis.agent",
        "agents.model_planning.agent",
        "agents.code_generation.agent",
        "agents.code_verification.agent",
        "agents.simulation_execution.agent",
        "agents.result_evaluation.agent",
        "agents.feedback_generation.agent",
        "agents.iteration_control.agent",
        "agents.base_agent",                  # Base agent class
        "utils.llm_utils"                     # LLM utilities
    ])
    
    return container

def check_api_key() -> bool:
    """
    Check if OpenAI API key is configured in keys.py
    
    Returns:
        bool: True if API key is configured, False otherwise
    """
    api_key = load_api_key("OPENAI_API_KEY")
    return api_key is not None

def plot_results(metrics_history):
    """Plot the simulation results."""
    # Extract data from metrics history
    time_steps = [m['time_step'] for m in metrics_history]
    susceptible = [m['susceptible_count'] for m in metrics_history]
    infected = [m['infected_count'] for m in metrics_history]
    recovered = [m['recovered_count'] for m in metrics_history]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    ax.plot(time_steps, susceptible, 'b-', label='Susceptible')
    ax.plot(time_steps, infected, 'r-', label='Infected')
    ax.plot(time_steps, recovered, 'g-', label='Recovered')
    
    # Add labels and legend
    ax.set_title('SIR Epidemic Model Simulation')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Population')
    ax.legend()
    ax.grid(True)
    
    # Save and show the figure
    plt.savefig('epidemic_simulation_results.png')
    plt.show()

def plot_results_zh(metrics_history):
    """Plot simulation results with Chinese labels (placeholder for now)."""
    # Currently just calls the English version
    plot_results(metrics_history)

def run_epidemic_example():
    # Get system locale
    import locale
    system_locale = locale.getdefaultlocale()[0]
    
    # Configuration for the simulation
    config = {
        'population_size': 1000,
        'initial_infected': 5,
        'transmission_rate': 0.3,
        'recovery_rate': 0.1,
        'contact_radius': 0.02,
        'seed': 42
    }
    
    # Create and initialize the simulation
    simulation = create_epidemic_simulation(config)
    simulation.initialize()
    
    # Run the simulation for 100 time steps
    results = simulation.run(100)
    
    # Print final state
    final_state = results['final_state']
    print(f"Simulation completed after {final_state['time_step']} time steps")
    print(f"Final state:")
    print(f"  Susceptible: {final_state['susceptible_count']}")
    print(f"  Infected: {final_state['infected_count']}")
    print(f"  Recovered: {final_state['recovered_count']}")
    
    # Plot the results
    plot_results(results['metrics_history'])

@inject
def run_workflow(
    args,
    logger,
    agent_container: AgentContainer = Provide[AgentContainer]
):
    """
    Run the workflow with dependency injection.
    
    Args:
        args: Command line arguments
        logger: Application logger
        agent_container: Dependency injection container
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Initialize workflow manager with the container
        workflow_manager = WorkflowManager(
            task_description=args.task,
            data_path=args.task_file,
            output_path=args.output,
            config_path=args.config,
            max_iterations=args.iterations,
            auto_mode=args.auto,
            agent_container=agent_container,
            mode=args.mode,
            selfloop=args.selfloop
        )
        
        # Run the workflow
        result = workflow_manager.run()
        
        # Log workflow completion depending on whether code file was generated
        if os.path.exists(result['code_path']):
            logger.info(f"Workflow completed. Results available at: {result['code_path']}")
        else:
            logger.warning(f"Workflow ended but code file not found at: {result['code_path']}. Please check the logs and artifacts.")
        return 0
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 1

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.output)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # If --setup-api-key is specified, just run the setup script
    if args.setup_api_key:
        # Run the setup script
        os.system("python setup_api_key.py")
        return 0
    
    # If --run-example is specified, run the example epidemic simulation
    if args.run_example:
        run_epidemic_example()
        return 0
    
    # For LLM-dependent features, check API key in keys.py
    if not check_api_key():
        logger.error("OpenAI API key not found in keys.py")
        logger.info("Please set up your API key using: python main.py --setup-api-key")
        return 1
    
    # Set up the dependency injection container
    container = setup_container(args.config)
    
    # Run the workflow with dependency injection
    return run_workflow(args, logger)

if __name__ == '__main__':
    sys.exit(main())

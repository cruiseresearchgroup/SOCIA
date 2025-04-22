<p align="center">
  <img src="docs/images/socia_logo.png" alt="SOCIA Logo" width="200px" />
</p>

# SOCIA: Simulation Orchestration for City Intelligence and Agents

An LLM-driven multi-agent social simulation generator that automatically creates simulation environments based on user requirements and data.

<!-- Illustration Image Placeholder -->
<p align="center">
  <img src="docs/images/socia_architecture.png" alt="SOCIA System Architecture" width="40%" />
</p>
<!-- Replace this with your actual architecture diagram -->

## Architecture

The system implements a distributed multi-agent architecture where each agent performs specialized tasks:

1. **Task Understanding Agent**: Parses user requirements
2. **Data Analysis Agent**: Analyzes real-world data
3. **Model Planning Agent**: Designs simulation approach and structure
4. **Code Generation Agent**: Transforms plans into Python code
5. **Code Verification Agent**: Tests generated code
6. **Simulation Execution Agent**: Runs simulations in sandbox
7. **Result Evaluation Agent**: Compares simulation with real data
8. **Feedback Generation Agent**: Creates improvement suggestions
9. **Iteration Control Agent**: Coordinates the workflow

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### API Key

The system uses OpenAI's API for the LLM-based agents. The API key is hardcoded in the `keys.py` file:

```python
# keys.py
OPENAI_API_KEY = "your-key-here"
```

You can use the included setup script to configure your API key:
```bash
python main.py --setup-api-key
```

This script will create or update the `keys.py` file with your API key.

## Usage

### Example Commands

Here are some example commands to help you get started with SOCIA:

```bash
# Display help information and available command-line options
python main.py --help
```

This command shows all available options and parameters for running SOCIA, including task description, output directories, and other configuration options.

```bash
# Run the built-in example simulation
python main.py --run-example
```

This command runs a pre-configured epidemic simulation example to demonstrate the system's capabilities. It's a good starting point to verify that your installation is working correctly.

```bash
# Run a full workflow with a specific task
python main.py --task "Create an epidemic simulation for a city with 1000 people" --output "./my_sim_output"
```

This command initiates the full SOCIA workflow:
1. LLM agents analyze the task description
2. The system designs and generates an epidemic simulation for a city of 1000 people
3. The simulation is executed and results are saved to the "./my_sim_output" directory
4. Visualizations and analysis are automatically generated

Use this command pattern when you want to create custom simulations based on your specific requirements. You can customize the task description to focus on different urban simulation scenarios.

### Running Custom Simulations

When running a simulation with a custom task, the system:
1. Parses your requirements using natural language processing
2. Selects appropriate models based on the task context
3. Generates and validates the simulation code
4. Executes the simulation in a controlled environment
5. Produces results and visualizations

For more advanced usage, see the examples directory for sample scripts that demonstrate specific simulation types.

## Project Structure

- `agents/`: Individual agent implementations
- `core/`: Core simulation framework
- `data/`: Data management utilities
- `models/`: Simulation model templates
- `orchestration/`: Agent coordination
- `utils/`: Utility functions
- `tests/`: Test suite 

## Dependency Injection

SOCIA uses the `dependency-injector` framework to manage agent lifecycles and dependencies. This architectural pattern provides several benefits:

### How Dependency Injection Works in SOCIA

1. **Container Definition**: The `AgentContainer` class in `orchestration/container.py` serves as a central registry for all agents and their dependencies:

```python
class AgentContainer(containers.DeclarativeContainer):
    # Configuration provider
    config = providers.Configuration()
    
    # Agent factory providers
    task_understanding_agent = providers.Factory(
        TaskUnderstandingAgent,
        config=config.agents.task_understanding
    )
    # Other agents...
```

2. **Instance Management**: The container creates and manages agent instances through different provider types:
   - `Factory`: Creates a new instance each time it's requested
   - `Singleton`: Creates a single instance that's reused throughout the application
   - `Resource`: Manages resources that require explicit acquisition and release

3. **Wiring**: The container is connected to modules that use the `@inject` decorator:

```python
container.wire(modules=[sys.modules[__name__], "orchestration.workflow_manager"])
```

4. **Dependency Injection**: Components request dependencies through constructor parameters:

```python
@inject
def __init__(
    self,
    task_description: str,
    agent_container: AgentContainer = Provide[AgentContainer]
):
```

### Benefits of Dependency Injection in SOCIA

- **Decoupling**: Components depend on abstractions rather than concrete implementations
- **Testability**: Makes it easy to substitute mock objects during testing
- **Lifecycle Management**: Centralized control over component creation and destruction
- **Configuration**: Centralized configuration management for all components
- **Flexibility**: Easy to add, remove, or replace components without changing client code
- **Resource Management**: Automatic cleanup of resources when they're no longer needed

### Extending the Dependency Injection System

To add a new agent type to the system:

1. Create your agent class inheriting from `BaseAgent`
2. Add a provider for your agent in `AgentContainer`:

```python
new_agent = providers.Factory(
    NewAgentClass,
    config=providers.Selector(config, "agents.new_agent")
)
```

3. Include it in the `agent_providers` dictionary:

```python
agent_providers = providers.Dict({
    # Existing agents...
    "new_agent": new_agent
})
```

4. Update the configuration in `config.yaml` to include settings for your new agent

The dependency injection framework will handle the rest, ensuring your agent is created with the correct configuration and dependencies when needed by the workflow.

### Testing with Dependency Injection

Testing with dependency injection is straightforward and allows for better isolation of components. Here's how to create tests using the dependency injection system:

```python
import pytest
from unittest.mock import MagicMock
from orchestration.container import AgentContainer
from orchestration.workflow_manager import WorkflowManager

def test_workflow_with_dependency_injection():
    # Create a container instance
    container = AgentContainer()
    
    # Configure the container with test configuration
    container.config.from_dict({
        "system": {"name": "SOCIA", "version": "0.1.0"},
        "agents": {
            "task_understanding": {"prompt_template": "templates/task_understanding_prompt.txt"},
            "data_analysis": {"prompt_template": "templates/data_analysis_prompt.txt"},
            "model_planning": {"prompt_template": "templates/model_planning_prompt.txt"},
            # Other agent configurations...
        }
    })
    
    # Create mock agents
    mock_task_agent = MagicMock()
    mock_task_agent.process.return_value = {"title": "Test Simulation", "entities": [{"name": "Person"}]}
    
    mock_model_agent = MagicMock()
    mock_model_agent.process.return_value = {"model_type": "agent_based", "entities": [{"name": "Person"}]}
    
    # Override container providers with mocks
    container.task_understanding_agent = lambda: mock_task_agent
    container.model_planning_agent = lambda: mock_model_agent
    
    # Create a dictionary with all mock agents
    agent_providers_dict = {
        "task_understanding": mock_task_agent,
        "model_planning": mock_model_agent,
        # Add other mock agents as needed...
    }
    
    # Override the agent_providers with our mocks
    container.agent_providers = lambda: agent_providers_dict
    
    # Wire the container to the modules that use @inject
    container.wire(modules=["orchestration.workflow_manager", "main"])
    
    # Create a workflow manager with the test container
    workflow_manager = WorkflowManager(
        task_description="Create a test simulation",
        output_path="./test_output",
        agent_container=container
    )
    
    # Run the workflow
    result = workflow_manager.run()
    
    # Assert expected behavior
    assert mock_task_agent.process.called
    assert mock_model_agent.process.called
    # Additional assertions...
```

### Running the System with Dependency Injection

To run the complete system with dependency injection:

```bash
python main.py --task "Create a simulation for a city with 1000 people" --output "./my_sim_output"
```

This command automatically:
1. Creates the dependency injection container
2. Configures it with settings from `config.yaml`
3. Wires it to all modules that use the `@inject` decorator
4. Passes it to the `WorkflowManager`
5. Runs the complete workflow using the injected dependencies

The dependency injection system makes it possible to easily swap components, configure their behavior, and test them in isolation. 
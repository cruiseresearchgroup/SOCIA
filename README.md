<p align="center">
  <img src="docs/images/socia_logo_large.jpg" alt="SOCIA Logo" width="200px" />
</p>

# 🌆 SOCIA: Simulation Orchestration for City Intelligence and Agents

An LLM-driven multi-agent social simulation generator that automatically creates simulation environments based on user requirements and data.

<!-- Illustration Image Placeholder -->
<p align="center">
  <img src="docs/images/socia_architecture.png" alt="SOCIA System Architecture" width="40%" />
</p>
<!-- Replace this with your actual architecture diagram -->

## 🏗️ Architecture

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

## 🔧 Installation

```bash
pip install -r requirements.txt
```

### 🐳 Docker Installation (Required for Sandbox Isolation)

SOCIA uses Docker for isolated code execution and verification. You need to install Docker on your system to fully utilize the code verification and simulation execution features.

**Please be kindly noted: 
Always keep Docker running while using SOCIA for social simulation creation.**

#### Linux
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Install Docker CE
sudo apt-get update
sudo apt-get install docker-ce

# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Add your user to the docker group to run Docker without sudo
sudo usermod -aG docker $USER
```

#### macOS
The easiest way to install Docker on macOS is using Docker Desktop:

1. Download Docker Desktop for Mac from [Docker's official website](https://www.docker.com/products/docker-desktop)
2. Install the application by dragging it to your Applications folder
3. Launch Docker Desktop and follow the setup wizard
4. Verify installation with: `docker --version`

## ⚙️ Configuration

### 🔑 API Key

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

### 🤖 Language Model Selection

SOCIA supports multiple large language model providers. You can easily switch between them by editing the `config.yaml` file:

```yaml
# In config.yaml
llm:
  provider: "gemini"  # options: openai, gemini, anthropic, llama
```

Available LLM providers and their configurations:

- **OpenAI (GPT models)**
  ```yaml
  llm_providers:
    openai:
      model: "gpt-4o"  # options: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.
      temperature: 0.7
      max_tokens: 4000
  ```

- **Google Gemini**
  ```yaml
  llm_providers:
    gemini:
      model: "models/gemini-2.5-flash-preview-04-17"  # options: gemini-2.5-flash-preview, gemini-pro, etc.
      temperature: 0.7
      max_tokens: 8192
  ```

- **Anthropic Claude**
  ```yaml
  llm_providers:
    anthropic:
      model: "claude-3-opus-20240229"  # options: claude-3-opus, claude-3-sonnet, etc.
      temperature: 0.7
      max_tokens: 4000
  ```

To use a specific LLM provider:
1. Make sure you have the appropriate API key in `keys.py`
2. Update the `llm.provider` field in `config.yaml`
3. Adjust provider-specific parameters in the `llm_providers` section if needed

## 🚀 Usage

### 💻 Example Commands

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
python main.py --task "Create a simple epidemic simulation model that models the spread of a virus in a population of 1000 people." --output "./output/my_sim_output"
```

This command initiates the full SOCIA workflow:
1. LLM agents analyze the task description
2. The system designs and generates an epidemic simulation for a city of 1000 people
3. The simulation is executed and results are saved to the "./my_sim_output" directory
4. Visualizations and analysis are automatically generated

Use this command pattern when you want to create custom simulations based on your specific requirements. You can customize the task description to focus on different urban simulation scenarios.

### 🧪 Running Custom Simulations

When running a simulation with a custom task, the system:
1. Parses your requirements using natural language processing
2. Selects appropriate models based on the task context
3. Generates and validates the simulation code
4. Executes the simulation in a controlled environment
5. Produces results and visualizations

For more advanced usage, see the examples directory for sample scripts that demonstrate specific simulation types.

## 🏃 Sandbox Isolation Environment

SOCIA implements a secure sandbox isolation environment using Docker containers for safely executing and validating generated simulation code. This approach ensures:

1. **Security**: Generated code runs in an isolated container with limited access to host resources
2. **Dependency Management**: Automatic installation and verification of required packages
3. **Error Detection**: Comprehensive execution monitoring and error reporting
4. **Resource Control**: Memory and CPU limitations to prevent resource exhaustion

### 🛡️ Code Verification Agent Sandbox

The Code Verification Agent uses a sandbox environment to perform a comprehensive verification of generated code:

1. **Syntax Verification**: Checks for syntax errors before execution
2. **Dependency Analysis**: Extracts imports and identifies required packages
3. **Dependency Installation**: Attempts to install all required packages in the sandbox
4. **Smoke Test Execution**: Runs a minimal test execution to verify basic functionality

When the system encounters verification issues, it will:
- Provide detailed error messages
- Generate suggestions for fixing the issues
- Allow for iterative improvement of the code

The verification process is implemented in `agents/code_verification/sandbox.py` and follows this workflow:
```
1. Syntax check → 2. Dependency analysis → 3. Docker sandbox creation → 4. Code execution → 5. Results collection
```

### 🖥️ Simulation Execution Agent Sandbox

The Simulation Execution Agent uses a similar Docker-based sandbox for safely running the full simulation:

1. **Environment Setup**: Creates a clean Docker container with Python
2. **Package Installation**: Installs common simulation packages (numpy, matplotlib, etc.)
3. **Metric Collection**: Injects code to collect performance and simulation metrics
4. **Execution Monitoring**: Tracks execution time, memory usage, and errors
5. **Results Extraction**: Captures simulation output and metrics for evaluation

If Docker is not available on your system, SOCIA will fall back to using the LLM to simulate the execution, which provides a less accurate but still useful approximation of the simulation results.

The execution sandbox is implemented in `agents/simulation_execution/agent.py` and follows a similar pattern to the verification sandbox, with additional focus on capturing simulation metrics and performance data.

Both sandbox implementations use the base `DockerSandbox` class that provides a secure, isolated execution environment with:

- Network access controls
- Memory limitations
- Execution timeouts
- File system isolation
- Standard output/error capturing 

## 📁 Project Structure

- `agents/`: Individual agent implementations
- `core/`: Core simulation framework
- `data/`: Data management utilities
- `models/`: Simulation model templates
- `orchestration/`: Agent coordination
- `utils/`: Utility functions
- `tests/`: Test suite 

## 📝 Log

SOCIA includes a comprehensive logging system that records all aspects of the simulation generation and execution process:

- 🔍 **Detailed Execution Logs**: The system automatically records INFO, WARNING, and ERROR level messages
- 📂 **Output Directory Integration**: Logs are saved directly in your output project directory
- 🔄 **LLM Model Tracking**: Every LLM call is logged with the specific model version used
- 🕒 **Timestamped Entries**: All log entries include precise timestamps for debugging and analysis

To view logs for a simulation run:
```bash
# Logs are automatically saved in your output directory
cat ./your_output_directory/socia.log

# Example for checking LLM model usage
grep "Using Gemini model" ./your_output_directory/socia.log
```

This logging system is invaluable for:
- Debugging simulation generation issues
- Tracking LLM model performance and usage
- Analyzing the workflow execution process
- Understanding agent decision-making

Log settings can be configured in the `config.yaml` file:
```yaml
logging:
  level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

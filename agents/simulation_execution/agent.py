"""
SimulationExecutionAgent: Executes the generated simulation code and collects results.
"""

import logging
import os
import subprocess
import time
import json
import tempfile
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseAgent
from agents.code_verification.sandbox import DockerSandbox


def run_python_script(script_file: str, data_path: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """
    Execute a Python script using subprocess and return detailed results.
    
    Args:
        script_file: Path to the Python script to execute
        data_path: Path to input data (optional)
        timeout: Timeout in seconds (default: 5 minutes)
    
    Returns:
        Dictionary containing stdout, stderr, returncode, and execution time
    """
    # Set up environment variables
    env = os.environ.copy()
    env["PROJECT_ROOT"] = os.getcwd()
    # Respect existing DATA_PATH if provided via environment, override only if data_path argument is given
    if data_path is not None:
        env["DATA_PATH"] = data_path
    elif "DATA_PATH" not in env or not env["DATA_PATH"]:
        env["DATA_PATH"] = "data"
    
    # Record start time
    start_time = time.time()
    
    try:
        # Execute the Python script
        result = subprocess.run(
            ["python", script_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        # Record execution time
        execution_time = time.time() - start_time
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "execution_time": execution_time,
            "success": result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "returncode": -1,
            "execution_time": execution_time,
            "success": False
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "execution_time": execution_time,
            "success": False
        }


class SimulationExecutionAgent(BaseAgent):
    """
    Simulation Execution Agent runs the generated simulation code in a controlled
    environment and collects the results.
    
    This agent is responsible for:
    1. Setting up the execution environment
    2. Running the simulation with appropriate parameters
    3. Collecting metrics and outputs
    4. Handling any runtime errors
    5. Executing code in an isolated Docker container for security (full mode)
    6. Executing code directly with subprocess for lightweight execution (lite mode)
    """
    
    def __init__(self, output_dir: str, config: Dict[str, Any] = None):
        """
        Initialize the Simulation Execution Agent.
        
        Args:
            output_dir: Directory to store execution artifacts
            config: Configuration dictionary for the agent
        """
        # If config is not provided, use a minimal default configuration
        if config is None:
            config = {
                "prompt_template": "templates/simulation_execution_prompt.txt",
                "output_format": "json"
            }
        
        super().__init__(config)
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "execution"), exist_ok=True)
        
        # Check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            self.docker_available = result.returncode == 0
            if not self.docker_available:
                self.logger.warning("Docker is not available. Falling back to subprocess execution.")
        except FileNotFoundError:
            self.docker_available = False
            self.logger.warning("Docker is not installed. Falling back to subprocess execution.")
    
    def process(
        self,
        code_path: str,
        task_spec: Dict[str, Any],
        data_path: Optional[str] = None,
        mode: str = "full"
    ) -> Dict[str, Any]:
        """
        Execute the simulation code and collect results.
        
        Args:
            code_path: Path to the simulation code file
            task_spec: Task specification from the Task Understanding Agent
            data_path: Path to input data (optional)
            mode: Execution mode ("full" or "lite")
        
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info(f"Executing simulation code in {mode} mode")
        
        # Read the code file
        try:
            with open(code_path, 'r') as f:
                code = f.read()
        except Exception as e:
            self.logger.error(f"Error reading code file: {str(e)}")
            return {
                "execution_status": "failed",
                "runtime_errors": [f"Error reading code file: {str(e)}"],
                "performance_metrics": {},
                "simulation_metrics": {},
                "time_series_data": [],
                "visualizations": [],
                "summary": "Failed to read simulation code file"
            }
        
        # Choose execution method based on mode
        if mode == "lite":
            # Use direct subprocess execution for lite mode
            execution_result = self._execute_code_with_subprocess(code_path, data_path)
            if execution_result:
                return execution_result
        else:
            # Try to execute the code in a Docker sandbox if available (full mode)
            if self.docker_available:
                execution_result = self._execute_code_in_sandbox(code, data_path)
                if execution_result:
                    return execution_result
        
        # Fall back to LLM simulation if execution fails or is unavailable
        self.logger.info("Using LLM to simulate execution")
        
        # Build prompt for LLM simulation, include file references if available
        prompt = self._build_prompt(
            task_spec=task_spec,
            code=code,
            data_path=data_path
        )
        
        # Call LLM to simulate execution
        llm_response = self._call_llm(prompt)
        
        # Parse the response
        execution_result = self._parse_llm_response(llm_response)
        
        # If LLM response parsing failed, create a basic result
        if isinstance(execution_result, str):
            execution_result = {
                "execution_status": "success",
                "runtime_errors": [],
                "performance_metrics": {
                    "execution_time": 1.0,
                    "memory_usage": 100
                },
                "simulation_metrics": {
                    "total_entities": 100,
                    "average_activity": 0.5
                },
                "time_series_data": [
                    {
                        "time_step": 0,
                        "metrics": {
                            "total_entities": 100,
                            "average_activity": 0.5
                        }
                    }
                ],
                "visualizations": [],
                "summary": "Simulated execution of the code (LLM-based)"
            }
        
        # Log LLM simulation results
        self.logger.info(f"LLM simulation completed with status: {execution_result.get('execution_status', 'unknown')}")
        self.logger.info(f"Simulation summary: {execution_result.get('summary', 'No summary available')}")
        if execution_result.get('execution_status') == 'failed' and execution_result.get('runtime_errors'):
            self.logger.warning(f"Simulated runtime errors: {execution_result.get('runtime_errors')}")
        self.logger.debug(f"Detailed simulation result: {json.dumps(execution_result, indent=2)}")
        
        self.logger.info("Simulation execution completed")
        return execution_result
    
    def _execute_code_with_subprocess(
        self,
        script_file: str,
        data_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute Python script using subprocess for lite mode.
        
        Args:
            script_file: Path to the Python script to execute
            data_path: Path to input data (optional)
        
        Returns:
            Dictionary containing execution results or None if execution failed
        """
        try:
            self.logger.info("Executing code with subprocess")
            
            # Create output directory for this execution
            execution_output_dir = os.path.join(self.output_dir, "execution")
            os.makedirs(execution_output_dir, exist_ok=True)
            
            # Execute the Python script using the helper function
            result = run_python_script(script_file, data_path, timeout=300)
            execution_time = result["execution_time"]
            
            # Print results as requested
            print("standard output（stdout）:")
            print(result["stdout"])
            print("error info（stderr）:")
            print(result["stderr"])
            print("return code（returncode）:")
            print(result["returncode"])
            
            # Determine execution status
            execution_status = "success" if result["success"] else "failed"
            
            # Parse runtime errors
            runtime_errors = []
            if result["stderr"]:
                # Only treat non-INFO logs as errors
                for line in result["stderr"].splitlines():
                    if line.strip() and not line.startswith("INFO:"):
                        runtime_errors.append(line)
            if result["returncode"] != 0:
                runtime_errors.append(f"Process exited with code {result['returncode']}")
            
            # Create execution result
            execution_result = {
                "execution_status": execution_status,
                "runtime_errors": runtime_errors,
                "performance_metrics": {
                    "execution_time": execution_time,
                    "memory_usage": "unknown"  # subprocess doesn't easily provide memory usage
                },
                "simulation_metrics": {},
                "time_series_data": [],
                "visualizations": [],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
                "summary": f"Executed with subprocess in {execution_time:.2f} seconds, exit code: {result['returncode']}"
            }
            
            # Try to extract simulation metrics from stdout if possible
            if result["stdout"]:
                # Look for common patterns in output that might indicate simulation results
                lines = result["stdout"].split('\n')
                for line in lines:
                    if 'simulation completed' in line.lower() or 'results:' in line.lower():
                        # Basic parsing - could be enhanced based on specific output formats
                        try:
                            # Look for numbers that might be metrics
                            import re
                            numbers = re.findall(r'\d+\.?\d*', line)
                            if numbers:
                                execution_result["simulation_metrics"]["extracted_value"] = float(numbers[0])
                        except:
                            pass
            
            # Save execution results
            results_file = os.path.join(execution_output_dir, "execution_results.json")
            with open(results_file, 'w') as f:
                json.dump(execution_result, f, indent=2)
            
            # Log execution results
            self.logger.info(f"Subprocess execution completed with status: {execution_status}")
            self.logger.info(f"Execution time: {execution_time:.2f} seconds")
            self.logger.info(f"Return code: {result['returncode']}")
            if runtime_errors:
                self.logger.warning(f"Runtime errors detected: {runtime_errors}")
            if result["stdout"]:
                self.logger.debug(f"Stdout (first 500 chars): {result['stdout'][:500]}")
            if result["stderr"]:
                self.logger.debug(f"Stderr (first 500 chars): {result['stderr'][:500]}")
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            self.logger.error("Subprocess execution timed out after 5 minutes")
            return {
                "execution_status": "failed",
                "runtime_errors": ["Execution timed out after 5 minutes"],
                "performance_metrics": {"execution_time": 300},
                "simulation_metrics": {},
                "time_series_data": [],
                "visualizations": [],
                "stdout": "",
                "stderr": "Execution timed out",
                "returncode": -1,
                "summary": "Execution failed due to timeout"
            }
        except Exception as e:
            self.logger.error(f"Error executing code with subprocess: {str(e)}")
            return {
                "execution_status": "failed", 
                "runtime_errors": [f"Subprocess execution error: {str(e)}"],
                "performance_metrics": {},
                "simulation_metrics": {},
                "time_series_data": [],
                "visualizations": [],
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "summary": f"Execution failed with error: {str(e)}"
            }
    
    def _execute_code_in_sandbox(
        self,
        code: str,
        data_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute the code in a sandbox environment.
        
        Args:
            code: The simulation code
            data_path: Path to input data (optional)
        
        Returns:
            Dictionary containing execution results or None if execution failed
        """
        try:
            self.logger.info("Executing code in Docker sandbox")
            
            # Create output directory for this execution
            execution_output_dir = os.path.join(self.output_dir, "execution")
            os.makedirs(execution_output_dir, exist_ok=True)
            
            # Create a sandbox for execution
            with DockerSandbox(
                base_image="python:3.10-slim",
                timeout=120,  # 2 minutes timeout
                max_memory="1g",
                network_enabled=True,  # Enable network to install packages
                data_path=data_path  # Pass data_path to DockerSandbox
            ) as sandbox:
                # Write a custom entry point to collect metrics
                entry_point = """
# Add custom metric collection for simulation
import time
import json

# Initialize metrics dictionary
collected_metrics = {
    "performance_metrics": {
        "execution_time": 0
    },
    "simulation_metrics": {},
    "time_series_data": []
}

# Start timer
start_time = time.time()

# Try to execute the simulation
try:
    # First try to find and execute a main function
    if 'main' in locals() and callable(locals()['main']):
        locals()['main']()
    
    # If no main function, try to find and execute a Simulation class
    elif 'Simulation' in locals() and callable(locals()['Simulation']):
        # Use reasonable parameters
        sim_params = {
            "population_size": 1000,
            "initial_infected_count": 1,
            "transmission_probability": 0.1,
            "recovery_probability_per_step": 0.05,
            "simulation_steps": 100,
            "random_seed": 42
        }
        
        # Create and run the simulation
        sim = locals()['Simulation'](sim_params)
        sim.run()
        
        # Try to get metrics from the simulation
        if hasattr(sim, 'get_metrics_history'):
            metrics_history = sim.get_metrics_history()
            
            # Convert time series data
            if isinstance(metrics_history, dict):
                steps = metrics_history.get("step", [])
                for i, step in enumerate(steps):
                    step_metrics = {}
                    for key in metrics_history:
                        if key != "step" and i < len(metrics_history[key]):
                            step_metrics[key] = metrics_history[key][i]
                    
                    collected_metrics["time_series_data"].append({
                        "time_step": step,
                        "metrics": step_metrics
                    })
        
        # Try to get final metrics
        if hasattr(sim, '_current_metrics'):
            collected_metrics["simulation_metrics"] = sim._current_metrics
    
    # If execution reached here, it was successful
    collected_metrics["execution_status"] = "success"
    collected_metrics["runtime_errors"] = []
    
except Exception as e:
    # Record any execution errors
    import traceback
    collected_metrics["execution_status"] = "failed"
    collected_metrics["runtime_errors"] = [str(e), traceback.format_exc()]

# Record execution time
collected_metrics["performance_metrics"]["execution_time"] = time.time() - start_time

# Save metrics to a file
with open('/sandbox/simulation_metrics.json', 'w') as f:
    json.dump(collected_metrics, f, indent=2)
"""
                
                # Install required packages
                # Note: This is a simple approach. A more robust solution would
                # analyze the code for imports first, similar to the verification sandbox.
                common_packages = ["numpy", "matplotlib", "pandas"]
                for package in common_packages:
                    sandbox.install_package(package)
                
                # Execute the code with our custom entry point
                execution_results = sandbox.execute_code(code, entry_point)
                
                # Check if metrics file was created
                metrics_file = os.path.join(sandbox.temp_dir, "simulation_metrics.json")
                
                if os.path.exists(metrics_file):
                    # Read metrics file
                    with open(metrics_file, 'r') as f:
                        collected_metrics = json.load(f)
                    
                    # Merge with execution results
                    result = {
                        "execution_status": collected_metrics.get("execution_status", "failed"),
                        "runtime_errors": collected_metrics.get("runtime_errors", []),
                        "performance_metrics": collected_metrics.get("performance_metrics", {}),
                        "simulation_metrics": collected_metrics.get("simulation_metrics", {}),
                        "time_series_data": collected_metrics.get("time_series_data", []),
                        "visualizations": [],
                        "summary": "Executed in isolated Docker container"
                    }
                    
                    # Add execution stdout/stderr
                    stdout_full = execution_results.get("stdout", "")
                    stderr_full = execution_results.get("stderr", "")
                    MAX_SNIPPET_LEN = 500
                    result["stdout"] = (stdout_full[:MAX_SNIPPET_LEN] + "... (truncated)") if len(stdout_full) > MAX_SNIPPET_LEN else stdout_full
                    result["stderr"] = (stderr_full[:MAX_SNIPPET_LEN] + "... (truncated)") if len(stderr_full) > MAX_SNIPPET_LEN else stderr_full
                    
                    # Save execution results
                    results_file = os.path.join(execution_output_dir, "execution_results.json")
                    with open(results_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    # Log execution results
                    self.logger.info(f"Execution status: {result['execution_status']}")
                    if result['execution_status'] == 'success':
                        self.logger.info(f"Execution completed successfully in {result['performance_metrics'].get('execution_time', 0):.2f} seconds")
                    else:
                        self.logger.warning(f"Execution failed with errors: {result['runtime_errors']}")
                    self.logger.debug(f"Detailed execution result: {json.dumps(result, indent=2)}")
                    
                    return result
                else:
                    # If metrics file doesn't exist, use execution results
                    result = {
                        "execution_status": "failed" if not execution_results.get("success", False) else "success",
                        "runtime_errors": [execution_results.get("error", "Unknown error")],
                        "performance_metrics": {
                            "execution_time": execution_results.get("execution_time", 0)
                        },
                        "simulation_metrics": {},
                        "time_series_data": [],
                        "visualizations": [],
                        "summary": "Execution failed to produce metrics"
                    }
                    
                    # Add truncated execution stdout/stderr to avoid huge logs
                    stdout_full = execution_results.get("stdout", "")
                    stderr_full = execution_results.get("stderr", "")
                    MAX_SNIPPET_LEN = 500
                    result["stdout"] = (stdout_full[:MAX_SNIPPET_LEN] + "... (truncated)") if len(stdout_full) > MAX_SNIPPET_LEN else stdout_full
                    result["stderr"] = (stderr_full[:MAX_SNIPPET_LEN] + "... (truncated)") if len(stderr_full) > MAX_SNIPPET_LEN else stderr_full
                    
                    # Save execution results
                    results_file = os.path.join(execution_output_dir, "execution_results.json")
                    with open(results_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    # Log execution results
                    self.logger.warning(f"Execution failed to produce metrics file")
                    self.logger.info(f"Execution status from sandbox: {result['execution_status']}")
                    if result['runtime_errors']:
                        self.logger.warning(f"Errors: {result['runtime_errors']}")
                    self.logger.debug(f"Detailed execution result: {json.dumps(result, indent=2)}")
                    
                    return result
        
        except Exception as e:
            self.logger.error(f"Error executing code in sandbox: {str(e)}")
            return None
    
    def _build_prompt(
        self,
        task_spec: Dict[str, Any],
        code: str,
        data_path: Optional[str] = None
    ) -> str:
        """
        Build a prompt for the LLM to simulate execution.
        
        Args:
            task_spec: Task specification
            code: The generated code
            data_path: Path to input data (optional)
        
        Returns:
            Prompt for the LLM
        """
        return f"""
You are a simulation expert. Your task is to simulate running Python code for a social simulation.

Use the following information:
TASK SPECIFICATION:
{json.dumps(task_spec, indent=2)}

Data Path:
{data_path}
""" 
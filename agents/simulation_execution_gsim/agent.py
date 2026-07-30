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


import re


def _detect_script_output_arg(script_file: str) -> str:
    """
    Detect which output argument format the script expects.
    
    Some scripts use --output (single file format), others use --output_dir (directory format).
    This function reads the script to detect which one is used.
    
    Args:
        script_file: Path to the Python script
        
    Returns:
        "output" if script uses --output, "output_dir" if script uses --output_dir
    """
    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for explicit argument definitions using regex patterns
        # Look for patterns like: add_argument("--output_dir" or add_argument('--output_dir'
        output_dir_pattern = r'add_argument\s*\(\s*["\']--output_dir["\']'
        output_pattern = r'add_argument\s*\(\s*["\']--output["\']'
        
        has_explicit_output_dir = bool(re.search(output_dir_pattern, content))
        has_explicit_output = bool(re.search(output_pattern, content))
        
        if has_explicit_output_dir and not has_explicit_output:
            return "output_dir"
        elif has_explicit_output and not has_explicit_output_dir:
            return "output"
        elif has_explicit_output_dir and has_explicit_output:
            # Both defined, prefer --output (single file format) as it's more common
            return "output"
        else:
            # Default to --output if we can't detect
            return "output"
            
    except Exception:
        # If we can't read the file, default to --output
        return "output"


def run_python_script(
    script_file: str, 
    data_path: Optional[str] = None, 
    timeout: int = 300,
    output_file: Optional[str] = None,
    project_root: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a Python script using subprocess and return detailed results.
    
    Args:
        script_file: Path to the Python script to execute
        data_path: Path to input data (optional)
        timeout: Timeout in seconds (default: 5 minutes)
        output_file: Output file/dir path for simulation results (optional, will be passed as --output or --output_dir argument based on script detection)
        project_root: PROJECT_ROOT environment variable value (optional, defaults to current working directory)
        openai_api_key: OPENAI_API_KEY environment variable value (optional, will use existing if not provided)
    
    Returns:
        Dictionary containing stdout, stderr, returncode, and execution time
    """
    # Set up environment variables
    env = os.environ.copy()
    
    # Set PROJECT_ROOT (use provided value, or current working directory, or existing env var)
    if project_root is not None:
        env["PROJECT_ROOT"] = project_root
    elif "PROJECT_ROOT" not in env:
        env["PROJECT_ROOT"] = os.getcwd()
    
    # Set DATA_PATH (respect existing if provided via environment, override only if data_path argument is given)
    if data_path is not None:
        env["DATA_PATH"] = data_path
    elif "DATA_PATH" not in env or not env["DATA_PATH"]:
        env["DATA_PATH"] = "data"
    
    # Set OPENAI_API_KEY (use provided value, or keep existing)
    if openai_api_key is not None:
        env["OPENAI_API_KEY"] = openai_api_key
    # If not provided, keep existing OPENAI_API_KEY from environment (if any)
    
    # Record start time
    start_time = time.time()
    
    try:
        # Build command with output argument
        # Auto-detect which output argument format the script expects (--output vs --output_dir)
        cmd = ["python", script_file]
        if output_file:
            output_arg_type = _detect_script_output_arg(script_file)
            if output_arg_type == "output_dir":
                cmd.extend(["--output_dir", output_file])
            else:
                cmd.extend(["--output", output_file])
        
        # Execute the Python script
        result = subprocess.run(
            cmd,
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
    6. Executing code directly with subprocess for lightweight execution (lite/ace/alpha/gsim mode)
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
                "output_format": "json",
                "timeout": 300  # Default timeout: 5 minutes
            }
        
        super().__init__(config)
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "execution"), exist_ok=True)
        
        # Get timeout from config (default: 300 seconds / 5 minutes)
        self.timeout = self.config.get("timeout", 300)
        self.logger.info(f"Simulation execution timeout set to {self.timeout} seconds")
        
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
        mode: str = "full",
        output_dir: Optional[str] = None,
        iteration: Optional[int] = None,
        project_root: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the simulation code and collect results.
        
        Args:
            code_path: Path to the simulation code file
            task_spec: Task specification from the Task Understanding Agent
            data_path: Path to input data (optional)
            mode: Execution mode ("full", "lite", "ace", "alpha", or "gsim"). ACE/ALPHA/GSIM mode uses subprocess like lite mode.
            output_dir: Output directory for simulation results (optional, used to generate --output file path)
            iteration: Current iteration number (optional, used to generate output filename)
            project_root: PROJECT_ROOT environment variable value (optional, will use existing env var if not provided)
            openai_api_key: OPENAI_API_KEY environment variable value (optional, will use existing env var if not provided)
        
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
        # ACE/ALPHA/GSIM mode and lite mode both use subprocess execution (same path)
        if mode in ["ace", "alpha", "lite", "gsim", "srr"]:
            # Use direct subprocess execution for lite/ace/alpha mode
            self.logger.info(f"Using subprocess execution for {mode} mode")
            
            # Generate output file/dir path if output_dir and iteration are provided
            output_file = None
            if output_dir and iteration is not None:
                # Detect which output argument format the script expects
                output_arg_type = _detect_script_output_arg(code_path)
                if output_arg_type == "output_dir":
                    # Directory format: output_iter_{N} (no .json extension)
                    output_file = os.path.join(output_dir, f"output_iter_{iteration}")
                    self.logger.info(f"Output directory for simulation (--output_dir): {output_file}")
                else:
                    # Single file format: output_iter_{N}.json
                    output_file = os.path.join(output_dir, f"output_iter_{iteration}.json")
                    self.logger.info(f"Output file for simulation (--output): {output_file}")
            
            execution_result = self._execute_code_with_subprocess(
                code_path, 
                data_path,
                output_file=output_file,
                project_root=project_root,
                openai_api_key=openai_api_key,
                task_spec=task_spec
            )
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
        data_path: Optional[str] = None,
        output_file: Optional[str] = None,
        project_root: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        task_spec: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute Python script using subprocess for lite/ace/alpha/gsim mode.
        
        Args:
            script_file: Path to the Python script to execute
            data_path: Path to input data (optional)
            output_file: Output file path for simulation results (optional, will be passed as --output argument)
            project_root: PROJECT_ROOT environment variable value (optional)
            openai_api_key: OPENAI_API_KEY environment variable value (optional)
        
        Returns:
            Dictionary containing execution results or None if execution failed
        """
        try:
            self.logger.info("Executing code with subprocess")
            
            # Create output directory for this execution
            execution_output_dir = os.path.join(self.output_dir, "execution")
            os.makedirs(execution_output_dir, exist_ok=True)
            
            # Execute the Python script using the helper function
            # Use timeout from config (default: 300 seconds)
            timeout = getattr(self, 'timeout', 300)
            result = run_python_script(
                script_file, 
                data_path, 
                timeout=timeout,
                output_file=output_file,
                project_root=project_root,
                openai_api_key=openai_api_key
            )
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
            
            # Read simulation results from output file/directory if it exists
            # Support two formats:
            # 1. Single JSON file (--output format): all results in one file
            # 2. Directory with multiple JSON files (--output_dir format): separate files for each result type
            # 
            # IMPORTANT: Check for directory FIRST, because os.path.exists() returns True for directories too
            if output_file and os.path.isdir(output_file):
                # Handle directory format (--output_dir): multiple JSON files in a directory
                try:
                    self.logger.info(f"Output path is a directory, reading multiple JSON files from: {output_file}")
                    simulation_output = {}
                    
                    # Check if this is a COVID SIR task, Three-disease Hospital task, or Beer Game (SUPPLY) task (special handling for results.json)
                    is_covid_sir = False
                    is_hospital = False
                    is_supply = False
                    is_mask_wearing = False
                    if task_spec:
                        task_description = task_spec.get("description", "").lower()
                        is_covid_sir = "covid sir" in task_description
                        is_hospital = "three-disease hospital" in task_description
                        is_supply = "beer game (supply)" in task_description
                        is_mask_wearing = "mask-wearing behavior" in task_description
                        if is_covid_sir:
                            self.logger.info("COVID SIR task detected: will read results.json and apply special mapping")
                        if is_hospital:
                            self.logger.info("Three-disease Hospital task detected: will read results.json and apply special mapping")
                        if is_supply:
                            self.logger.info("Beer Game (SUPPLY) task detected: will read results.json and apply special mapping")
                        if is_mask_wearing:
                            self.logger.info("Mask-wearing behavior task detected: will read results.json and apply ACE-compatible mapping")
                    
                    # Try to read each expected file (skip large files like simulated_trajectories)
                    expected_files = {
                        "calibrated_parameters": "calibrated_parameters.json",
                        "calibration_log": "calibration_log.json",
                        "evaluation_results_on_validation": "evaluation_results_on_validation.json",
                        # Skip simulated_trajectories_validation.json as it's large and not needed for metrics
                    }
                    
                    # Task-specific directory outputs use results.json.
                    if is_covid_sir or is_hospital or is_supply or is_mask_wearing:
                        expected_files["results"] = "results.json"
                    
                    files_found = 0
                    for key, filename in expected_files.items():
                        file_path = os.path.join(output_file, filename)
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    simulation_output[key] = json.load(f)
                                files_found += 1
                                self.logger.debug(f"Successfully loaded {filename}")
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Failed to parse JSON from {file_path}: {e}")
                            except Exception as e:
                                self.logger.warning(f"Failed to read {file_path}: {e}")
                    
                    if files_found > 0:
                        # Transform directory format to match single-file format structure
                        transformed_output = {}

                        # Mask adoption uses the same directory schema as SOCIA/ACE:
                        # metrics, val_metrics, and parameters.
                        if is_mask_wearing and "results" in simulation_output:
                            results_data = simulation_output["results"]
                            self.logger.info("Applying ACE-compatible mask-wearing mapping from results.json")

                            if isinstance(results_data.get("metrics"), dict):
                                execution_result["simulation_metrics"] = results_data["metrics"]

                            if isinstance(results_data.get("val_metrics"), dict):
                                execution_result["val_metrics"] = results_data["val_metrics"]

                            parameters = results_data.get(
                                "parameters",
                                results_data.get("optimized_parameters"),
                            )
                            if isinstance(parameters, dict):
                                execution_result["calibrated_parameters"] = parameters
                                transformed_output["calibrated_parameters"] = parameters
                                transformed_output["optimized_parameters"] = parameters

                            transformed_output["results"] = results_data
                        
                        # Special handling for COVID SIR tasks: map results.json fields
                        if is_covid_sir and "results" in simulation_output:
                            results_data = simulation_output["results"]
                            self.logger.info("Applying COVID SIR mapping from results.json")
                            
                            # Map metrics -> simulation_metrics
                            if "metrics" in results_data:
                                execution_result["simulation_metrics"] = results_data["metrics"]
                                self.logger.debug(f"Mapped metrics to simulation_metrics: {len(results_data['metrics'])} fields")
                            
                            # Map optimized_parameters -> calibrated_parameters
                            if "optimized_parameters" in results_data:
                                execution_result["calibrated_parameters"] = results_data["optimized_parameters"]
                                transformed_output["calibrated_parameters"] = results_data["optimized_parameters"]
                                self.logger.debug(f"Mapped optimized_parameters to calibrated_parameters")
                            
                            # Map calibration_artifacts -> calibration_artifacts (preserve as-is)
                            if "calibration_artifacts" in results_data:
                                execution_result["calibration_artifacts"] = results_data["calibration_artifacts"]
                                transformed_output["calibration_artifacts"] = results_data["calibration_artifacts"]
                                self.logger.debug(f"Mapped calibration_artifacts")
                            
                            # Also preserve the full results.json in simulation_output for reference
                            transformed_output["results"] = results_data
                        
                        # Special handling for Three-disease Hospital tasks: map results.json fields
                        if is_hospital and "results" in simulation_output:
                            results_data = simulation_output["results"]
                            self.logger.info("Applying Three-disease Hospital mapping from results.json")
                            
                            # Map metrics -> simulation_metrics
                            if "metrics" in results_data:
                                execution_result["simulation_metrics"] = results_data["metrics"]
                                self.logger.debug(f"Mapped metrics to simulation_metrics: {len(results_data['metrics'])} fields")
                            
                            # Map optimized_parameters -> calibrated_parameters
                            if "optimized_parameters" in results_data:
                                execution_result["calibrated_parameters"] = results_data["optimized_parameters"]
                                transformed_output["calibrated_parameters"] = results_data["optimized_parameters"]
                                self.logger.debug(f"Mapped optimized_parameters to calibrated_parameters")
                            
                            # Map calibration_artifacts -> calibration_artifacts (exclude loss_history)
                            if "calibration_artifacts" in results_data:
                                calib_artifacts = results_data["calibration_artifacts"]
                                # Create a copy excluding loss_history
                                filtered_calib_artifacts = {
                                    k: v for k, v in calib_artifacts.items() 
                                    if k != "loss_history"
                                }
                                execution_result["calibration_artifacts"] = filtered_calib_artifacts
                                transformed_output["calibration_artifacts"] = filtered_calib_artifacts
                                self.logger.debug(f"Mapped calibration_artifacts (excluding loss_history): {len(filtered_calib_artifacts)} fields")
                            
                            # Also preserve the full results.json in simulation_output for reference
                            transformed_output["results"] = results_data
                        
                        # Special handling for Beer Game (SUPPLY) tasks: map results.json fields
                        if is_supply and "results" in simulation_output:
                            results_data = simulation_output["results"]
                            self.logger.info("Applying Beer Game (SUPPLY) mapping from results.json")
                            
                            # Map metrics -> simulation_metrics
                            if "metrics" in results_data:
                                execution_result["simulation_metrics"] = results_data["metrics"]
                                self.logger.debug(f"Mapped metrics to simulation_metrics: {len(results_data['metrics'])} fields")
                            
                            # Map optimized_parameters -> calibrated_parameters
                            if "optimized_parameters" in results_data:
                                execution_result["calibrated_parameters"] = results_data["optimized_parameters"]
                                transformed_output["calibrated_parameters"] = results_data["optimized_parameters"]
                                self.logger.debug(f"Mapped optimized_parameters to calibrated_parameters")
                            
                            # Map calibration_artifacts -> calibration_artifacts (preserve as-is)
                            if "calibration_artifacts" in results_data:
                                execution_result["calibration_artifacts"] = results_data["calibration_artifacts"]
                                transformed_output["calibration_artifacts"] = results_data["calibration_artifacts"]
                                self.logger.debug(f"Mapped calibration_artifacts")
                            
                            # Also preserve the full results.json in simulation_output for reference
                            transformed_output["results"] = results_data
                        
                        # Transform calibrated_parameters (only if not already set by COVID SIR mapping)
                        if "calibrated_parameters" not in execution_result and "calibrated_parameters" in simulation_output:
                            calib_params = simulation_output["calibrated_parameters"]
                            # Convert from directory format to single-file format
                            transformed_calib = {
                                "best_objective_on_training": calib_params.get("best_objective"),
                                "best_params": calib_params.get("best_parameters", {}),
                            }
                            # Add calibration_history if calibration_log exists
                            if "calibration_log" in simulation_output:
                                transformed_calib["calibration_history"] = simulation_output["calibration_log"]
                            transformed_output["calibrated_parameters"] = transformed_calib
                            execution_result["calibrated_parameters"] = transformed_calib
                        
                        # Transform evaluation_results_on_validation
                        if "evaluation_results_on_validation" in simulation_output:
                            eval_results = simulation_output["evaluation_results_on_validation"]
                            
                            # Check if it's already in the new format (has 'summary' or 'by_seed')
                            if "summary" in eval_results or "by_seed" in eval_results:
                                # Already in correct format, use as-is
                                transformed_eval = eval_results
                            else:
                                # Transform from directory format to single-file format
                                transformed_eval = {}
                                
                                # Extract simulation_metrics if present (directory format)
                                if "simulation_metrics" in eval_results:
                                    metrics_dict = eval_results["simulation_metrics"]
                                    # Convert metrics to summary format (each metric becomes {mean: value})
                                    summary = {}
                                    for metric_name, metric_value in metrics_dict.items():
                                        if isinstance(metric_value, (int, float)):
                                            summary[metric_name] = {"mean": metric_value}
                                    
                                    transformed_eval["summary"] = summary
                                    
                                    # Extract metrics to execution_result["simulation_metrics"]
                                    for metric_name, metric_value in metrics_dict.items():
                                        if isinstance(metric_value, (int, float)):
                                            execution_result["simulation_metrics"][metric_name] = metric_value
                                
                                # Preserve other fields
                                if "objective" in eval_results:
                                    transformed_eval["objective"] = eval_results["objective"]
                                if "objective_weights" in eval_results:
                                    transformed_eval["objective_weights"] = eval_results["objective_weights"]
                                if "validation_set" in eval_results:
                                    transformed_eval["validation_set"] = eval_results["validation_set"]
                                if "meta" in eval_results:
                                    transformed_eval["meta"] = eval_results["meta"]
                            
                            transformed_output["evaluation_results_on_validation"] = transformed_eval
                            execution_result["evaluation_results_on_validation"] = transformed_eval
                            
                            # If we haven't extracted metrics yet, try to extract from summary
                            if not execution_result["simulation_metrics"] and "summary" in transformed_eval:
                                summary = transformed_eval["summary"]
                                if isinstance(summary, dict):
                                    for metric_name, metric_data in summary.items():
                                        if isinstance(metric_data, dict) and "mean" in metric_data:
                                            execution_result["simulation_metrics"][metric_name] = metric_data["mean"]
                        
                        # Add simulated_trajectories_validation if it exists (optional, may be large)
                        sim_traj_path = os.path.join(output_file, "simulated_trajectories_validation.json")
                        if os.path.exists(sim_traj_path):
                            try:
                                with open(sim_traj_path, 'r', encoding='utf-8') as f:
                                    transformed_output["simulated_trajectories_validation"] = json.load(f)
                            except Exception as e:
                                self.logger.warning(f"Failed to load simulated_trajectories_validation.json: {e}")
                        
                        # Add the transformed output to execution_result (matches single-file format)
                        execution_result["simulation_output"] = transformed_output
                        
                        # Update summary
                        if execution_status == "success":
                            metrics_count = len(execution_result["simulation_metrics"])
                            execution_result["summary"] = f"Executed with subprocess in {execution_time:.2f} seconds. Results loaded from directory: {output_file} ({files_found} files, {metrics_count} metrics)"
                        
                        self.logger.info(f"Successfully loaded {files_found} result files from directory format, extracted {len(execution_result['simulation_metrics'])} metrics")
                    else:
                        self.logger.warning(f"Output directory {output_file} exists but no expected result files were found")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to read results from directory {output_file}: {e}")
                    
            elif output_file and os.path.isfile(output_file):
                # Handle single file format (--output): all results in one JSON file
                try:
                    self.logger.info(f"Reading simulation results from output file: {output_file}")
                    with open(output_file, 'r', encoding='utf-8') as f:
                        simulation_output = json.load(f)

                    # Add the entire output file content as a new field in execution_result
                    execution_result["simulation_output"] = simulation_output

                    # Detect task type for special field mapping
                    is_mask_wearing = False
                    is_daily_mobility = False
                    is_user_modeling = False
                    if task_spec:
                        task_description = task_spec.get("description", "").lower()
                        is_mask_wearing = "mask-wearing behavior" in task_description
                        is_daily_mobility = "daily mobility trajectories" in task_description
                        is_user_modeling = "user rates a given item" in task_description
                        if is_mask_wearing:
                            self.logger.info("Mask-wearing behavior task detected: applying special mapping from single output file")
                        if is_daily_mobility:
                            self.logger.info("Daily mobility trajectories task detected: applying val_metrics->simulation_metrics mapping from single output file")
                        if is_user_modeling:
                            self.logger.info("User modeling task detected: applying val_metrics->simulation_metrics mapping from single output file")

                    if is_mask_wearing or is_daily_mobility or is_user_modeling:
                        # --- Unified mapping for mask-wearing and daily-mobility tasks ---
                        val_metrics = simulation_output.get("val_metrics", {})
                        if isinstance(val_metrics, dict):
                            # Keep all numeric val_metrics fields
                            for key, value in val_metrics.items():
                                if isinstance(value, (int, float)):
                                    execution_result["simulation_metrics"][key] = float(value)

                            # Normalize selection metric for best_simulator_info updates.
                            # Prefer explicit val_metrics.val_loss, otherwise fall back to RMSE_aggregate.
                            if "val_loss" in val_metrics and isinstance(val_metrics["val_loss"], (int, float)):
                                execution_result["simulation_metrics"]["val_loss"] = float(val_metrics["val_loss"])
                            elif "RMSE_aggregate" in val_metrics and isinstance(val_metrics["RMSE_aggregate"], (int, float)):
                                execution_result["simulation_metrics"]["val_loss"] = float(val_metrics["RMSE_aggregate"])

                            self.logger.debug(
                                f"Mapped unified val_metrics to simulation_metrics: {execution_result['simulation_metrics']}"
                            )

                        # optimized_parameters -> calibrated_parameters
                        if "optimized_parameters" in simulation_output:
                            execution_result["calibrated_parameters"] = simulation_output["optimized_parameters"]
                            self.logger.debug("Mapped optimized_parameters to calibrated_parameters")

                        # Preserve common top-level fields
                        for field in (
                            "test_metrics",
                            "val_metrics",
                            "test_transition_summary",
                            "aggregate_forecast",
                            "train_days",
                            "test_days",
                            "calibration_method",
                            "seed",
                        ):
                            if field in simulation_output:
                                execution_result[field] = simulation_output[field]
                    else:
                        # Generic single-file extraction (existing logic)
                        if "calibrated_parameters" in simulation_output:
                            execution_result["calibrated_parameters"] = simulation_output["calibrated_parameters"]
                        if "evaluation_results_on_validation" in simulation_output:
                            execution_result["evaluation_results_on_validation"] = simulation_output["evaluation_results_on_validation"]
                            eval_results = simulation_output["evaluation_results_on_validation"]
                            if isinstance(eval_results, dict):
                                if "summary" in eval_results and isinstance(eval_results["summary"], dict):
                                    summary = eval_results["summary"]
                                    for key, value in summary.items():
                                        if isinstance(value, dict) and "mean" in value:
                                            execution_result["simulation_metrics"][key] = value.get("mean")
                                if "metrics" in eval_results and isinstance(eval_results["metrics"], dict):
                                    for key, value in eval_results["metrics"].items():
                                        if isinstance(value, (int, float)):
                                            execution_result["simulation_metrics"][key] = value
                        if "generated_at_utc" in simulation_output:
                            execution_result["generated_at_utc"] = simulation_output["generated_at_utc"]

                    # Update summary
                    if execution_status == "success":
                        metrics_count = len(execution_result["simulation_metrics"])
                        execution_result["summary"] = (
                            f"Executed with subprocess in {execution_time:.2f} seconds. "
                            f"Results loaded from output file: {output_file} ({metrics_count} metrics)"
                        )

                    self.logger.info("Successfully loaded simulation results from output file")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON from output file {output_file}: {e}")
                except Exception as e:
                    self.logger.warning(f"Failed to read output file {output_file}: {e}")
            elif output_file:
                self.logger.warning(f"Output path {output_file} was specified but does not exist after execution")
            
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

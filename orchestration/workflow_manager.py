"""
WorkflowManager: Coordinates the workflow of agent interactions for simulation generation.
"""

import logging
import os
import yaml
import json
import sys
from typing import Dict, Any, List, Optional

from dependency_injector.wiring import inject, Provide
from orchestration.container import AgentContainer

class WorkflowManager:
    """
    Manages the workflow of agent interactions for generating simulation code.
    
    This class coordinates the interaction between various agents responsible for
    understanding tasks, analyzing data, planning models, generating code, verifying
    code, executing simulations, evaluating results, generating feedback, and
    controlling iteration.
    """
    
    @inject
    def __init__(
        self,
        task_description: str,
        data_path: Optional[str] = None,
        output_path: str = "./output",
        config_path: str = "./config.yaml",
        max_iterations: int = 3,
        agent_container: AgentContainer = Provide[AgentContainer]
    ):
        """
        Initialize the WorkflowManager.
        
        Args:
            task_description: Description of the simulation task
            data_path: Path to the task description JSON file
            output_path: Path to the output directory
            config_path: Path to the configuration file
            max_iterations: Maximum number of iterations
            agent_container: Dependency injection container for agents
        """
        self.logger = logging.getLogger("SOCIA.WorkflowManager")
        self.task_description = task_description
        self.task_file = data_path
        self.output_path = output_path
        self.config_path = config_path
        # Hard maximum iteration limit (user-specified) and initial soft window limit
        self.hard_max_iterations = max_iterations
        self.soft_max_iterations = 3
        
        # Initialize historical fix log to track issues across iterations
        self.historical_fix_log = {}
        
        # Load task file if provided
        self.task_data = None
        if self.task_file:
            try:
                with open(self.task_file, 'r') as f:
                    self.task_data = json.load(f)
                self.logger.info(f"Successfully loaded task description from {self.task_file}")
                
                # Set data path from task data
                self.data_path = self.task_data.get("data_folder", None)
                if self.data_path:
                    self.logger.info(f"Data folder set to: {self.data_path}")
                    
                    # Fix path to be relative to current directory
                    if not os.path.isabs(self.data_path):
                        # If the path contains a project name prefix like "SOCIA/", remove it
                        if self.data_path.startswith("SOCIA/"):
                            self.data_path = self.data_path[6:]  # Remove "SOCIA/" prefix
                        
                        # Check if path exists
                        if os.path.exists(self.data_path):
                            self.logger.info(f"Found data directory at: {self.data_path}")
                        else:
                            self.logger.warning(f"Specified data path doesn't exist: {self.data_path}")
                            
                            # Try looking in the same directory as the task file
                            task_file_dir = os.path.dirname(self.task_file)
                            alternative_path = os.path.join(task_file_dir, os.path.basename(self.data_path))
                            if os.path.exists(alternative_path):
                                self.data_path = alternative_path
                                self.logger.info(f"Using alternative data path: {self.data_path}")
                            else:
                                self.logger.warning(f"Alternative data path doesn't exist either: {alternative_path}")
            except Exception as e:
                self.logger.error(f"Error loading task file {self.task_file}: {e}")
                raise ValueError(f"Could not read task file: {self.task_file}")
        else:
            self.data_path = None
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize container with config
        self.container = agent_container
        self.container.config.from_dict(self.config)
        
        # Set output path in container
        self.container.output_path.override(self.output_path)
        self.logger.debug(f"Set container output_path to: {self.output_path}")
        
        # Get agent instances via dependency injection
        self.agents = self._initialize_agents()
        
        # State management
        self.current_iteration = 0
        self.state = {
            "task_spec": None,
            "data_analysis": None,
            "model_plan": None,
            "generated_code": None,
            "verification_results": None,
            "simulation_results": None,
            "evaluation_results": None,
            "feedback": None,
            "iteration_decision": None
        }
        # Initialize code memory to store generated code per iteration
        self.code_memory = {}
        # Add code_memory to state for persistence
        self.state["code_memory"] = self.code_memory
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from the config file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Use default configuration if loading fails
            return {
                "system": {"name": "SOCIA", "version": "0.1.0"},
                "workflow": {"max_iterations": self.hard_max_iterations}
            }
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents via dependency injection."""
        try:
            # Use the agent providers from the container
            agents = self.container.agent_providers()
            self.logger.info("Agents initialized via dependency injection")
            return agents
        except Exception as e:
            self.logger.error(f"Error initializing agents via DI: {e}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """
        Run the workflow to generate simulation code.
        
        Returns:
            Dict containing the paths to the final simulation code and evaluation results
        """
        self.logger.info("Starting workflow")
        
        while self.current_iteration < self.hard_max_iterations:
            # Log both hard maximum and current soft window limit
            self.logger.info(f"Starting iteration {self.current_iteration + 1}/{self.hard_max_iterations} (soft limit: {self.soft_max_iterations})")
            
            # Run the workflow for one iteration
            self._run_iteration()
            
            # Check if we should continue
            if not self.state["iteration_decision"]["continue"]:
                self.logger.info(f"Stopping after {self.current_iteration + 1} iterations: "
                                f"{self.state['iteration_decision']['reason']}")
                break
            
            self.current_iteration += 1
            # If we've hit the soft window limit but haven't reached the hard limit, extend the soft limit
            if self.current_iteration >= self.soft_max_iterations and self.soft_max_iterations < self.hard_max_iterations:
                new_soft = min(self.soft_max_iterations + 3, self.hard_max_iterations)
                self.logger.info(f"Reached soft iteration limit ({self.soft_max_iterations}), extending soft limit to {new_soft}")
                self.soft_max_iterations = new_soft
        
        # Save the final state
        self._save_state()
        
        return {
            "code_path": os.path.join(self.output_path, f"simulation_code_iter_{self.current_iteration}.py"),
            "evaluation_path": os.path.join(self.output_path, f"evaluation_iter_{self.current_iteration}.json")
        }
    
    def _run_iteration(self):
        """Run a single iteration of the workflow."""
        # Determine whether to skip initial setup for code-fix iterations
        skip_initial = self.current_iteration > 0 and self.state.get("simulation_results") is None
        if not skip_initial:
            # Step 1: Task Understanding
            if self.task_data:
                self.state["task_spec"] = self.agents["task_understanding"].process(
                    task_description=self.task_description,
                    task_data=self.task_data
                )
            else:
                self.state["task_spec"] = self.agents["task_understanding"].process(
                    task_description=self.task_description
                )
            self._save_artifact("task_spec", self.state["task_spec"])

            # Step 2: Data Analysis (if data is provided)
            if self.data_path:
                try:
                    self.state["data_analysis"] = self.agents["data_analysis"].process(
                        data_path=self.data_path,
                        task_spec=self.state["task_spec"]
                    )
                    self._save_artifact("data_analysis", self.state["data_analysis"])
                except Exception as e:
                    self.logger.error(f"Data analysis step failed: {e}. Aborting workflow.")
                    sys.exit(1)

            # Step 3: Model Planning
            self.state["model_plan"] = self.agents["model_planning"].process(
                task_spec=self.state["task_spec"],
                data_analysis=self.state["data_analysis"]
            )
            self._save_artifact("model_plan", self.state["model_plan"])
        else:
            self.logger.info("Skipping task understanding, data analysis, and model planning due to previous code verification failure.")

        # Step 4: Code Generation
        # Load previous iteration code if exists
        prev_code = None
        if self.current_iteration > 0 and self.current_iteration - 1 in self.code_memory:
            prev_code = self.code_memory[self.current_iteration - 1]
        # Generate code using CodeGenerationAgent with previous code context
        self.state["generated_code"] = self.agents["code_generation"].process(
            task_spec=self.state["task_spec"],
            data_analysis=self.state["data_analysis"],
            model_plan=self.state["model_plan"],
            feedback=self.state["feedback"],  # Will be None in first iteration
            data_path=self.data_path,
            previous_code=prev_code,
            historical_fix_log=self.historical_fix_log  # Pass historical fix log to code generation agent
        )
        self._save_artifact("generated_code", self.state["generated_code"])
        
        # Record generated code in memory and save as file
        gen_code_dict = self.state["generated_code"]["code"]
        self.code_memory[self.current_iteration] = {f"simulation_code_iter_{self.current_iteration}.py": gen_code_dict}
        code_file_path = os.path.join(self.output_path, f"simulation_code_iter_{self.current_iteration}.py")
        with open(code_file_path, 'w') as f:
            f.write(self.state["generated_code"]["code"])
        
        # Step 5: Code Verification
        self.state["verification_results"] = self.agents["code_verification"].process(
            code=self.state["generated_code"]["code"],
            task_spec=self.state["task_spec"],
            data_path=self.data_path
        )
        self._save_artifact("verification_results", self.state["verification_results"])
        
        # Log verification results clearly
        if self.state["verification_results"]["passed"]:
            self.logger.info(f"Iteration {self.current_iteration + 1}: Code verification PASSED")
        else:
            self.logger.warning(f"Iteration {self.current_iteration + 1}: Code verification FAILED")
            if "critical_issues" in self.state["verification_results"]:
                for issue in self.state["verification_results"]["critical_issues"]:
                    self.logger.warning(f"Critical issue: {issue}")
        
        # If code verification failed, skip execution and evaluation
        if not self.state["verification_results"]["passed"]:
            self.logger.warning("Code verification failed, skipping execution and evaluation")
            self.state["simulation_results"] = None
            self.state["evaluation_results"] = None
        else:
            # Step 6: Simulation Execution
            self.logger.info(f"Iteration {self.current_iteration + 1}: Starting simulation execution")
            self.state["simulation_results"] = self.agents["simulation_execution"].process(
                code_path=code_file_path,
                task_spec=self.state["task_spec"],
                data_path=self.data_path
            )
            self._save_artifact("simulation_results", self.state["simulation_results"])
            
            # Log simulation execution results
            if self.state["simulation_results"]["execution_status"] == "success":
                self.logger.info(f"Iteration {self.current_iteration + 1}: Simulation execution completed successfully")
            else:
                self.logger.warning(f"Iteration {self.current_iteration + 1}: Simulation execution failed: {self.state['simulation_results'].get('summary', 'Unknown error')}")
            
            # Step 7: Result Evaluation
            self.state["evaluation_results"] = self.agents["result_evaluation"].process(
                simulation_results=self.state["simulation_results"],
                task_spec=self.state["task_spec"],
                data_analysis=self.state["data_analysis"]
            )
            self._save_artifact("evaluation_results", self.state["evaluation_results"])
        
        # Step 8: Feedback Generation
        previous_code = None
        if self.current_iteration > 0 and self.current_iteration - 1 in self.code_memory:
            # Get previous iteration's code
            previous_code_dict = self.code_memory[self.current_iteration - 1]
            previous_code_filename = f"simulation_code_iter_{self.current_iteration - 1}.py"
            if previous_code_filename in previous_code_dict:
                previous_code = previous_code_dict[previous_code_filename]
        
        # Current code from code_memory
        current_code_dict = self.code_memory[self.current_iteration]
        current_code_filename = f"simulation_code_iter_{self.current_iteration}.py"
        current_code = current_code_dict[current_code_filename]
        
        self.state["feedback"] = self.agents["feedback_generation"].process(
            task_spec=self.state["task_spec"],
            model_plan=self.state["model_plan"],
            generated_code=self.state["generated_code"],
            verification_results=self.state["verification_results"],
            simulation_results=self.state["simulation_results"],
            evaluation_results=self.state["evaluation_results"],
            current_code=current_code,
            previous_code=previous_code,
            iteration=self.current_iteration,
            historical_fix_log=self.historical_fix_log  # Pass historical fix log to feedback agent
        )
        self._save_artifact("feedback", self.state["feedback"])
        
        # Update historical fix log with new critical issues
        self._update_historical_fix_log()
        
        # Step 9: Iteration Decision
        self.state["iteration_decision"] = self.agents["iteration_control"].process(
            current_iteration=self.current_iteration,
            max_iterations=self.soft_max_iterations,
            task_spec=self.state["task_spec"],
            verification_results=self.state["verification_results"],
            evaluation_results=self.state["evaluation_results"],
            feedback=self.state["feedback"]
        )
        self._save_artifact("iteration_decision", self.state["iteration_decision"])
        
        # Log iteration decision
        continue_msg = "CONTINUE" if self.state["iteration_decision"]["continue"] else "STOP"
        self.logger.info(f"Iteration {self.current_iteration + 1} decision: {continue_msg} - {self.state['iteration_decision'].get('reason', 'No reason provided')}")
        # Persist agent interactions for this iteration
        interactions = {
            "iteration": self.current_iteration,
            "interactions": {
                "task_understanding": {
                    "input": {"task_description": self.task_description, "task_data": self.task_data},
                    "output": self.state["task_spec"]
                },
                "data_analysis": {
                    "input": {"data_path": self.data_path, "task_spec": self.state["task_spec"]},
                    "output": self.state["data_analysis"]
                },
                "model_planning": {
                    "input": {"task_spec": self.state["task_spec"], "data_analysis": self.state["data_analysis"]},
                    "output": self.state["model_plan"]
                },
                "code_generation": {
                    "input": {"task_spec": self.state["task_spec"], "data_analysis": self.state["data_analysis"], "model_plan": self.state["model_plan"], "feedback": self.state["feedback"]},
                    "output": self.state["generated_code"]
                },
                "code_verification": {
                    "input": {"code": self.state["generated_code"]["code"], "task_spec": self.state["task_spec"]},
                    "output": self.state["verification_results"]
                },
                "simulation_execution": {
                    "input": {"code_path": code_file_path, "task_spec": self.state["task_spec"], "data_path": self.data_path},
                    "output": self.state["simulation_results"]
                },
                "result_evaluation": {
                    "input": {"simulation_results": self.state["simulation_results"], "task_spec": self.state["task_spec"], "data_analysis": self.state["data_analysis"]},
                    "output": self.state["evaluation_results"]
                },
                "feedback_generation": {
                    "input": {"task_spec": self.state["task_spec"], "model_plan": self.state["model_plan"], "generated_code": self.state["generated_code"], "verification_results": self.state["verification_results"], "simulation_results": self.state["simulation_results"], "evaluation_results": self.state["evaluation_results"], "code_file_path": code_file_path},
                    "output": self.state["feedback"]
                },
                "iteration_control": {
                    "input": {"current_iteration": self.current_iteration, "max_iterations": self.soft_max_iterations, "task_spec": self.state["task_spec"], "verification_results": self.state["verification_results"], "evaluation_results": self.state["evaluation_results"], "feedback": self.state["feedback"]},
                    "output": self.state["iteration_decision"]
                }
            }
        }
        interactions_file = os.path.join(self.output_path, f"interactions_iter_{self.current_iteration}.json")
        with open(interactions_file, 'w') as f:
            json.dump(interactions, f, indent=2)
    
    def _save_artifact(self, name: str, data: Any):
        """Save an artifact to the output directory."""
        if data is None:
            return
        
        filepath = os.path.join(self.output_path, f"{name}_iter_{self.current_iteration}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_state(self):
        """Save the current state to the output directory."""
        filepath = os.path.join(self.output_path, f"state_iter_{self.current_iteration}.json")
        with open(filepath, 'w') as f:
            # Convert state to a serializable format
            serializable_state = {k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v
                                for k, v in self.state.items()}
            json.dump(serializable_state, f, indent=2)
    
    def _update_historical_fix_log(self):
        """Update the historical fix log with critical issues from the current iteration."""
        try:
            # Check if feedback contains critical issues
            if self.state["feedback"] and "critical_issues" in self.state["feedback"]:
                critical_issues = self.state["feedback"].get("critical_issues", [])
                
                # Create iteration key
                iteration_key = f"iteration_{self.current_iteration}"
                
                # For each critical issue, add it to the historical fix log with status and fixed_log
                if iteration_key not in self.historical_fix_log:
                    self.historical_fix_log[iteration_key] = []
                
                for issue in critical_issues:
                    issue_with_status = issue.copy()
                    issue_with_status["status"] = "open"
                    issue_with_status["fixed_log"] = ""
                    self.historical_fix_log[iteration_key].append(issue_with_status)
                
                self.logger.info(f"Added {len(critical_issues)} critical issues to historical fix log for {iteration_key}")
            else:
                self.logger.info("No critical issues found in feedback to add to historical fix log")
                
            # Save historical fix log to file
            self._save_historical_fix_log()
            
        except Exception as e:
            self.logger.error(f"Error updating historical fix log: {e}")
    
    def _save_historical_fix_log(self):
        """Save the historical fix log to a file."""
        try:
            historical_log_path = os.path.join(self.output_path, "historical_fix_log.json")
            with open(historical_log_path, 'w') as f:
                json.dump(self.historical_fix_log, f, indent=2)
            self.logger.info(f"Saved historical fix log to {historical_log_path}")
        except Exception as e:
            self.logger.error(f"Error saving historical fix log: {e}") 
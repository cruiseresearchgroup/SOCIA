"""
WorkflowManager: Coordinates the workflow of agent interactions for simulation generation.
"""

import logging
import os
import yaml
import json
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
            data_path: Path to the input data directory
            output_path: Path to the output directory
            config_path: Path to the configuration file
            max_iterations: Maximum number of iterations
            agent_container: Dependency injection container for agents
        """
        self.logger = logging.getLogger("SOCIA.WorkflowManager")
        self.task_description = task_description
        self.data_path = data_path
        self.output_path = output_path
        self.config_path = config_path
        self.max_iterations = max_iterations
        
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
                "workflow": {"max_iterations": self.max_iterations}
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
        
        while self.current_iteration < self.max_iterations:
            self.logger.info(f"Starting iteration {self.current_iteration + 1}/{self.max_iterations}")
            
            # Run the workflow for one iteration
            self._run_iteration()
            
            # Check if we should continue
            if not self.state["iteration_decision"]["continue"]:
                self.logger.info(f"Stopping after {self.current_iteration + 1} iterations: "
                                f"{self.state['iteration_decision']['reason']}")
                break
            
            self.current_iteration += 1
        
        # Save the final state
        self._save_state()
        
        return {
            "code_path": os.path.join(self.output_path, f"simulation_code_iter_{self.current_iteration}.py"),
            "evaluation_path": os.path.join(self.output_path, f"evaluation_iter_{self.current_iteration}.json")
        }
    
    def _run_iteration(self):
        """Run a single iteration of the workflow."""
        # Step 1: Task Understanding
        self.state["task_spec"] = self.agents["task_understanding"].process(
            task_description=self.task_description
        )
        self._save_artifact("task_spec", self.state["task_spec"])
        
        # Step 2: Data Analysis (if data is provided)
        if self.data_path:
            self.state["data_analysis"] = self.agents["data_analysis"].process(
                data_path=self.data_path,
                task_spec=self.state["task_spec"]
            )
            self._save_artifact("data_analysis", self.state["data_analysis"])
        
        # Step 3: Model Planning
        self.state["model_plan"] = self.agents["model_planning"].process(
            task_spec=self.state["task_spec"],
            data_analysis=self.state["data_analysis"]
        )
        self._save_artifact("model_plan", self.state["model_plan"])
        
        # Step 4: Code Generation
        self.state["generated_code"] = self.agents["code_generation"].process(
            task_spec=self.state["task_spec"],
            data_analysis=self.state["data_analysis"],
            model_plan=self.state["model_plan"],
            feedback=self.state["feedback"]  # Will be None in first iteration
        )
        self._save_artifact("generated_code", self.state["generated_code"])
        
        # Save the generated code as a Python file
        code_file_path = os.path.join(self.output_path, f"simulation_code_iter_{self.current_iteration}.py")
        with open(code_file_path, 'w') as f:
            f.write(self.state["generated_code"]["code"])
        
        # Step 5: Code Verification
        self.state["verification_results"] = self.agents["code_verification"].process(
            code=self.state["generated_code"]["code"],
            task_spec=self.state["task_spec"]
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
        self.state["feedback"] = self.agents["feedback_generation"].process(
            task_spec=self.state["task_spec"],
            model_plan=self.state["model_plan"],
            generated_code=self.state["generated_code"],
            verification_results=self.state["verification_results"],
            simulation_results=self.state["simulation_results"],
            evaluation_results=self.state["evaluation_results"]
        )
        self._save_artifact("feedback", self.state["feedback"])
        
        # Step 9: Iteration Decision
        self.state["iteration_decision"] = self.agents["iteration_control"].process(
            current_iteration=self.current_iteration,
            max_iterations=self.max_iterations,
            task_spec=self.state["task_spec"],
            verification_results=self.state["verification_results"],
            evaluation_results=self.state["evaluation_results"],
            feedback=self.state["feedback"]
        )
        self._save_artifact("iteration_decision", self.state["iteration_decision"])
        
        # Log iteration decision
        continue_msg = "CONTINUE" if self.state["iteration_decision"]["continue"] else "STOP"
        self.logger.info(f"Iteration {self.current_iteration + 1} decision: {continue_msg} - {self.state['iteration_decision'].get('reason', 'No reason provided')}")
    
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
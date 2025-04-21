"""
WorkflowManager: Coordinates the workflow of agent interactions for simulation generation.
"""

import logging
import os
import yaml
import json
from typing import Dict, Any, List, Optional

class WorkflowManager:
    """
    Manages the workflow of agent interactions for generating simulation code.
    
    This class coordinates the interaction between various agents responsible for
    understanding tasks, analyzing data, planning models, generating code, verifying
    code, executing simulations, evaluating results, generating feedback, and
    controlling iteration.
    """
    
    def __init__(
        self,
        task_description: str,
        data_path: Optional[str] = None,
        output_path: str = "./output",
        config_path: str = "./config.yaml",
        max_iterations: int = 3
    ):
        """
        Initialize the WorkflowManager.
        
        Args:
            task_description: Description of the simulation task
            data_path: Path to the input data directory
            output_path: Path to the output directory
            config_path: Path to the configuration file
            max_iterations: Maximum number of iterations
        """
        self.logger = logging.getLogger("SOCIA.WorkflowManager")
        self.task_description = task_description
        self.data_path = data_path
        self.output_path = output_path
        self.config_path = config_path
        self.max_iterations = max_iterations
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize agents
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
        """Initialize all agents based on configuration."""
        from agents.task_understanding.agent import TaskUnderstandingAgent
        from agents.data_analysis.agent import DataAnalysisAgent
        from agents.model_planning.agent import ModelPlanningAgent
        from agents.code_generation.agent import CodeGenerationAgent
        from agents.code_verification.agent import CodeVerificationAgent
        from agents.simulation_execution.agent import SimulationExecutionAgent
        from agents.result_evaluation.agent import ResultEvaluationAgent
        from agents.feedback_generation.agent import FeedbackGenerationAgent
        from agents.iteration_control.agent import IterationControlAgent
        
        agents = {
            "task_understanding": TaskUnderstandingAgent(self.config["agents"]["task_understanding"]),
            "data_analysis": DataAnalysisAgent(self.config["agents"]["data_analysis"]),
            "model_planning": ModelPlanningAgent(self.config["agents"]["model_planning"]),
            "code_generation": CodeGenerationAgent(self.config["agents"]["code_generation"]),
            "code_verification": CodeVerificationAgent(self.config["agents"]["code_verification"]),
            "simulation_execution": SimulationExecutionAgent(self.config["agents"]["simulation_execution"]),
            "result_evaluation": ResultEvaluationAgent(self.config["agents"]["result_evaluation"]),
            "feedback_generation": FeedbackGenerationAgent(self.config["agents"]["feedback_generation"]),
            "iteration_control": IterationControlAgent(self.config["agents"]["iteration_control"])
        }
        
        return agents
    
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
        
        # If code verification failed, skip execution and evaluation
        if not self.state["verification_results"]["passed"]:
            self.logger.warning("Code verification failed, skipping execution and evaluation")
            self.state["simulation_results"] = None
            self.state["evaluation_results"] = None
        else:
            # Step 6: Simulation Execution
            self.state["simulation_results"] = self.agents["simulation_execution"].process(
                code_path=code_file_path,
                task_spec=self.state["task_spec"],
                data_path=self.data_path
            )
            self._save_artifact("simulation_results", self.state["simulation_results"])
            
            # Step 7: Result Evaluation
            self.state["evaluation_results"] = self.agents["result_evaluation"].process(
                simulation_results=self.state["simulation_results"],
                task_spec=self.state["task_spec"],
                data_analysis=self.state["data_analysis"]
            )
            self._save_artifact("evaluation_results", self.state["evaluation_results"])
        
        # Step 8: Feedback Generation
        self.state["feedback"] = self.agents["feedback_generation"].process(
            verification_results=self.state["verification_results"],
            simulation_results=self.state["simulation_results"],
            evaluation_results=self.state["evaluation_results"],
            task_spec=self.state["task_spec"]
        )
        self._save_artifact("feedback", self.state["feedback"])
        
        # Step 9: Iteration Control
        self.state["iteration_decision"] = self.agents["iteration_control"].process(
            current_iteration=self.current_iteration,
            max_iterations=self.max_iterations,
            verification_results=self.state["verification_results"],
            evaluation_results=self.state["evaluation_results"],
            feedback=self.state["feedback"]
        )
        self._save_artifact("iteration_decision", self.state["iteration_decision"])
    
    def _save_artifact(self, name: str, data: Any):
        """Save an artifact to the output directory."""
        if data is None:
            return
        
        file_path = os.path.join(self.output_path, f"{name}_iter_{self.current_iteration}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_state(self):
        """Save the current state to the output directory."""
        file_path = os.path.join(self.output_path, f"state_iter_{self.current_iteration}.json")
        with open(file_path, 'w') as f:
            # Convert the state to a JSON-serializable format
            serializable_state = {k: v for k, v in self.state.items() if k != "generated_code"}
            if "generated_code" in self.state and self.state["generated_code"]:
                serializable_state["generated_code"] = {
                    "metadata": self.state["generated_code"].get("metadata", {}),
                    "code_summary": self.state["generated_code"].get("code_summary", "")
                }
            json.dump(serializable_state, f, indent=2) 
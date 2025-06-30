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
        auto_mode: bool = False,
        mode: str = "full",
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
            auto_mode: If True, uses automatic feedback generation; if False, prompts user for manual feedback
            mode: Workflow mode (e.g., 'lite', 'medium', 'full')
            agent_container: Dependency injection container for agents
        """
        self.logger = logging.getLogger("SOCIA.WorkflowManager")
        self.task_description = task_description
        self.task_file = data_path
        self.output_path = output_path
        self.config_path = config_path
        # Hard maximum iteration limit (user-specified) and initial soft window limit
        if auto_mode:
            self.hard_max_iterations = max_iterations
            self.soft_max_iterations = 3
        else:
            # In manual mode, set a high hard limit since user controls stopping with #STOP#
            # Default to 100 unless user explicitly set a higher value
            self.hard_max_iterations = max(max_iterations, 100)
            self.soft_max_iterations = 3  # Start with soft limit of 3, can expand
        self.auto_mode = auto_mode
        self.mode = mode
        
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
    
    def _get_user_feedback(
        self,
        verification_results: Optional[Dict[str, Any]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        generated_code: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prompt user for manual feedback input, showing current iteration summary first.
        
        Args:
            verification_results: Results from code verification
            simulation_results: Results from simulation execution
            evaluation_results: Results from result evaluation
            generated_code: Generated code information
        
        Returns:
            User's feedback as a string
        """
        # First show the iteration summary
        self._display_iteration_summary(verification_results, simulation_results, evaluation_results, generated_code)
        
        print("\n" + "="*80)
        print("MANUAL FEEDBACK INPUT")
        print("="*80)
        print("Based on the execution summary above, please provide your feedback for the current iteration.")
        print("This feedback will be used to improve the simulation code in the next iteration.")
        print("You can include suggestions for:")
        print("- Code improvements or bug fixes")
        print("- Model accuracy enhancements")
        print("- Performance optimizations")
        print("- Any other observations or recommendations")
        print("\n⚠️  ITERATION CONTROL:")
        print("- If you want to STOP iterations and finalize results, type: #STOP#")
        print("- Otherwise, the system will continue to the next iteration after your feedback")
        print("\nIf you don't want to provide feedback, just press Enter twice to skip.")
        print("Otherwise, enter your feedback (press Enter twice to finish):")
        print("-"*80)
        
        feedback_lines = []
        empty_line_count = 0
        
        while True:
            try:
                line = input()
                if line.strip() == "":
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                    feedback_lines.append(line)
                else:
                    empty_line_count = 0
                    feedback_lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\nFeedback input interrupted.")
                break
        
        user_feedback = "\n".join(feedback_lines).strip()
        
        if user_feedback:
            print("-"*80)
            print("Your feedback has been recorded:")
            print(user_feedback)
            print("="*80)
        else:
            print("-"*80)
            print("No feedback provided. Using system-generated feedback only.")
            print("="*80)
        
        return user_feedback
    
    def _display_iteration_summary(
        self,
        verification_results: Optional[Dict[str, Any]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        generated_code: Optional[Dict[str, Any]] = None
    ):
        """
        Display a summary of the current iteration's execution status.
        
        Args:
            verification_results: Results from code verification
            simulation_results: Results from simulation execution
            evaluation_results: Results from result evaluation
            generated_code: Generated code information
        """
        print("\n" + "="*80)
        print(f"ITERATION {self.current_iteration + 1} EXECUTION SUMMARY")
        print("="*80)
        
        # Code Generation Summary
        print("📝 CODE GENERATION:")
        if generated_code:
            if "metadata" in generated_code:
                metadata = generated_code["metadata"]
                print(f"   ✓ Model Type: {metadata.get('model_type', 'Unknown')}")
                if 'entities' in metadata:
                    print(f"   ✓ Entities: {', '.join(metadata['entities'])}")
                if 'behaviors' in metadata:
                    print(f"   ✓ Behaviors: {', '.join(metadata['behaviors'])}")
            
            code_summary = generated_code.get("code_summary", "No summary available")
            print(f"   ✓ Summary: {code_summary}")
        else:
            print("   ❌ No code generation results available")
        
        # Code Verification Summary
        print("\n🔍 CODE VERIFICATION:")
        if verification_results:
            if verification_results.get("passed", False):
                print("   ✅ Status: PASSED")
                print(f"   ✓ Summary: {verification_results.get('summary', 'Verification successful')}")
            else:
                print("   ❌ Status: FAILED")
                if "critical_issues" in verification_results:
                    print("   ❌ Critical Issues:")
                    for issue in verification_results["critical_issues"]:
                        print(f"      • {issue}")
                if "warnings" in verification_results:
                    print("   ⚠️  Warnings:")
                    for warning in verification_results["warnings"]:
                        print(f"      • {warning}")
                print(f"   📋 Summary: {verification_results.get('summary', 'Verification failed')}")
        else:
            print("   ❓ No verification results available")
        
        # Simulation Execution Summary
        print("\n🚀 SIMULATION EXECUTION:")
        if simulation_results:
            execution_status = simulation_results.get("execution_status", "unknown")
            if execution_status == "success":
                print("   ✅ Status: SUCCESS")
                
                # Performance metrics
                if "performance_metrics" in simulation_results:
                    perf = simulation_results["performance_metrics"]
                    if "execution_time" in perf:
                        print(f"   ⏱️  Execution Time: {perf['execution_time']:.2f} seconds")
                    if "memory_usage" in perf:
                        print(f"   💾 Memory Usage: {perf['memory_usage']} MB")
                
                # Simulation metrics
                if "simulation_metrics" in simulation_results:
                    sim_metrics = simulation_results["simulation_metrics"]
                    print("   📊 Simulation Metrics:")
                    for key, value in sim_metrics.items():
                        print(f"      • {key}: {value}")
                
                # Time series data summary
                if "time_series_data" in simulation_results:
                    ts_data = simulation_results["time_series_data"]
                    if ts_data:
                        print(f"   📈 Time Series: {len(ts_data)} data points collected")
                
            elif execution_status == "failed":
                print("   ❌ Status: FAILED")
                if "runtime_errors" in simulation_results:
                    print("   ❌ Runtime Errors:")
                    for error in simulation_results["runtime_errors"]:
                        # Truncate very long error messages
                        error_str = str(error)
                        if len(error_str) > 200:
                            error_str = error_str[:200] + "..."
                        print(f"      • {error_str}")
            else:
                print(f"   ❓ Status: {execution_status.upper()}")
            
            summary = simulation_results.get("summary", "No summary available")
            print(f"   📋 Summary: {summary}")
        else:
            print("   ❓ No simulation execution results available")
        
        # Result Evaluation Summary
        print("\n📊 RESULT EVALUATION:")
        if evaluation_results:
            if "overall_score" in evaluation_results:
                score = evaluation_results["overall_score"]
                print(f"   📈 Overall Score: {score}")
            
            if "metrics" in evaluation_results:
                metrics = evaluation_results["metrics"]
                print("   📋 Evaluation Metrics:")
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            print(f"      • {metric_name}: {metric_value:.4f}")
                        else:
                            print(f"      • {metric_name}: {metric_value}")
                elif isinstance(metrics, list):
                    for i, metric in enumerate(metrics):
                        if isinstance(metric, dict):
                            for key, value in metric.items():
                                print(f"      • {key}: {value}")
                        else:
                            print(f"      • Metric {i+1}: {metric}")
                else:
                    print(f"      • {metrics}")
            
            if "recommendations" in evaluation_results:
                recommendations = evaluation_results["recommendations"]
                if recommendations:
                    print("   💡 Recommendations:")
                    for rec in recommendations[:3]:  # Show only first 3 recommendations
                        print(f"      • {rec}")
        else:
            print("   ❓ No evaluation results available")
        
        print("="*80)
    
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
    
    def _generate_feedback_and_control_iteration(self):
        """
        Common method to generate feedback and make iteration control decisions.
        Used by all workflow modes (lite, medium, full).
        """
        # Get current and previous code for feedback
        current_code_dict = self.code_memory[self.current_iteration]
        current_code_filename = f"simulation_code_iter_{self.current_iteration}.py"
        current_code = current_code_dict[current_code_filename]
        
        previous_code = None
        if self.current_iteration > 0 and self.current_iteration - 1 in self.code_memory:
            prev_code_dict = self.code_memory[self.current_iteration - 1]
            prev_code_filename = f"simulation_code_iter_{self.current_iteration - 1}.py"
            if isinstance(prev_code_dict, dict) and prev_code_filename in prev_code_dict:
                previous_code = prev_code_dict[prev_code_filename]
            elif isinstance(prev_code_dict, str):
                previous_code = prev_code_dict

        # Generate system feedback using the feedback agent
        if self.auto_mode:
            # In automatic mode, use system-generated feedback only
            self.state["feedback"] = self.agents["feedback_generation"].process(
                task_spec=self.state["task_spec"],
                model_plan=self.state["model_plan"],  # May be None in lite mode
                generated_code=self.state["generated_code"],
                verification_results=self.state["verification_results"],
                simulation_results=self.state["simulation_results"],
                evaluation_results=self.state["evaluation_results"],
                current_code=current_code,
                previous_code=previous_code,
                iteration=self.current_iteration,
                historical_fix_log=self.historical_fix_log
            )
        else:
            # In manual mode, get user feedback and combine with system feedback
            self.logger.info("Manual feedback mode enabled - prompting user for feedback")
            
            # Generate system feedback first
            system_feedback = self.agents["feedback_generation"].process(
                task_spec=self.state["task_spec"],
                model_plan=self.state["model_plan"],  # May be None in lite mode
                generated_code=self.state["generated_code"],
                verification_results=self.state["verification_results"],
                simulation_results=self.state["simulation_results"],
                evaluation_results=self.state["evaluation_results"],
                current_code=current_code,
                previous_code=previous_code,
                iteration=self.current_iteration,
                historical_fix_log=self.historical_fix_log
            )
            
            # Get user feedback with current iteration status
            user_feedback_text = self._get_user_feedback(
                verification_results=self.state["verification_results"],
                simulation_results=self.state["simulation_results"],
                evaluation_results=self.state["evaluation_results"],
                generated_code=self.state["generated_code"]
            )
            
            # Combine system and user feedback
            combined_feedback = dict(system_feedback)  # Copy system feedback
            
            # Add user feedback to the combined feedback
            if user_feedback_text:
                # Create user feedback section
                user_feedback_section = {
                    "source": "user",
                    "content": user_feedback_text,
                    "note": "This is user-provided feedback. Please pay special attention to these suggestions."
                }
                
                # Add user feedback to the combined feedback structure
                if "feedback_sections" not in combined_feedback:
                    combined_feedback["feedback_sections"] = []
                
                # Insert user feedback at the beginning to prioritize it
                combined_feedback["feedback_sections"].insert(0, {
                    "section": "USER_FEEDBACK",
                    "priority": "CRITICAL",
                    "feedback": user_feedback_section
                })
                
                # Also add to summary if it exists
                if "summary" in combined_feedback:
                    combined_feedback["summary"] = f"USER FEEDBACK: {user_feedback_text}\n\nSYSTEM FEEDBACK: {combined_feedback['summary']}"
                else:
                    combined_feedback["summary"] = f"USER FEEDBACK: {user_feedback_text}"
                    
                self.logger.info("User feedback has been integrated with system feedback")
            else:
                self.logger.info("No user feedback provided - using system feedback only")
            
            self.state["feedback"] = combined_feedback
        
        self._save_artifact("feedback", self.state["feedback"])
        
        # Update historical fix log with new critical issues
        self._update_historical_fix_log()
        
        # Iteration Decision
        # Extract user feedback if available for manual mode
        user_feedback_text = None
        if not self.auto_mode and self.state["feedback"]:
            # Try to extract user feedback from the combined feedback structure
            feedback_sections = self.state["feedback"].get("feedback_sections", [])
            for section in feedback_sections:
                if section.get("section") == "USER_FEEDBACK":
                    user_feedback_text = section.get("feedback", {}).get("content", "")
                    break
        
        self.state["iteration_decision"] = self.agents["iteration_control"].process(
            feedback=self.state["feedback"],
            verification_results=self.state["verification_results"],
            evaluation_results=self.state["evaluation_results"],
            current_iteration=self.current_iteration,
            max_iterations=self.soft_max_iterations,
            auto_mode=self.auto_mode,  # Ensure manual mode is passed correctly so #STOP# works
            user_feedback=user_feedback_text  # Pass user feedback for stop command check
        )
        self._save_artifact("iteration_decision", self.state["iteration_decision"])

    def _run_iteration(self):
        """Run a single iteration of the workflow."""
        if self.mode == 'full':
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
                historical_fix_log=self.historical_fix_log,  # Pass historical fix log to code generation agent
                mode="full"  # Use full mode
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
                if self.state["simulation_results"] and self.state["simulation_results"].get("execution_status") == "success":
                    self.logger.info(f"Iteration {self.current_iteration + 1}: Simulation execution completed successfully")
                else:
                    summary = "Unknown error"
                    if self.state["simulation_results"]:
                        summary = self.state["simulation_results"].get('summary', 'Unknown error')
                    self.logger.warning(f"Iteration {self.current_iteration + 1}: Simulation execution failed: {summary}")
                
                # Step 7: Result Evaluation
                self.state["evaluation_results"] = self.agents["result_evaluation"].process(
                    simulation_results=self.state["simulation_results"],
                    task_spec=self.state["task_spec"],
                    data_analysis=self.state["data_analysis"]
                )
                self._save_artifact("evaluation_results", self.state["evaluation_results"])
            
            # Step 8 & 9: Common feedback generation and iteration control
            self._generate_feedback_and_control_iteration()

        elif self.mode == 'medium':
            self.logger.info("Medium mode is a placeholder and not yet implemented. Stopping workflow.")
            self.state["iteration_decision"] = {"continue": False, "reason": "Medium mode not implemented."}
            
        elif self.mode == 'lite':
            # Lite mode workflow: Skip task understanding, data analysis, and model planning
            # Use task description directly as task spec for code generation
            
            # Step 1: Use task description directly as task spec
            self.state["task_spec"] = {
                "task_description": self.task_description,
                "simulation_type": "lite_mode",
                "objective": self.task_description
            }
            self._save_artifact("task_spec", self.state["task_spec"])
            
            # Step 2: Skip data analysis and model planning - set to None
            self.state["data_analysis"] = None
            self.state["model_plan"] = None
            
            # Step 3: Code Generation using lite template
            # Load previous iteration code if exists
            prev_code = None
            if self.current_iteration > 0 and self.current_iteration - 1 in self.code_memory:
                prev_code_dict = self.code_memory[self.current_iteration - 1]
                prev_code_filename = f"simulation_code_iter_{self.current_iteration - 1}.py"
                if isinstance(prev_code_dict, dict) and prev_code_filename in prev_code_dict:
                    prev_code = prev_code_dict[prev_code_filename]
                elif isinstance(prev_code_dict, str):
                    # Handle case where prev_code_dict is already a string
                    prev_code = prev_code_dict
            
            # Generate code using CodeGenerationAgent with lite template
            self.state["generated_code"] = self.agents["code_generation"].process(
                task_spec=self.state["task_spec"],
                data_analysis=None,  # Not used in lite mode
                model_plan=None,     # Not used in lite mode
                feedback=self.state["feedback"],  # Will be None in first iteration
                data_path=self.data_path,
                previous_code=prev_code,
                historical_fix_log=self.historical_fix_log,
                mode="lite"  # Use lite mode
            )
            self._save_artifact("generated_code", self.state["generated_code"])
            
            # Record generated code in memory and save as file
            gen_code_dict = self.state["generated_code"]["code"]
            self.code_memory[self.current_iteration] = {f"simulation_code_iter_{self.current_iteration}.py": gen_code_dict}
            code_file_path = os.path.join(self.output_path, f"simulation_code_iter_{self.current_iteration}.py")
            with open(code_file_path, 'w') as f:
                f.write(self.state["generated_code"]["code"])
            
            # Step 4: Code Verification
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
                # Step 5: Simulation Execution
                self.logger.info(f"Iteration {self.current_iteration + 1}: Starting simulation execution")
                self.state["simulation_results"] = self.agents["simulation_execution"].process(
                    code_path=code_file_path,
                    task_spec=self.state["task_spec"],
                    data_path=self.data_path
                )
                self._save_artifact("simulation_results", self.state["simulation_results"])
                
                # Log simulation execution results
                if self.state["simulation_results"] and self.state["simulation_results"].get("execution_status") == "success":
                    self.logger.info(f"Iteration {self.current_iteration + 1}: Simulation execution completed successfully")
                else:
                    summary = "Unknown error"
                    if self.state["simulation_results"]:
                        summary = self.state["simulation_results"].get('summary', 'Unknown error')
                    self.logger.warning(f"Iteration {self.current_iteration + 1}: Simulation execution failed: {summary}")
                
                # Step 6: Result Evaluation (simplified for lite mode)
                self.state["evaluation_results"] = self.agents["result_evaluation"].process(
                    simulation_results=self.state["simulation_results"],
                    task_spec=self.state["task_spec"],
                    data_analysis=None  # Not available in lite mode
                )
                self._save_artifact("evaluation_results", self.state["evaluation_results"])
            
            # Step 7 & 8: Common feedback generation and iteration control
            self._generate_feedback_and_control_iteration()
    
    def _save_artifact(self, name: str, data: Any):
        """Save an artifact to a JSON file."""
        if not data:
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
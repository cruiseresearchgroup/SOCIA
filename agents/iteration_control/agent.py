"""
IterationControlAgent: Controls the iteration process, deciding when to continue or stop.
"""

import logging
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseAgent

class IterationControlAgent(BaseAgent):
    """
    Iteration Control Agent determines whether another iteration of the simulation
    generation process is needed within the current soft window, and if so,
    what adjustments should be made for the next iteration.

    This agent uses a soft iteration limit (passed as max_iterations) to guide
    its decision. WorkflowManager may extend the soft window until the hard
    maximum iteration limit is reached.
    """
    
    def process(
        self,
        current_iteration: int,
        max_iterations: int,
        verification_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None,
        task_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decide whether to continue with another iteration and what to focus on.
        
        Args:
            current_iteration: Current iteration number (0-based)
            max_iterations: Soft maximum number of iterations in current window
            verification_results: Results from the Code Verification Agent (optional)
            evaluation_results: Results from the Result Evaluation Agent (optional)
            feedback: Feedback from the Feedback Generation Agent (optional)
            task_spec: Task specification from the Task Understanding Agent (optional)
        
        Returns:
            Dictionary containing the iteration decision
        """
        self.logger.info("Making iteration decision")
        
        # Build prompt for LLM to make the decision
        prompt = self._build_prompt(
            current_iteration=current_iteration,
            max_iterations=max_iterations,
            verification_results=verification_results,
            evaluation_results=evaluation_results,
            feedback=feedback,
            task_spec=task_spec
        )
        
        # Call LLM to make the decision
        llm_response = self._call_llm(prompt)
        
        # Parse the response
        iteration_decision = self._parse_llm_response(llm_response)
        
        # If LLM response parsing failed, create a basic result
        if isinstance(iteration_decision, str):
            iteration_decision = self._create_default_decision(current_iteration, max_iterations)
        
        # Ensure the result has the expected structure
        if "continue" not in iteration_decision:
            continue_iteration = current_iteration < max_iterations - 1
            iteration_decision["continue"] = continue_iteration
            iteration_decision["reason"] = "Default decision based on iteration count"
        
        self.logger.info(f"Iteration decision: {'continue' if iteration_decision['continue'] else 'stop'}")
        return iteration_decision
    
    def _create_default_decision(self, current_iteration: int, max_iterations: int) -> Dict[str, Any]:
        """
        Create a default iteration decision based on the iteration count.
        
        Args:
            current_iteration: Current iteration number (0-based)
            max_iterations: Maximum number of iterations
        
        Returns:
            Dictionary containing the default iteration decision
        """
        continue_iteration = current_iteration < max_iterations - 1
        
        return {
            "continue": continue_iteration,
            "reason": "Default decision based on iteration count",
            "convergence_assessment": {
                "code_quality": 0.5,
                "model_accuracy": 0.5,
                "overall_convergence": 0.5
            },
            "next_iteration_focus": {
                "primary_focus": "both",
                "specific_areas": [
                    "Improve code implementation",
                    "Enhance model accuracy"
                ]
            },
            "agent_adjustments": {
                "task_understanding": {
                    "adjust": False,
                    "adjustments": "No adjustments needed"
                },
                "data_analysis": {
                    "adjust": False,
                    "adjustments": "No adjustments needed"
                },
                "model_planning": {
                    "adjust": True,
                    "adjustments": "Refine the model based on evaluation results"
                },
                "code_generation": {
                    "adjust": True,
                    "adjustments": "Incorporate feedback to improve code quality"
                },
                "code_verification": {
                    "adjust": False,
                    "adjustments": "No adjustments needed"
                },
                "simulation_execution": {
                    "adjust": False,
                    "adjustments": "No adjustments needed"
                },
                "result_evaluation": {
                    "adjust": False,
                    "adjustments": "No adjustments needed"
                },
                "feedback_generation": {
                    "adjust": False,
                    "adjustments": "No adjustments needed"
                }
            }
        }
    
    def _assess_convergence(
        self,
        verification_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Assess the convergence of the simulation generation process.
        
        Args:
            verification_results: Results from the Code Verification Agent (optional)
            evaluation_results: Results from the Result Evaluation Agent (optional)
        
        Returns:
            Dictionary containing convergence assessment scores
        """
        # Default values
        code_quality = 0.5
        model_accuracy = 0.5
        
        # Assess code quality from verification results
        if verification_results:
            if verification_results.get("passed", False):
                code_quality = 0.8
            else:
                issues = verification_results.get("issues", [])
                if issues:
                    # Count critical issues
                    critical_count = sum(1 for issue in issues if issue.get("severity") in ["critical", "high"])
                    if critical_count > 0:
                        code_quality = 0.2
                    else:
                        code_quality = 0.6
        
        # Assess model accuracy from evaluation results
        if evaluation_results:
            overall_eval = evaluation_results.get("overall_evaluation", {})
            if "score" in overall_eval:
                model_accuracy = overall_eval["score"]
        
        # Calculate overall convergence as weighted average
        overall_convergence = 0.4 * code_quality + 0.6 * model_accuracy
        
        return {
            "code_quality": code_quality,
            "model_accuracy": model_accuracy,
            "overall_convergence": overall_convergence
        } 
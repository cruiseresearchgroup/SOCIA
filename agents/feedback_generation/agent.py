"""
FeedbackGenerationAgent: Generates feedback for improving the simulation based on verification and evaluation results.
"""

import logging
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseAgent

class FeedbackGenerationAgent(BaseAgent):
    """
    Feedback Generation Agent synthesizes the results of verification, execution,
    and evaluation to produce actionable feedback for improving the simulation.
    
    This agent is responsible for:
    1. Identifying critical issues that need to be addressed
    2. Suggesting improvements to the model and code
    3. Prioritizing actions for the next iteration
    4. Providing specific guidance on how to implement improvements
    """
    
    def process(
        self,
        task_spec: Dict[str, Any],
        verification_results: Optional[Dict[str, Any]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        model_plan: Optional[Dict[str, Any]] = None,
        generated_code: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate feedback for improving the simulation.
        
        Args:
            task_spec: Task specification from the Task Understanding Agent
            verification_results: Results from the Code Verification Agent (optional)
            simulation_results: Results from the Simulation Execution Agent (optional)
            evaluation_results: Results from the Result Evaluation Agent (optional)
            model_plan: Model plan from the Model Planning Agent (optional)
            generated_code: Generated code from the Code Generation Agent (optional)
        
        Returns:
            Dictionary containing feedback for improvement
        """
        self.logger.info("Generating feedback for improvement")
        
        # Build prompt for LLM to generate feedback
        prompt = self._build_prompt(
            task_spec=task_spec,
            verification_results=verification_results,
            simulation_results=simulation_results,
            evaluation_results=evaluation_results,
            model_plan=model_plan,
            generated_code=generated_code
        )
        
        # Call LLM to generate feedback
        llm_response = self._call_llm(prompt)
        
        # Parse the response
        feedback = self._parse_llm_response(llm_response)
        
        # If LLM response parsing failed, create a basic result
        if isinstance(feedback, str):
            feedback = self._create_placeholder_feedback()
        
        self.logger.info("Feedback generation completed")
        return feedback
    
    def _create_placeholder_feedback(self) -> Dict[str, Any]:
        """Create a placeholder feedback result."""
        return {
            "summary": "The simulation provides a good starting point but needs refinements in both model design and implementation",
            "critical_issues": [
                {
                    "issue": "Lack of validation against real data",
                    "impact": "Simulation may not accurately reflect real-world behavior",
                    "solution": "Implement more detailed validation metrics comparing simulation outputs to real data"
                }
            ],
            "model_improvements": [
                {
                    "aspect": "Agent behavior",
                    "current_approach": "Simple rule-based behavior",
                    "suggested_approach": "More sophisticated decision-making model based on utility functions",
                    "expected_benefit": "More realistic agent decisions that better match observed patterns"
                }
            ],
            "code_improvements": [
                {
                    "file": "simulation.py",
                    "modification": "Add error handling for edge cases",
                    "reason": "Currently, the simulation may crash when unexpected inputs are provided"
                }
            ],
            "data_alignment_suggestions": [
                {
                    "metric": "Activity distribution",
                    "current_gap": "Simulation shows uniform activity, real data shows peaks",
                    "suggestion": "Add time-dependency to agent activity levels"
                }
            ],
            "prioritized_actions": [
                "Fix critical bugs in the implementation",
                "Improve the agent behavior model",
                "Add more detailed validation metrics"
            ],
            "additional_comments": "Overall, the simulation shows promise but needs refinement in key areas"
        }
    
    def _extract_issues_from_verification(
        self,
        verification_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract issues from verification results.
        
        Args:
            verification_results: Results from the Code Verification Agent
        
        Returns:
            List of critical issues extracted from verification results
        """
        critical_issues = []
        
        # Extract issues from verification results
        if verification_results and "issues" in verification_results:
            for issue in verification_results["issues"]:
                if issue.get("severity") in ["critical", "high"]:
                    critical_issues.append({
                        "issue": issue.get("description", "Unknown issue"),
                        "impact": "May cause the simulation to fail or produce incorrect results",
                        "solution": issue.get("solution", "Fix the issue")
                    })
        
        return critical_issues
    
    def _extract_issues_from_evaluation(
        self,
        evaluation_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract issues from evaluation results.
        
        Args:
            evaluation_results: Results from the Result Evaluation Agent
        
        Returns:
            List of critical issues extracted from evaluation results
        """
        critical_issues = []
        
        # Extract issues from evaluation results
        if evaluation_results:
            # Extract weaknesses
            if "weaknesses" in evaluation_results:
                for weakness in evaluation_results["weaknesses"]:
                    critical_issues.append({
                        "issue": weakness,
                        "impact": "Reduces the accuracy or usefulness of the simulation",
                        "solution": "Address this weakness in the next iteration"
                    })
            
            # Extract poor matches from detailed comparisons
            if "detailed_comparisons" in evaluation_results:
                for comparison in evaluation_results["detailed_comparisons"]:
                    if comparison.get("match_quality") == "poor":
                        critical_issues.append({
                            "issue": f"Poor match in {comparison.get('aspect')}",
                            "impact": "Simulation does not accurately reflect reality in this aspect",
                            "solution": "Revise the model to better match real-world behavior"
                        })
        
        return critical_issues 
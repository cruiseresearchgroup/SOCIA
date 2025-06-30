"""
CodeGenerationAgent: Generates simulation code based on the model plan.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseAgent

class CodeGenerationAgent(BaseAgent):
    """
    Code Generation Agent transforms the model plan into executable Python code
    for the simulation.
    
    This agent is responsible for:
    1. Generating code that implements the model plan
    2. Creating modular, maintainable, and well-documented code
    3. Following best practices and coding standards
    4. Incorporating feedback from previous iterations (if available)
    """
    
    def process(
        self,
        task_spec: Dict[str, Any],
        model_plan: Optional[Dict[str, Any]] = None,
        data_analysis: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None,
        previous_code: Optional[Dict[str, str]] = None,
        historical_fix_log: Optional[Dict[str, Any]] = None,
        mode: str = "full"
    ) -> Dict[str, Any]:
        """
        Generate simulation code based on the model plan.
        
        Args:
            task_spec: Task specification from the Task Understanding Agent
            model_plan: Model plan from the Model Planning Agent (optional, not used in lite mode)
            data_analysis: Data analysis results from the Data Analysis Agent (optional)
            feedback: Feedback from previous iterations (optional)
            data_path: Original data directory path (optional)
            previous_code: Code from the previous iteration for context (optional)
            historical_fix_log: Log of historical issues and their fix status (optional)
            mode: Workflow mode ('lite', 'medium', 'full'). Defaults to 'full'.
        
        Returns:
            Dictionary containing the generated code and metadata
        """
        self.logger.info("Generating simulation code")
        
        # Override model_plan data_sources with processed file paths (skip in lite mode)
        if mode != "lite" and model_plan and data_analysis and "file_references" in data_analysis:
            self.logger.info("Overriding model_plan data_sources with processed file paths")
            # Copy model_plan to avoid mutating original
            model_plan = dict(model_plan)
            new_sources = []
            for ds in model_plan.get("data_sources", []):
                name = ds.get("name")
                # If processed path exists, include it
                if name in data_analysis["file_references"]:
                    ds["path"] = data_analysis["file_references"][name]
                new_sources.append(ds)
            model_plan["data_sources"] = new_sources
        
        # Build prompt from template, including original data path
        prompt = self._build_prompt(
            task_spec=task_spec,
            model_plan=model_plan,
            data_analysis=data_analysis,
            feedback=feedback,
            data_path=data_path,
            previous_code=previous_code,
            mode=mode
        )
        
        # Call LLM to generate code
        llm_response = self._call_llm(prompt)
        
        # Extract code from the response
        # Since code generation typically produces Python code rather than JSON,
        # we handle the response differently
        code = self._extract_code(llm_response)
        # Remove any leftover markdown fences
        code = self._strip_markdown_fences(code)
        # Apply feedback snippets if available
        if feedback and isinstance(feedback, dict) and 'code_snippets' in feedback:
            for snippet in feedback['code_snippets']:
                before = snippet.get('before', '')
                after = snippet.get('after', '')
                if before and after and before in code:
                    self.logger.info(f"Applying feedback snippet from {snippet.get('file')}")
                    code = code.replace(before, f"# FIXED: Applied feedback snippet from {snippet.get('file')}\n{after}")
        # Automatically fix unclosed triple-quoted strings
        code = self._fix_unclosed_docstrings(code)
        
        # Run self-checking loop to improve the code - always run, regardless of feedback or historical_fix_log
        # Skip self-checking in lite mode for faster execution
        if mode != "lite":
            self.logger.info("Starting self-checking loop for code improvement")
            code = self._run_self_checking_loop(
                code=code,
                task_spec=task_spec,
                model_plan=model_plan,
                feedback=feedback,
                historical_fix_log=historical_fix_log
            )
        
        # Generate a summary of the code
        code_summary = self._generate_code_summary(code)
        
        result = {
            "code": code,
            "code_summary": code_summary,
            "metadata": {
                "model_type": model_plan.get("model_type", mode) if model_plan else mode,
                "entities": [e.get("name") for e in model_plan.get("entities", [])] if model_plan else [],
                "behaviors": [b.get("name") for b in model_plan.get("behaviors", [])] if model_plan else [],
                "mode": mode
            }
        }
        
        self.logger.info("Code generation completed")
        # Post-generation syntax check and auto-fix
        try:
            compile(result['code'], '<generated>', 'exec')
        except SyntaxError as err:
            self.logger.warning(f"Syntax error detected: {err}. Attempting to auto-fix via LLM.")
            # Build fix prompt
            fix_prompt = (
                "The following Python code has a syntax error. Please provide a corrected version of the code.\n"
                f"Error: {err}\n"
                "Original code:\n```python\n" + result['code'] + "\n```\n"
            )
            # Call LLM to fix syntax
            llm_fix_response = self._call_llm(fix_prompt)
            # Extract corrected code
            fixed_code = self._extract_code(llm_fix_response)
            # Remove any leftover markdown fences from the LLM fix
            fixed_code = self._strip_markdown_fences(fixed_code)
            # Apply local docstring and entry-point fixes
            fixed_code = self._fix_unclosed_docstrings(fixed_code)
            fixed_code = self._ensure_entry_point(fixed_code)
            # Update result
            result['code'] = fixed_code
            result['code_summary'] = self._generate_code_summary(fixed_code)
        return result
    
    def _run_self_checking_loop(
        self,
        code: str,
        task_spec: Dict[str, Any],
        model_plan: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None,
        historical_fix_log: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Run a self-checking loop to improve the generated code.
        
        The loop performs several self-checks:
        1. Comprehensive code quality check (covers syntax errors, placeholders, missing implementations, etc.)
        2. Check if all required fixes from feedback are implemented (if feedback exists)
        3. Check if the code repeats past errors from the historical fix log (if historical_fix_log exists)
        
        If issues are found, the code is improved and the checks are run again.
        This process is repeated up to three times.
        
        Args:
            code: The generated code
            task_spec: Task specification from the Task Understanding Agent
            model_plan: Model plan from the Model Planning Agent
            feedback: Feedback from previous iterations (optional)
            historical_fix_log: Log of historical issues and their fix status (optional)
            
        Returns:
            Improved code after self-checking loop
        """
        improved_code = code
        max_attempts = 3
        
        for attempt in range(max_attempts):
            self.logger.info(f"Self-checking loop - Attempt {attempt + 1}/{max_attempts}")
            
            # Run all self-checks and collect issues
            issues = []
            
            # Always run comprehensive code quality check
            # This covers syntax errors, placeholders, missing implementations, undefined references, etc.
            issues.extend(self._perform_code_quality_check(improved_code, task_spec, model_plan))
            
            # Run feedback-based check only if feedback exists
            if feedback:
                issues.extend(self._check_feedback_implementation(improved_code, feedback))
            
            # Run historical issues check only if historical_fix_log exists
            if historical_fix_log:
                issues.extend(self._check_historical_issues(improved_code, historical_fix_log))
            
            # If no issues found, we're done
            if not issues:
                self.logger.info(f"Self-checking passed on attempt {attempt + 1}")
                break
            
            # Log issues found
            self.logger.info(f"Found {len(issues)} issues in self-checking. Attempting to improve code.")
            
            # Collect relevant fixed_log entries for reference
            fixed_log_references = self._collect_fixed_log_references(issues, historical_fix_log)
            
            # Improve the code based on issues and fixed_log references
            improved_code = self._improve_code_based_on_issues(
                code=improved_code,
                issues=issues,
                fixed_log_references=fixed_log_references,
                task_spec=task_spec,
                model_plan=model_plan
            )
            
            # Fix any unclosed docstrings that might have been introduced
            improved_code = self._fix_unclosed_docstrings(improved_code)
            
            # Check for syntax errors
            try:
                compile(improved_code, '<improved>', 'exec')
                self.logger.info("Improved code passed syntax check")
            except SyntaxError as err:
                self.logger.warning(f"Syntax error in improved code: {err}")
                # If this is the last attempt, we should at least make sure the code compiles
                if attempt == max_attempts - 1:
                    improved_code = self._fix_syntax(improved_code, err)
            
            # If this is the last attempt, log a warning
            if attempt == max_attempts - 1 and issues:
                self.logger.warning("Maximum self-checking attempts reached but issues remain")
        
        return improved_code
    
    def _perform_code_quality_check(
        self,
        code: str,
        task_spec: Dict[str, Any],
        model_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Perform a comprehensive code quality check.
        
        This check looks for:
        1. Syntax errors and potential runtime errors
        2. Placeholder functions (those with just 'pass')
        3. Unimplemented methods or functionality
        4. Inconsistencies between the model plan and implementation
        5. Inefficient code or algorithms
        6. Poor error handling
        7. Lack of type annotations
        8. Missing or incomplete documentation
        
        Args:
            code: The code to check
            task_spec: Task specification from the Task Understanding Agent
            model_plan: Model plan from the Model Planning Agent
            
        Returns:
            List of issues found
        """
        self.logger.info("Performing comprehensive code quality check")
        
        # Build prompt for code quality check
        prompt = f"""
        You are a code quality analyst for Python. Your task is to analyze the following code for a simulation model and identify any quality issues or potential problems.
        
        Generated code:
        ```python
        {code}
        ```
        
        Task specification:
        {json.dumps(task_spec, indent=2)}
        
        Model plan:
        {json.dumps(model_plan, indent=2)}
        
        SPECIAL REQUIREMENTS:
        - At the end of the file, include a direct call to the main() function (e.g., `# Execute main for both direct execution and sandbox wrapper invocation\nmain()`) instead of using the traditional `if __name__ == "__main__"` guard to ensure compatibility with sandbox execution. This is a STANDARD REQUIREMENT for all simulations in this system and should NOT be considered an issue.
        
        Perform a comprehensive code review checking for the following issues:
        
        1. SYNTAX ERRORS: Any syntax errors or code that might cause runtime errors
        2. PLACEHOLDERS: Placeholder functions or methods (containing only 'pass' without implementation)
        3. MISSING IMPLEMENTATIONS: Required methods or functionality mentioned in the model plan but not implemented
        4. INCONSISTENCIES: Inconsistencies between class/method names in the model plan and the implementation
        5. UNDEFINED REFERENCES: References to undefined variables, methods, or classes
        6. ERROR HANDLING: Missing or inadequate error handling, especially for file operations and data processing
        7. COMPLETENESS: Check that all components from the model plan (entities, behaviors, interactions) are implemented
        8. DOCUMENTATION: Missing or incomplete docstrings for classes and methods
        9. TYPE ANNOTATIONS: Missing or incorrect type annotations
        10. ALGORITHM EFFICIENCY: Inefficient algorithms or code patterns
        
        Return a JSON array of issues found. Each issue should have:
        1. "type": One of the categories above
        2. "severity": "critical" (must fix), "major" (should fix), or "minor" (nice to fix)
        3. "description": Detailed description of the issue
        4. "location": Where in the code the issue occurs (e.g., class/method name or approx. line reference)
        5. "recommendation": Your specific recommendation on how to fix it
        
        If no issues are found, return an empty array.
        
        Format your response as a valid JSON array like this:
        [
          {{
            "type": "PLACEHOLDERS",
            "severity": "critical",
            "description": "The method process_interaction() in Environment class only has a placeholder 'pass' statement",
            "location": "Environment.process_interaction()",
            "recommendation": "Implement the interaction logic to update agent states based on proximity and interaction rules"
          }}
        ]
        """
        
        # Call LLM to perform code quality check
        llm_response = self._call_llm(prompt)
        
        # Parse LLM response
        try:
            # Extract JSON from response
            first_bracket = llm_response.find('[')
            last_bracket = llm_response.rfind(']')
            
            if first_bracket == -1 or last_bracket == -1:
                self.logger.warning("Could not find JSON array in LLM response for code quality check")
                return []
            
            json_str = llm_response[first_bracket:last_bracket+1]
            issues = json.loads(json_str)
            
            if not issues:
                self.logger.info("No quality issues found in the code")
            else:
                self.logger.warning(f"Found {len(issues)} code quality issues")
                # Log critical issues
                critical_issues = [issue for issue in issues if issue.get("severity") == "critical"]
                if critical_issues:
                    self.logger.warning(f"Found {len(critical_issues)} CRITICAL issues that must be fixed")
                    for issue in critical_issues:
                        self.logger.warning(f"Critical issue: {issue.get('description')} in {issue.get('location')}")
                
            return issues
        except Exception as e:
            self.logger.error(f"Error parsing code quality check response: {e}")
            return []
    
    def _check_feedback_implementation(self, code: str, feedback: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Check if all required fixes from feedback are implemented.
        
        Args:
            code: The code to check
            feedback: Feedback from previous iterations
            
        Returns:
            List of issues found
        """
        if not feedback:
            return []
        
        self.logger.info("Checking if all required fixes from feedback are implemented")
        
        # Build prompt for checking feedback implementation
        prompt = f"""
        You are a code quality checker. Your task is to check if the following code has implemented all required fixes from the feedback.
        
        Feedback that needs to be implemented:
        {json.dumps(feedback, indent=2)}
        
        Generated code:
        ```python
        {code}
        ```
        
        SPECIAL REQUIREMENTS:
        - At the end of the file, include a direct call to the main() function (e.g., `# Execute main for both direct execution and sandbox wrapper invocation\nmain()`) instead of using the traditional `if __name__ == "__main__"` guard to ensure compatibility with sandbox execution. This is a STANDARD REQUIREMENT for all simulations in this system and should NOT be considered an issue.
        
        Check if all critical issues, required code improvements, and prioritized actions from the feedback have been implemented in the code.
        
        Return a JSON array of issues that are not properly implemented. Each issue should have:
        1. "type": The type of issue (e.g., "critical_issue", "code_improvement", "prioritized_action")
        2. "description": Description of the issue that was not implemented
        3. "recommendation": Your recommendation on how to fix it
        
        If all issues are properly implemented, return an empty array.
        
        Format your response as a valid JSON array like this:
        [
          {{
            "type": "critical_issue",
            "description": "The error handling for file operations is missing",
            "recommendation": "Add try-except blocks around file operations"
          }}
        ]
        """
        
        # Call LLM to check feedback implementation
        llm_response = self._call_llm(prompt)
        
        # Parse LLM response
        try:
            # Extract JSON from response
            first_bracket = llm_response.find('[')
            last_bracket = llm_response.rfind(']')
            
            if first_bracket == -1 or last_bracket == -1:
                self.logger.warning("Could not find JSON array in LLM response for feedback implementation check")
                return []
            
            json_str = llm_response[first_bracket:last_bracket+1]
            issues = json.loads(json_str)
            
            if not issues:
                self.logger.info("All feedback issues are properly implemented")
            else:
                self.logger.warning(f"Found {len(issues)} feedback issues that are not properly implemented")
                
            return issues
        except Exception as e:
            self.logger.error(f"Error parsing feedback implementation check response: {e}")
            return []
    
    def _check_historical_issues(self, code: str, historical_fix_log: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Check if the code repeats issues from the historical fix log.
        
        Args:
            code: The code to check
            historical_fix_log: Log of historical issues and their fix status
            
        Returns:
            List of issues found
        """
        if not historical_fix_log:
            return []
        
        self.logger.info("Checking if code repeats issues from historical fix log")
        
        # Extract fixed issues from historical fix log
        fixed_issues = []
        for iteration_key, issues in historical_fix_log.items():
            for issue in issues:
                if issue.get("status") == "fixed" and issue.get("fixed_log"):
                    fixed_issues.append({
                        "issue": issue.get("issue", ""),
                        "fixed_log": issue.get("fixed_log", ""),
                        "iteration": iteration_key
                    })
        
        if not fixed_issues:
            self.logger.info("No fixed issues found in historical fix log")
            return []
        
        # Build prompt for checking historical issues
        prompt = f"""
        You are a code quality checker. Your task is to check if the following code repeats issues that were fixed in the past.
        
        Generated code:
        ```python
        {code}
        ```
        
        Previously fixed issues:
        {json.dumps(fixed_issues, indent=2)}
        
        SPECIAL REQUIREMENTS:
        - At the end of the file, include a direct call to the main() function (e.g., `# Execute main for both direct execution and sandbox wrapper invocation\nmain()`) instead of using the traditional `if __name__ == "__main__"` guard to ensure compatibility with sandbox execution. This is a STANDARD REQUIREMENT for all simulations in this system and should NOT be considered an issue.
        
        Check if the code repeats any of the issues that were fixed previously. 
        Consider both the issue description and the fix log to understand what was fixed.
        
        Return a JSON array of issues found. Each issue should have:
        1. "issue": The original issue text
        2. "fixed_log": The fixed log text that explains how it was fixed before
        3. "description": Your description of how the current code repeats this issue
        4. "iteration": The iteration key where this issue was originally fixed
        
        If no issues are found, return an empty array.
        
        Format your response as a valid JSON array like this:
        [
          {{
            "issue": "Missing error handling for file operations",
            "fixed_log": "Added try-except blocks around file operations",
            "description": "The code still lacks error handling for file operations in the save_results method",
            "iteration": "iteration_1"
          }}
        ]
        """
        
        # Call LLM to check historical issues
        llm_response = self._call_llm(prompt)
        
        # Parse LLM response
        try:
            # Extract JSON from response
            first_bracket = llm_response.find('[')
            last_bracket = llm_response.rfind(']')
            
            if first_bracket == -1 or last_bracket == -1:
                self.logger.warning("Could not find JSON array in LLM response for historical issues check")
                return []
            
            json_str = llm_response[first_bracket:last_bracket+1]
            issues = json.loads(json_str)
            
            if not issues:
                self.logger.info("Code does not repeat any issues from historical fix log")
            else:
                self.logger.warning(f"Found {len(issues)} repeats of previously fixed issues")
                
            return issues
        except Exception as e:
            self.logger.error(f"Error parsing historical issues check response: {e}")
            return []
    
    def _collect_fixed_log_references(self, issues: List[Dict[str, Any]], historical_fix_log: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect fixed_log references from historical_fix_log based on issues found.
        
        Args:
            issues: List of issues found
            historical_fix_log: Log of historical issues and their fix status
            
        Returns:
            String with fixed_log references
        """
        if not historical_fix_log or not issues:
            return ""
        
        # Collect fixed_log references from historical issues check
        fixed_log_refs = []
        for issue in issues:
            if "fixed_log" in issue and issue["fixed_log"]:
                fixed_log_refs.append(f"Issue: {issue.get('issue', '')}\nFix: {issue['fixed_log']}")
        
        if fixed_log_refs:
            return "Reference fixes from historical log:\n" + "\n\n".join(fixed_log_refs)
        else:
            return ""
    
    def _improve_code_based_on_issues(
        self,
        code: str,
        issues: List[Dict[str, Any]],
        fixed_log_references: str,
        task_spec: Dict[str, Any],
        model_plan: Dict[str, Any]
    ) -> str:
        """
        Improve code based on issues found and fixed_log references.
        
        Args:
            code: The code to improve
            issues: List of issues found
            fixed_log_references: References to fixed_log entries
            task_spec: Task specification from the Task Understanding Agent
            model_plan: Model plan from the Model Planning Agent
            
        Returns:
            Improved code
        """
        self.logger.info("Improving code based on self-checking issues")
        
        # Format issues for the prompt
        issues_text = json.dumps(issues, indent=2)
        
        # Build prompt for improving code
        prompt = f"""
        You are a code improvement agent. Your task is to improve the following code based on the issues found during self-checking.
        
        Generated code:
        ```python
        {code}
        ```
        
        Issues found during self-checking:
        {issues_text}
        
        {fixed_log_references}
        
        Task specification:
        {json.dumps(task_spec, indent=2)}
        
        Model plan:
        {json.dumps(model_plan, indent=2)}
        
        Carefully address each issue:
        1. For feedback implementation issues, ensure all required fixes are properly implemented
        2. For undefined method issues, define missing methods or fix incorrect method calls
        3. For functionality preservation issues, ensure all required functionality is present
        4. For historical issues, use the fixed_log references to understand how to fix them
        
        Return the improved code as pure Python code. Do not include any explanation or markdown formatting.
        """
        
        # Call LLM to improve code
        llm_response = self._call_llm(prompt)
        
        # Extract improved code
        improved_code = self._extract_code(llm_response)
        # Remove any leftover markdown fences
        improved_code = self._strip_markdown_fences(improved_code)
        
        self.logger.info("Code improved based on self-checking issues")
        return improved_code
    
    def _fix_syntax(self, code: str, error: SyntaxError) -> str:
        """
        Fix syntax errors in code.
        
        Args:
            code: The code to fix
            error: The syntax error
            
        Returns:
            Fixed code
        """
        self.logger.warning(f"Fixing syntax error: {error}")
        
        # Build prompt for fixing syntax
        prompt = f"""
        The following Python code has a syntax error. Please provide a corrected version of the code.
        
        Error: {error}
        
        Original code:
        ```python
        {code}
        ```
        
        Return only the corrected code. Do not include any explanation or markdown formatting.
        """
        
        # Call LLM to fix syntax
        llm_response = self._call_llm(prompt)
        
        # Extract fixed code
        fixed_code = self._extract_code(llm_response)
        # Remove any leftover markdown fences
        fixed_code = self._strip_markdown_fences(fixed_code)
        # Apply local docstring and entry-point fixes
        fixed_code = self._fix_unclosed_docstrings(fixed_code)
        fixed_code = self._ensure_entry_point(fixed_code)
        
        self.logger.info("Syntax fixed")
        return fixed_code
    
    def _build_prompt(
        self,
        task_spec: Dict[str, Any],
        model_plan: Optional[Dict[str, Any]] = None,
        data_analysis: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None,
        previous_code: Optional[Dict[str, str]] = None,
        mode: str = "full"
    ) -> str:
        """
        Build a prompt for the LLM to generate code.
        
        Args:
            task_spec: Task specification from the Task Understanding Agent
            model_plan: Model plan from the Model Planning Agent (optional)
            data_analysis: Data analysis results from the Data Analysis Agent (optional)
            feedback: Feedback from previous iterations (optional)
            data_path: Original data directory path (optional)
            previous_code: Code from the previous iteration for context (optional)
            mode: Workflow mode ('lite', 'medium', 'full'). Defaults to 'full'.
            
        Returns:
            A prompt for the LLM to generate code
        """
        # Load the appropriate code generation prompt template based on mode
        if mode == "lite":
            template_name = "code_generation_prompt_lite.txt"
        elif mode == "medium":
            template_name = "code_generation_prompt_medium.txt"
        else:  # full
            template_name = "code_generation_prompt.txt"
        prompt_template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "templates",
            template_name
        )
        
        try:
            with open(prompt_template_path, 'r') as f:
                prompt_template = f.read()
        except Exception as e:
            self.logger.error(f"Error loading code generation prompt template: {e}")
            # Fallback to a basic prompt
            prompt_template = """
            You are a code generation agent. Your task is to generate simulation code based on the following:
            
            Task Specification:
            {task_spec}
            
            Model Plan:
            {model_plan}
            
            Data Analysis:
            {data_analysis}
            
            Feedback:
            {feedback}
            
            Previous Code:
            {previous_code}
            
            Data Path:
            {data_path}
            
            Please generate Python code that implements the specified simulation model.
            """
        
        if mode == "lite":
            # Format for lite template (uses fewer placeholders)
            task_spec_str = json.dumps(task_spec, indent=2) if task_spec else "No task specification provided"
            
            # Format the previous code as a string for the prompt
            previous_code_str = ""
            if previous_code:
                if isinstance(previous_code, dict):
                    for filename, code in previous_code.items():
                        previous_code_str += f"File: {filename}\n```python\n{code}\n```\n\n"
                elif isinstance(previous_code, str):
                    previous_code_str = f"```python\n{previous_code}\n```\n\n"
            if not previous_code_str:
                previous_code_str = "No previous code available"
            
            # Format the feedback as a string for the prompt
            feedback_str = json.dumps(feedback, indent=2) if feedback else "No feedback provided"
            
            # Fill in the lite template
            prompt = prompt_template.format(
                task_spec=task_spec_str,
                feedback=feedback_str,
                previous_code=previous_code_str
            )
        else:
            # Format for full template (uses all placeholders)
            task_spec_str = json.dumps(task_spec, indent=2) if task_spec else "No task specification provided"
            model_plan_str = json.dumps(model_plan, indent=2) if model_plan else "No model plan provided"
            data_analysis_str = json.dumps(data_analysis, indent=2) if data_analysis else "No data analysis provided"
            
            # Format the previous code as a string for the prompt
            previous_code_str = ""
            if previous_code:
                if isinstance(previous_code, dict):
                    for filename, code in previous_code.items():
                        previous_code_str += f"File: {filename}\n```python\n{code}\n```\n\n"
                elif isinstance(previous_code, str):
                    previous_code_str = f"```python\n{previous_code}\n```\n\n"
            
            # Format the feedback as a string for the prompt
            feedback_str = json.dumps(feedback, indent=2) if feedback else "No feedback provided"
            
            # Data path string
            data_path_str = f"Data directory: {data_path}" if data_path else "No data path provided"
            
            # Fill in the full template
            prompt = prompt_template.format(
                task_spec=task_spec_str,
                model_plan=model_plan_str,
                data_analysis=data_analysis_str,
                feedback=feedback_str,
                previous_code=previous_code_str,
                data_path=data_path_str
            )
        
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from the LLM response.
        """
        # Look for code blocks marked with ```python and ```
        code_start = response.find("```python")
        if code_start >= 0:
            code_start += len("```python")
            code_end = response.find("```", code_start)
            if code_end >= 0:
                extracted_code = response[code_start:code_end].strip()
                return self._ensure_entry_point(extracted_code)
        
        # If no Python code blocks found, look for generic code blocks
        code_start = response.find("```")
        if code_start >= 0:
            code_start += len("```")
            code_end = response.find("```")
            if code_end >= 0:
                extracted_code = response[code_start:code_end].strip()
                return self._ensure_entry_point(extracted_code)
        
        # If no code blocks found, assume the entire response is code
        # This is the expected behavior with the updated prompt
        return self._ensure_entry_point(response)
    
    def _ensure_entry_point(self, code: str) -> str:
        """
        Ensure the code has a proper entry point.
        
        The entry point should be a main() function and a direct call to main(). This is
        required for the code to run when executed directly or within the sandbox.
        
        Args:
            code: The generated code
        
        Returns:
            Code with entry point added if missing
        """
        has_main = "def main(" in code
        has_entry = "if __name__ == '__main__':" in code or "if __name__ == \"__main__\":" in code
        
        # Check for direct main call
        direct_main_call = "main()" in code.splitlines()
        
        if not has_main:
            self.logger.warning("Generated code lacks main() function; inserting stub.")
            code = "def main():\n    pass\n\n" + code
        
        # Remove any if __name__ == "__main__" guard if present
        if has_entry:
            self.logger.warning("Generated code has __main__ guard; removing and inserting direct main call.")
            code_lines = code.splitlines()
            filtered_lines = []
            skip_main_guard = False
            for line in code_lines:
                if "if __name__ == \"__main__\":" in line or "if __name__ == '__main__':" in line:
                    skip_main_guard = True
                    continue
                if skip_main_guard and "main()" in line and line.strip().startswith("main()"):
                    skip_main_guard = False
                    continue
                if skip_main_guard and not line.strip():
                    continue
                if skip_main_guard and line.startswith(" "):
                    continue
                filtered_lines.append(line)
            code = "\n".join(filtered_lines)
        
        # Add direct main call if not present
        if not direct_main_call or has_entry:
            self.logger.warning("Generated code lacks direct main() call; inserting call at end of file.")
            code += "\n\n# Execute main for both direct execution and sandbox wrapper invocation\nmain()"
        return code
    
    def _strip_markdown_fences(self, code: str) -> str:
        """
        Remove any remaining markdown code fence markers (``` or ```python) to avoid syntax errors.
        """
        # Remove all lines containing any triple backticks
        lines = code.splitlines()
        cleaned = [line for line in lines if '```' not in line]
        return '\n'.join(cleaned)
    
    def _fix_unclosed_docstrings(self, code: str) -> str:
        """
        Detects unbalanced triple-quoted strings and appends closing quotes if needed.
        """
        # Fix unbalanced triple double-quotes
        dd = code.count('"""')
        if dd % 2 != 0:
            self.logger.warning("Unbalanced triple-double-quotes detected. Appending closing triple-quote.")
            code += '\n"""'
        # Fix unbalanced triple single-quotes
        ss = code.count("'''")
        if ss % 2 != 0:
            self.logger.warning("Unbalanced triple-single-quotes detected. Appending closing triple-quote.")
            code += "\n'''"
        return code
    
    def _generate_code_summary(self, code: str) -> str:
        """
        Generate a summary of the generated code.
        
        Args:
            code: The generated code
        
        Returns:
            A summary of the code
        """
        # Count lines of code
        lines = code.split("\n")
        num_lines = len(lines)
        
        # Count classes and functions
        num_classes = sum(1 for line in lines if line.strip().startswith("class "))
        num_functions = sum(1 for line in lines if line.strip().startswith("def "))
        
        # Generate a simple summary
        summary = f"Generated {num_lines} lines of code containing {num_classes} classes and {num_functions} functions."
        
        return summary
    
    def _generate_default_code(self, model_plan: Dict[str, Any]) -> str:
        """
        Generate default code based on the model plan.
        
        Args:
            model_plan: The model plan
        
        Returns:
            Default code implementation
        """
        model_type = model_plan.get("model_type", "agent_based")
        entities = model_plan.get("entities", [])
        behaviors = model_plan.get("behaviors", [])
        interactions = model_plan.get("interactions", [])
        
        # Generate imports
        code = """#!/usr/bin/env python3
# Generated Simulation Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
from typing import Dict, List, Any, Tuple, Optional
"""
        
        # Generate entity classes
        code += "\n\n# Entity Classes\n"
        for entity in entities:
            entity_name = entity.get("name", "Entity")
            attributes = entity.get("attributes", [])
            
            code += f"class {entity_name}:\n"
            code += f"    def __init__(self, entity_id: str):\n"
            code += f"        self.id = entity_id\n"
            
            # Add attributes
            for attr in attributes:
                code += f"        self.{attr} = None\n"
            
            # Add methods
            code += "\n    def get_state(self) -> Dict[str, Any]:\n"
            code += "        return {\n"
            code += "            'id': self.id,\n"
            for attr in attributes:
                code += f"            '{attr}': self.{attr},\n"
            code += "        }\n"
            
            # Add behavior methods
            entity_behaviors = [b for b in behaviors if entity_name in b.get("applicable_to", [])]
            for behavior in entity_behaviors:
                behavior_name = behavior.get("name", "behave")
                code += f"\n    def {behavior_name}(self, environment):\n"
                code += f"        # Implement {behavior_name} behavior\n"
                code += f"        pass\n"
            
            code += "\n\n"
        
        # Generate environment class
        code += "# Environment Class\n"
        code += "class Environment:\n"
        code += "    def __init__(self, config: Dict[str, Any]):\n"
        code += "        self.config = config\n"
        code += "        self.entities = {}\n"
        code += "        self.time = 0.0\n"
        code += "        self.metrics = {}\n"
        
        # Add methods
        code += "\n    def add_entity(self, entity):\n"
        code += "        self.entities[entity.id] = entity\n"
        
        code += "\n    def remove_entity(self, entity_id: str):\n"
        code += "        if entity_id in self.entities:\n"
        code += "            del self.entities[entity_id]\n"
        
        code += "\n    def get_entity(self, entity_id: str):\n"
        code += "        return self.entities.get(entity_id)\n"
        
        code += "\n    def get_all_entities(self):\n"
        code += "        return list(self.entities.values())\n"
        
        code += "\n    def step(self, time_step: float = 1.0):\n"
        code += "        # Update all entities\n"
        code += "        for entity in self.entities.values():\n"
        
        # Call behavior methods for each entity type
        for entity in entities:
            entity_name = entity.get("name", "Entity")
            entity_behaviors = [b for b in behaviors if entity_name in b.get("applicable_to", [])]
            
            if entity_behaviors:
                code += f"            if isinstance(entity, {entity_name}):\n"
                for behavior in entity_behaviors:
                    behavior_name = behavior.get("name", "behave")
                    code += f"                entity.{behavior_name}(self)\n"
        
        code += "\n        # Process interactions\n"
        
        # Add interaction processing
        for interaction in interactions:
            interaction_name = interaction.get("name", "interaction")
            entities_involved = interaction.get("entities_involved", [])
            
            if len(entities_involved) >= 2:
                code += f"        # Process {interaction_name}\n"
                code += f"        self._process_{interaction_name}()\n"
        
        code += "\n        # Update time\n"
        code += "        self.time += time_step\n"
        
        code += "\n        # Return metrics for this step\n"
        code += "        return self.metrics\n"
        
        # Add interaction methods
        for interaction in interactions:
            interaction_name = interaction.get("name", "interaction")
            code += f"\n    def _process_{interaction_name}(self):\n"
            code += f"        # Implement {interaction_name} interaction\n"
            code += f"        pass\n"
        
        # Generate simulation class
        code += "\n\n# Simulation Class\n"
        code += "class Simulation:\n"
        code += "    def __init__(self, config: Dict[str, Any]):\n"
        code += "        self.config = config\n"
        code += "        self.environment = Environment(config)\n"
        code += "        self.results = {\n"
        code += "            'config': config,\n"
        code += "            'metrics': {},\n"
        code += "            'time_series': []\n"
        code += "        }\n"
        
        # Add initialization method
        code += "\n    def initialize(self):\n"
        code += "        # Create initial entities\n"
        
        # Initialize each entity type
        for entity in entities:
            entity_name = entity.get("name", "Entity")
            code += f"        # Create {entity_name} entities\n"
            code += f"        for i in range(self.config.get('num_{entity_name.lower()}s', 10)):\n"
            code += f"            entity = {entity_name}(f'{entity_name.lower()}_{i}')\n"
            
            # Initialize attributes
            for attr in entity.get("attributes", []):
                code += f"            entity.{attr} = random.random()  # Initialize with random value\n"
            
            code += f"            self.environment.add_entity(entity)\n"
        
        # Add run method
        code += "\n    def run(self, steps: int = 100):\n"
        code += "        # Initialize the simulation\n"
        code += "        self.initialize()\n"
        code += "\n        # Run the simulation for the specified number of steps\n"
        code += "        for step in range(steps):\n"
        code += "            # Execute one step of the simulation\n"
        code += "            metrics = self.environment.step()\n"
        code += "            \n"
        code += "            # Record the results\n"
        code += "            self.results['time_series'].append({\n"
        code += "                'step': step,\n"
        code += "                'time': self.environment.time,\n"
        code += "                'metrics': metrics\n"
        code += "            })\n"
        code += "\n        # Compile final metrics\n"
        code += "        self.results['metrics'] = self.environment.metrics\n"
        code += "        \n"
        code += "        return self.results\n"
        
        # Add visualization method
        code += "\n    def visualize(self):\n"
        code += "        # Create visualizations of the simulation results\n"
        code += "        plt.figure(figsize=(10, 6))\n"
        code += "        \n"
        code += "        # Example: Plot a metric over time\n"
        code += "        if self.results['time_series']:\n"
        code += "            time_points = [entry['time'] for entry in self.results['time_series']]\n"
        code += "            \n"
        code += "            # Plot each available metric\n"
        code += "            for metric_name in self.environment.metrics:\n"
        code += "                if metric_name in self.results['time_series'][0]['metrics']:\n"
        code += "                    metric_values = [entry['metrics'].get(metric_name, 0) for entry in self.results['time_series']]\n"
        code += "                    plt.plot(time_points, metric_values, label=metric_name)\n"
        code += "            \n"
        code += "            plt.xlabel('Time')\n"
        code += "            plt.ylabel('Value')\n"
        code += "            plt.title('Simulation Metrics Over Time')\n"
        code += "            plt.legend()\n"
        code += "            plt.grid(True)\n"
        code += "        \n"
        code += "        plt.tight_layout()\n"
        code += "        plt.savefig('simulation_results.png')\n"
        code += "        plt.show()\n"
        
        # Add save method
        code += "\n    def save_results(self, filename: str = 'simulation_results.json'):\n"
        code += "        # Save the simulation results to a file\n"
        code += "        with open(filename, 'w') as f:\n"
        code += "            json.dump(self.results, f, indent=2)\n"
        
        # Add main function
        code += "\n\n# Main Function\n"
        code += "def main():\n"
        code += "    # Configuration\n"
        code += "    config = {\n"
        
        # Add parameters from model plan
        params = model_plan.get("parameters", {})
        for param_name, param_value in params.items():
            code += f"        '{param_name}': {param_value},\n"
        
        # Add additional configuration
        if "population_size" in model_plan.get("initialization", {}):
            pop_size = model_plan["initialization"]["population_size"]
            for entity in entities:
                entity_name = entity.get("name", "Entity")
                code += f"        'num_{entity_name.lower()}s': {pop_size // len(entities)},\n"
        
        code += "    }\n"
        code += "\n    # Create and run the simulation\n"
        code += "    simulation = Simulation(config)\n"
        code += "    results = simulation.run(steps=100)\n"
        code += "\n    # Visualize and save the results\n"
        code += "    simulation.visualize()\n"
        code += "    simulation.save_results()\n"
        
        # Add script entry point
        code += "\n\nif __name__ == '__main__':\n"
        code += "    main()\n"
        
        return code 
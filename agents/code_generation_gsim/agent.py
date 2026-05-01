"""
CodeGenerationAgent: Generates simulation code based on the model plan.
"""
# todo self-loop code check and code improve, not using model plan
# todo reasoning - medium
import logging
import os
import json
import ast
import re
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
        mode: str = "full",
        selfloop: int = 3,
        blueprint: Optional[Any] = None,
        output_dir: Optional[str] = None,
        iteration: Optional[int] = None,
        playbook: Optional[Dict[str, Any]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
        best_simulator_info: Optional[Dict[str, Any]] = None,
        simulation_info_history: Optional[List[Dict[str, Any]]] = None,
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
            selfloop: Number of self-checking loop attempts
            blueprint: Blueprint object for blueprint mode (optional)
            output_dir: Output directory for saving intermediate code versions (optional)
            iteration: Current iteration number for file naming (optional)
        
        Returns:
            Dictionary containing the generated code and metadata
        """
        self.logger.info("Generating simulation code")
        
        # Log blueprint / playbook usage if available
        if blueprint is not None:
            self.logger.info("Using blueprint for code generation in blueprint mode")
            self.logger.debug(f"Blueprint contains {len(blueprint)} items")
        if playbook is not None:
            self.logger.info("Playbook provided to code generation (ACE/ODD mode)")
        
        # # Override model_plan data_sources with processed file paths (skip in lite mode)
        # if mode != "lite" and model_plan and data_analysis and "file_references" in data_analysis:
        #     self.logger.info("Overriding model_plan data_sources with processed file paths")
        #     # Copy model_plan to avoid mutating original
        #     model_plan = dict(model_plan)
        #     new_sources = []
        #     for ds in model_plan.get("data_sources", []):
        #         name = ds.get("name")
        #         # If processed path exists, include it
        #         if name in data_analysis["file_references"]:
        #             ds["path"] = data_analysis["file_references"][name]
        #         new_sources.append(ds)
        #     model_plan["data_sources"] = new_sources
        
        # Build prompt from template, including original data path and blueprint
        prompt_args = {
            "task_spec": task_spec,
            "model_plan": model_plan,
            "data_analysis": data_analysis,
            "feedback": feedback,
            "data_path": data_path,
            "previous_code": previous_code,
            "mode": mode,
            "playbook": playbook,
            "simulation_results": simulation_results,
            "iteration": iteration,
        }
        
        # Use patch prompt for iteration >= 1 (second iteration and beyond)
        # For ACE mode: use patch prompt
        # For ALPHA mode: use patch prompt if simulation_info_history is available
        if iteration is not None and iteration >= 1:
            if mode == "ace":
                self.logger.info(f"Using patch prompt for iteration {iteration} (ACE mode)")
                prompt = self._build_patch_prompt(
                    task_spec=task_spec,
                    previous_code=previous_code,
                    simulation_results=simulation_results,
                    playbook=playbook,
                )
            elif mode == "alpha" and simulation_info_history is not None:
                self.logger.info(f"Using patch prompt for iteration {iteration} (ALPHA mode with simulation_info_history)")
                prompt = self._build_patch_prompt(
                    task_spec=task_spec,
                    previous_code=previous_code,
                    simulation_results=simulation_results,
                    playbook=playbook,
                    best_simulator_info=best_simulator_info,
                    simulation_info_history=simulation_info_history,
                    iteration=iteration,
                )
            else:
                prompt = self._build_prompt(**prompt_args)
        else:
            prompt = self._build_prompt(**prompt_args)

        # Call LLM to generate code
        # Use medium effort for initial generation to reduce timeout risk
        # Self-loop will improve the code quality in subsequent iterations
        llm_response = self._call_llm(prompt, reasoning={"effort": "medium"})
        
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
        
        # Ensure model_plan is a dictionary (lite mode may pass None)
        if model_plan is None:
            model_plan = {}
        
        # Run self-checking loop to improve the code. Previously this was skipped in lite mode, but
        # we now enable it to keep consistency across modes while still keeping the workflow lightweight
        # by avoiding expensive simulation execution.
        self.logger.info("Starting self-checking loop for code improvement (mode=%s)", mode)
        code = self._run_self_checking_loop(
            code=code,
            task_spec=task_spec,
            model_plan=model_plan,
            feedback=feedback,
            historical_fix_log=historical_fix_log,
            max_attempts=selfloop,
            mode=mode,
            output_dir=output_dir,
            iteration=iteration
        )

        # Generate simulator description once per iteration (not inside self-loop)
        simulator_description = ""
        try:
            simulator_description = self._generate_simulator_description(code, task_spec)
        except Exception as e:
            self.logger.warning(f"Failed to generate simulator_description: {e}")
        
        # Generate a summary of the code
        code_summary = self._generate_code_summary(code)
        
        result = {
            "code": code,
            "code_summary": code_summary,
            "simulator_description": simulator_description,
            "metadata": {
                "model_type": model_plan.get("model_type", mode) if model_plan else mode,
                "entities": [e.get("name") for e in model_plan.get("entities", [])] if model_plan else [],
                "behaviors": [b.get("name") for b in model_plan.get("behaviors", [])] if model_plan else [],
                "mode": mode
            }
        }
        
        self.logger.info("Code generation completed")
        # Note: Syntax checking is already handled in _run_self_checking_loop
        # No need for additional compile check here to avoid redundancy
        
        # Update blueprint if available
        if blueprint is not None:
            self._update_blueprint_from_generated_code(blueprint, result, task_spec)
            
        return result
    
    def _run_self_checking_loop(
        self,
        code: str,
        task_spec: Dict[str, Any],
        model_plan: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None,
        historical_fix_log: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
        mode: str = "full",
        output_dir: Optional[str] = None,
        iteration: Optional[int] = None
    ) -> str:
        """
        Run a self-checking loop to improve the generated code.
        
        This implements a "Low-Level Code Inspector" (Linter/Sanitizer) with three steps:
        1. Step 1 (Regex): Automatically strip Markdown markers using Python regex
        2. Step 2 (AST): Check syntax errors using Python ast.parse() (free, 100% accurate)
        3. Step 3 (LLM Linter): Check code implementation issues using LLM
        
        If issues are found, the code is improved and the checks are run again.
        This process is repeated up to three times.
        
        Args:
            code: The generated code
            task_spec: Task specification from the Task Understanding Agent
            model_plan: Model plan from the Model Planning Agent
            feedback: Feedback from previous iterations (optional)
            historical_fix_log: Log of historical issues and their fix status (optional)
            max_attempts: Number of self-checking loop attempts
            mode: Workflow mode ("full", "odd", etc.)
            output_dir: Output directory for saving intermediate code versions (optional)
            iteration: Current iteration number for file naming (optional)
            
        Returns:
            Improved code after self-checking loop
        """
        improved_code = code
        if max_attempts <= 0:
            self.logger.info("Self-checking loop disabled (max_attempts <= 0)")
            return improved_code
        
        # Track best code to prevent catastrophic degradation
        best_code = improved_code
        best_issues_count = float('inf')  # Start with infinity
        best_iteration = -1
        
        for attempt in range(max_attempts):
            self.logger.info(f"Self-checking loop - Attempt {attempt + 1}/{max_attempts}")
            
            # Step 1: Strip Markdown fences (programmatic)
            improved_code = self._strip_markdown_fences(improved_code)
            self.logger.info("Step 1: Markdown fences stripped")
            
            # Step 2: AST syntax check (programmatic, free and 100% accurate)
            ast_issues = []
            try:
                ast.parse(improved_code)
                self.logger.info("Step 2: AST syntax check passed")
            except SyntaxError as err:
                self.logger.warning(f"Step 2: AST syntax error detected: {err}")
                ast_issues.append({
                    "type": "SYNTAX_ERROR",
                    "severity": "critical",
                    "description": f"Syntax error at line {err.lineno}: {err.msg}",
                    "location": f"Line {err.lineno}",
                    "recommendation": f"Fix syntax error: {err.msg}"
                })
            
            # Step 3: LLM Linter check (high-level issues)
            llm_issues = self._perform_code_quality_check(improved_code, task_spec, model_plan, mode)
            
            # Merge all issues
            issues = ast_issues + llm_issues
            
            # If no issues found, we're done
            if not issues:
                self.logger.info(f"Self-checking passed on attempt {attempt + 1}")
                break
            
            # Log issues found
            self.logger.info(f"Found {len(issues)} issues in self-checking ({len(ast_issues)} AST, {len(llm_issues)} LLM). Attempting to improve code.")
            
            # Count issues before improvement for comparison
            critical_issues_before = [issue for issue in issues if issue.get("severity") == "critical"]
            total_issues_before = len(issues)
            critical_issues_count_before = len(critical_issues_before)
            
            # Initialize best_issues_count on first iteration if needed
            if attempt == 0 and best_issues_count == float('inf'):
                best_issues_count = critical_issues_count_before
                self.logger.info(f"Initial code has {critical_issues_count_before} critical issues")
            
            # Improve the code based on issues
            improved_code = self._improve_code_based_on_issues(
                code=improved_code,
                issues=issues,
                task_spec=task_spec,
                model_plan=model_plan,
                mode=mode
            )
            
            # Post-improvement cleanup
            improved_code = self._strip_markdown_fences(improved_code)
            improved_code = self._fix_unclosed_docstrings(improved_code)
            
            # Check if the improved code contains timeout error messages
            timeout_error_patterns = [
                "Error: Request timed out",
                "Request timed out",
                "Error calling OpenAI API: Request timed out"
            ]
            has_timeout_error = any(pattern in improved_code for pattern in timeout_error_patterns)
            
            # Check for syntax errors after improvement
            has_syntax_error = False
            try:
                ast.parse(improved_code)
                self.logger.info("Improved code passed AST syntax check")
            except SyntaxError as err:
                has_syntax_error = True
                self.logger.warning(f"Syntax error in improved code: {err}")
                # If this is the last attempt, try to fix syntax
                if attempt == max_attempts - 1:
                    improved_code = self._fix_syntax(improved_code, err)
            
            # Save intermediate code to file (as iter{iteration}_loop{attempt})
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    if iteration is not None:
                        loop_code_path = os.path.join(output_dir, f"simulation_code_iter{iteration}_loop{attempt}.py")
                    else:
                        loop_code_path = os.path.join(output_dir, f"simulation_code_iter_loop{attempt}.py")
                    with open(loop_code_path, 'w', encoding='utf-8') as f:
                        f.write(improved_code)
                    self.logger.info(f"Saved self-loop iteration {attempt} code to {loop_code_path}")
                except Exception as e:
                    self.logger.error(f"Error saving intermediate code: {e}")
            
            # Re-check code quality after improvement to compare
            # Run AST check again
            ast_issues_after = []
            try:
                ast.parse(improved_code)
            except SyntaxError as err:
                ast_issues_after.append({
                    "type": "SYNTAX_ERROR",
                    "severity": "critical",
                    "description": f"Syntax error at line {err.lineno}: {err.msg}",
                    "location": f"Line {err.lineno}",
                    "recommendation": f"Fix syntax error: {err.msg}"
                })
            
            # Run LLM Linter check again
            llm_issues_after = self._perform_code_quality_check(improved_code, task_spec, model_plan, mode)
            
            # Merge all issues
            issues_after = ast_issues_after + llm_issues_after
            critical_issues_after = [issue for issue in issues_after if issue.get("severity") == "critical"]
            total_issues_after = len(issues_after)
            critical_issues_count_after = len(critical_issues_after)
            
            # Detect catastrophic degradation using multiple signals
            is_degraded = self._detect_code_degradation(
                code=improved_code,
                has_syntax_error=has_syntax_error,
                has_timeout_error=has_timeout_error,
                current_critical_issues=critical_issues_count_after,
                previous_critical_issues=critical_issues_count_before,
                current_total_issues=total_issues_after,
                previous_total_issues=total_issues_before
            )
            
            if is_degraded:
                self.logger.warning(f"Iteration {attempt}: Code degradation detected, reverting to best code for next iteration")
                # Revert to best code for next iteration
                improved_code = best_code
            else:
                # Update best code if quality improved or maintained (prioritize later versions)
                # If issues decreased: clear improvement
                # If issues same but not degraded: prioritize later version (may have other improvements)
                if critical_issues_count_after < best_issues_count:
                    self.logger.info(f"Iteration {attempt}: Code quality improved ({best_issues_count} -> {critical_issues_count_after} critical issues)")
                    best_code = improved_code
                    best_issues_count = critical_issues_count_after
                    best_iteration = attempt
                elif critical_issues_count_after == best_issues_count:
                    # Issues count same, but prefer later version (may have other improvements like new features, better structure, non-critical fixes)
                    self.logger.info(f"Iteration {attempt}: Code quality maintained ({critical_issues_count_after} critical issues), updating to latest version (may contain additional improvements)")
                    best_code = improved_code
                    best_issues_count = critical_issues_count_after
                    best_iteration = attempt
                else:
                    # Issues increased (but not detected as degraded by _detect_code_degradation)
                    # This should rarely happen, but keep the best version
                    self.logger.warning(f"Iteration {attempt}: Code quality worsened ({best_issues_count} -> {critical_issues_count_after} critical issues), keeping previous best version")
            
            # If this is the last attempt, log a warning
            if attempt == max_attempts - 1 and issues:
                self.logger.warning("Maximum self-checking attempts reached but issues remain")
        
        # Return best code instead of last iteration code
        if best_iteration >= 0:
            self.logger.info(f"Returning best code from iteration {best_iteration} with {best_issues_count} critical issues")
        else:
            self.logger.info("Returning original/initial code (no improvements made)")
        
        return best_code
    
    def _detect_code_degradation(
        self,
        code: str,
        has_syntax_error: bool,
        has_timeout_error: bool,
        current_critical_issues: int,
        previous_critical_issues: int,
        current_total_issues: int,
        previous_total_issues: int
    ) -> bool:
        """
        Detect if code has catastrophically degraded (e.g., due to API timeout).
        
        Uses multiple signals:
        1. Timeout error messages in code (from LLM response)
        2. Code quality regression (more issues than before)
        3. Code structure degradation (too short, syntax errors, etc.)
        
        Args:
            code: The code to check
            has_syntax_error: Whether the code has syntax errors
            has_timeout_error: Whether the code contains timeout error messages
            current_critical_issues: Number of critical issues in current code
            previous_critical_issues: Number of critical issues before improvement
            current_total_issues: Total number of issues in current code
            previous_total_issues: Total number of issues before improvement
            
        Returns:
            True if code is degraded, False otherwise
        """
        # Check 1: Timeout error in code (highest priority - direct indicator of API failure)
        if has_timeout_error:
            self.logger.warning("Code degradation: Contains timeout error message from LLM API")
            return True
        
        # Check 2: Critical issues increased significantly (more than 10% increase)
        if previous_critical_issues > 0:
            critical_issues_increase = (current_critical_issues - previous_critical_issues) / previous_critical_issues
            if critical_issues_increase > 0.1:  # More than 10% increase
                self.logger.warning(
                    f"Code degradation: Critical issues increased significantly "
                    f"({previous_critical_issues} -> {current_critical_issues}, "
                    f"{critical_issues_increase*100:.1f}% increase)"
                )
                return True
        
        # Check 3: Total issues increased significantly (more than 10% increase)
        if previous_total_issues > 0:
            total_issues_increase = (current_total_issues - previous_total_issues) / previous_total_issues
            if total_issues_increase > 0.1:  # More than 10% increase
                self.logger.warning(
                    f"Code degradation: Total issues increased significantly "
                    f"({previous_total_issues} -> {current_total_issues}, "
                    f"{total_issues_increase*100:.1f}% increase)"
                )
                return True
        
        # Check 4: Code is suspiciously short (< 200 characters)
        if len(code) < 200:
            self.logger.warning(f"Code degradation: Code too short ({len(code)} chars)")
            return True
        
        # Check 5: Contains error messages from API timeout (fallback check)
        error_patterns = [
            "Error: Request timed out",
            "Error calling OpenAI API",
            "Request timed out",
            "failed to generate"
        ]
        code_lower = code.lower()
        for pattern in error_patterns:
            if pattern.lower() in code_lower:
                self.logger.warning(f"Code degradation: Contains error message '{pattern}'")
                return True
        
        # Check 6: Only has empty main() function
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        if len(lines) <= 3:  # e.g., "def main():", "pass", "main()"
            self.logger.warning("Code degradation: Code has too few non-comment lines")
            return True
        
        # Check 7: Has syntax error
        if has_syntax_error:
            self.logger.warning("Code degradation: Code has syntax errors")
            return True
        
        return False
    
    def _perform_code_quality_check(
        self,
        code: str,
        task_spec: Dict[str, Any],
        model_plan: Dict[str, Any],
        mode: str = "full"
    ) -> List[Dict[str, Any]]:
        """
        Perform LLM Linter check focusing on high-level issues.
        
        This is Step 3 of the self-checking loop (after Regex and AST checks).
        It acts as a "low-level code inspector" (Linter/Sanitizer), NOT a QA.
        
        Goal: Ensure the code looks like complete, legal Python code before running.
        
        High-level issues checked:
        1. Markdown stripping detection (residual markdown artifacts)
        2. Truncation detection (incomplete code, unclosed brackets)
        3. Lazy coding / Placeholder detection (# ..., TODO, pass-only functions)
        4. Hallucinated imports/attributes (non-existent libraries or functions)
        5. Namespace & scope issues (undefined variables, mismatched parameters)
        6. Execution entry point check (missing or empty main block)
        7. Hazardous patterns (infinite loops, dangerous file operations)
        
        Args:
            code: The code to check (already passed Regex and AST checks)
            task_spec: Task specification
            model_plan: Model plan
            mode: Workflow mode
            
        Returns:
            List of high-level issues found
        """
        self.logger.info("Step 3: Performing LLM Linter check (high-level issues)")

        if mode in ("odd", "persona", "ace"):
            # Extract blueprint from task_spec (excluding file_summaries)
            if "data_analysis_result" in task_spec:
                blueprint = {
                    k: v
                    for k, v in task_spec["data_analysis_result"].items()
                    if k != "file_summaries"
                }
                task_info = json.dumps(blueprint, indent=2)
                self.logger.info(
                    "Extracted blueprint from task_spec for %s mode", mode
                )
            else:
                task_info = json.dumps(task_spec, indent=2)
            
            # Check task description and load appropriate patch
            task_description = task_spec.get('description', '').lower()
            
            if 'mask-wearing' in task_description:
                self.logger.info("Loading mask adoption patch for code quality check")
                try:
                    # Get project root directory (3 levels up from agents/code_generation_ace/agent.py)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "mask_adoption_patch.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        patch_content = f"\n\n{f.read()}"
                    task_info += patch_content
                except Exception as e:
                    self.logger.error(f"Error loading mask_adoption_patch.txt: {e}")
            
            elif 'user rates' in task_description or 'daily mobility trajectories' in task_description:
                self.logger.info("Loading LLM calling patch for code quality check")
                try:
                    # Get project root directory (3 levels up from agents/code_generation_ace/agent.py)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "llm_api_call_patch_prompt.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        patch_content = f"\n\n{f.read()}"
                    task_info += patch_content
                except Exception as e:
                    self.logger.error(f"Error loading llm_api_call_patch_prompt.txt: {e}")
            
            # Persona-specific patch for psychometric test simulators
            if mode == "persona" and 'psychometric tests' in task_description:
                self.logger.info("Loading persona patch for code quality check")
                try:
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "persona_patch.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        patch_content = f"\n\n{f.read()}"
                    task_info += patch_content
                except Exception as e:
                    self.logger.error(f"Error loading persona_patch.txt: {e}")
        else:
            # For other modes, use standard format
            task_info = json.dumps(task_spec, indent=2)
        
        # Build prompt for LLM Linter (high-level issues)
        prompt = f"""
        You are a Python Code Linter (Low-Level Inspector). Your role is to ensure the code is COMPLETE and LEGAL Python before it runs.
        
        You are NOT a QA. You do NOT check algorithm correctness or logic completeness.
        Your ONLY job: catch issues that will cause the code to fail at parse time or early runtime.
        
        Generated code (already passed AST syntax check):
        ```python
        {code}
        ```
        
        Perform the following checks:
        
        1. MARKDOWN RESIDUE
        Check if code contains residual markdown artifacts:
        - Text like "Here is the code:" or "Hope this helps"
        - Incomplete code fence markers
        - Natural language explanations mixed with code
        
        2. TRUNCATION DETECTION (CRITICAL!)
        Check if code is incomplete or truncated:
        - Last line ends abruptly (e.g., "def function_name(" with no body)
        - Unclosed brackets, parentheses, or quotes
        - Class or function definitions without bodies
        - Missing return statements in non-void functions
        
        3. LAZY CODING / PLACEHOLDERS (MOST IMPORTANT!)
        Check for any form of "laziness" or unimplemented code:
        - Comments like "# ... rest of the code", "# ... implement logic here", "# TODO: fill this part"
        - Functions with only "pass" and no implementation
        - Ellipsis (...) used as placeholder
        - Any form of "省略" or "omitted for brevity"
        **ZERO TOLERANCE**: Any placeholder means CRITICAL issue!
        
        4. HALLUCINATED IMPORTS/ATTRIBUTES
        Check for non-existent imports or functions:
        - Imported libraries that don't exist (except: numpy, pandas, scipy, matplotlib, sklearn, networkx, json, os, sys, math, random, collections, itertools, functools, datetime, typing)
        - Function calls on libraries that don't have those methods (e.g., numpy.calculate_infection_rate())
        - References to undefined global variables or classes
        - Using attributes that don't exist on standard objects
        
        5. NAMESPACE & SCOPE ISSUES
        Check for scope and definition issues:
        - Variables used before being defined
        - Function calls with mismatched argument counts
        - Circular references or variable shadowing
        - Methods called on objects that don't have those methods
        
        6. EXECUTION ENTRY POINT (IMPORTANT!)
        The code should have a proper entry point:
        - There should be a main() function
        - At the end of file, there should be a direct call: main() (NOT if __name__ == "__main__")
        - The main() function should NOT be empty or only contain pass
        - The main() should orchestrate the actual workflow
        
        7. HAZARDOUS PATTERNS
        Check for dangerous patterns:
        - while True: loops without clear break conditions
        - Hardcoded absolute file paths (should use os.path.join with environment variables)
        - File operations without proper error handling
        - Division operations without checking for zero
        
        Return a JSON array of issues. Each issue MUST have:
        - "type": One of the 7 categories above (e.g., "TRUNCATION", "PLACEHOLDERS", "HALLUCINATED_IMPORTS", etc.)
        - "severity": "critical" (code will fail), "major" (risky), or "minor" (cosmetic)
        - "description": Exact location and what's wrong
        - "location": Function/class/line where issue occurs
        - "recommendation": Specific fix
        
        If no issues found, return [].
        
        Example:
        [
          {{
            "type": "PLACEHOLDERS",
            "severity": "critical",
            "description": "Function calculate_metric() contains only 'pass' with comment '# ... implement calculation here'",
            "location": "calculate_metric() at line 45",
            "recommendation": "Implement the complete calculation logic for the metric"
          }},
          {{
            "type": "HALLUCINATED_IMPORTS",
            "severity": "critical",
            "description": "numpy.simulate_infection() does not exist in numpy library",
            "location": "Line 78",
            "recommendation": "Remove the call to numpy.simulate_infection() or implement the function yourself"
          }}
        ]
        """
        
        # Call LLM to perform linter check
        # Use low effort for linting task (analysis only, no code generation)
        llm_response = self._call_llm(prompt, reasoning={"effort": "low"})
        
        # Parse LLM response
        try:
            # Extract JSON from response
            first_bracket = llm_response.find('[')
            last_bracket = llm_response.rfind(']')
            
            if first_bracket == -1 or last_bracket == -1:
                self.logger.warning("Could not find JSON array in LLM Linter response")
                return []
            
            json_str = llm_response[first_bracket:last_bracket+1]
            issues = json.loads(json_str)
            
            if not issues:
                self.logger.info("LLM Linter: No high-level issues found")
            else:
                self.logger.warning(f"LLM Linter: Found {len(issues)} issues")
                # Log critical issues
                critical_issues = [issue for issue in issues if issue.get("severity") == "critical"]
                if critical_issues:
                    self.logger.warning(f"LLM Linter: {len(critical_issues)} CRITICAL issues detected")
                    for issue in critical_issues:
                        self.logger.warning(f"  - [{issue.get('type')}] {issue.get('description')}")
                
            return issues
        except Exception as e:
            self.logger.error(f"Error parsing LLM Linter response: {e}")
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
        # llm_response = self._call_llm(prompt, reasoning={"effort": "high"})
        
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
        # llm_response = self._call_llm(prompt, reasoning={"effort": "high"})
        
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
        task_spec: Dict[str, Any],
        model_plan: Dict[str, Any],
        mode: str = "full"
    ) -> str:
        """
        Improve code based on issues found during self-checking.
        
        Args:
            code: The code to improve
            issues: List of issues found (focusing on compilation-blocking issues)
            task_spec: Task specification from the Task Understanding Agent
            model_plan: Model plan from the Model Planning Agent
            mode: Workflow mode ("full", "odd", "ace", etc.)
            
        Returns:
            Improved code
        """
        self.logger.info("Improving code based on self-checking issues")
        
        # Format issues for the prompt
        issues_text = json.dumps(issues, indent=2)
        
        # Prepare task_info based on mode
        if mode in ("odd", "persona", "ace"):
            # Extract blueprint from task_spec (excluding file_summaries)
            if "data_analysis_result" in task_spec:
                blueprint = {
                    k: v
                    for k, v in task_spec["data_analysis_result"].items()
                    if k != "file_summaries"
                }
                task_info = json.dumps(blueprint, indent=2)
                self.logger.info(
                    "Extracted blueprint from task_spec for %s mode", mode
                )
            else:
                task_info = json.dumps(task_spec, indent=2)
            
            # Check task description and load appropriate patch
            task_description = task_spec.get('description', '').lower()
            
            if 'mask-wearing' in task_description:
                self.logger.info("Loading mask adoption patch for code improvement")
                try:
                    # Get project root directory (3 levels up from agents/code_generation_ace/agent.py)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "mask_adoption_patch.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        patch_content = f"\n\n{f.read()}"
                    task_info += patch_content
                except Exception as e:
                    self.logger.error(f"Error loading mask_adoption_patch.txt: {e}")
            
            elif 'user rates' in task_description or 'human trait scores' in task_description or 'daily mobility trajectories' in task_description:
                self.logger.info("Loading LLM calling patch for code improvement")
                try:
                    # Get project root directory (3 levels up from agents/code_generation_ace/agent.py)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "llm_api_call_patch_prompt.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        patch_content = f"\n\n{f.read()}"
                    task_info += patch_content
                except Exception as e:
                    self.logger.error(f"Error loading llm_api_call_patch_prompt.txt: {e}")
            
            # Persona-specific patch for psychometric test simulators
            if mode == "persona" and 'psychometric tests' in task_description:
                self.logger.info("Loading persona patch for code improvement")
                try:
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "persona_patch.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        patch_content = f"\n\n{f.read()}"
                    task_info += patch_content
                except Exception as e:
                    self.logger.error(f"Error loading persona_patch.txt: {e}")
        else:
            # For other modes, use standard format
            task_info = json.dumps(task_spec, indent=2)
        
        # Build prompt for improving code
        # Align with the 7 categories checked by LLM Linter
        prompt = f"""
        You are a code fixer (Low-Level Code Sanitizer). Your role is to fix issues found during code inspection.
        
        You are NOT a QA. You do NOT improve algorithm correctness or logic.
        Your ONLY job: fix issues that will cause the code to fail at parse time or early runtime.
        
        Generated code:
        ```python
        {code}
        ```
        
        Issues found during self-checking (from AST and LLM Linter):
        {issues_text}
        
        Task specification:
        {task_info}
        
        Model plan:
        {json.dumps(model_plan, indent=2)}
        
        Fix each issue according to its type:
        
        1. SYNTAX_ERROR (from AST check)
        - Fix syntax errors to make the code compilable
        - Ensure proper indentation, brackets, parentheses, and quotes are closed
        
        2. MARKDOWN_RESIDUE
        - Remove any residual markdown artifacts (e.g., "Here is the code:", "Hope this helps")
        - Remove incomplete code fence markers
        - Remove natural language explanations mixed with code
        
        3. TRUNCATION
        - Complete incomplete code (e.g., function definitions without bodies)
        - Close unclosed brackets, parentheses, or quotes
        - Add missing return statements in non-void functions
        
        4. PLACEHOLDERS (CRITICAL - ZERO TOLERANCE!)
        - Replace placeholder comments like "# ... rest of the code", "# TODO: fill this part" with actual implementation
        - Implement functions that only contain "pass"
        - Remove ellipsis (...) used as placeholder
        - Remove any form of "省略" or "omitted for brevity"
        
        5. HALLUCINATED_IMPORTS / HALLUCINATED_ATTRIBUTES
        - Remove non-existent library imports
        - Fix function calls on libraries that don't have those methods (e.g., numpy.calculate_infection_rate())
        - Define missing global variables or classes that are referenced
        - Fix attributes that don't exist on standard objects
        
        6. NAMESPACE_SCOPE / UNDEFINED_REFERENCES
        - Define variables before they are used
        - Fix function calls with mismatched argument counts
        - Fix circular references or variable shadowing
        - Fix methods called on objects that don't have those methods
        
        7. EXECUTION_ENTRY_POINT
        - Ensure there is a main() function
        - Ensure at the end of file there is a direct call: main() (NOT if __name__ == "__main__")
        - Ensure main() function is NOT empty or only contains pass
        - Ensure main() orchestrates the actual workflow
        
        8. HAZARDOUS_PATTERNS
        - Fix while True: loops without clear break conditions
        - Replace hardcoded absolute file paths with os.path.join using environment variables
        - Add proper error handling for file operations
        - Add zero-division checks
        
        Return the fixed code as pure Python code. Do not include any explanation or markdown formatting.
        """
        
        # Call LLM to improve code
        # Use low effort for code fixing - these are straightforward fixes to low-level issues
        # Multiple iterations provide quality control, so low effort is sufficient
        llm_response = self._call_llm(prompt, reasoning={"effort": "low"})
        
        # Extract improved code
        improved_code = self._extract_code(llm_response)
        # Remove any leftover markdown fences
        improved_code = self._strip_markdown_fences(improved_code)
        
        self.logger.info("Code improved based on self-checking issues")
        return improved_code
    
    def _build_simulator_description_prompt(self, code: str, task_spec: Dict[str, Any]) -> str:
        """
        Build a prompt to summarize the generated simulator code.
        
        The LLM should return a concise reasoning-oriented description of the model.
        """
        task_description = task_spec.get("description", "No task description provided")
        blueprint = {k: v for k, v in task_spec.get("data_analysis_result", {}).items() if k != "file_summaries"}
        blueprint_str = json.dumps(blueprint, indent=2) if blueprint else "No blueprint provided"
        code_summary = self._generate_code_summary(code)
        
        prompt = f"""
You are a simulation reviewer. Given the generated Python simulator code, produce a concise description (one short paragraph) explaining what the simulator models and why the structure/assumptions make sense.

STRICT OUTPUT: Return ONLY a JSON object with key "simulator_description" whose value is a string (no markdown, no code fences).

Context:
- Task: {task_description}
- Code summary: {code_summary}

Full generated code:
```python
{code}
```

Respond as:
{{
  "simulator_description": "<concise description and reasoning>"
}}
"""
        return prompt
    
    def _parse_simulator_description(self, llm_response: str) -> str:
        """
        Parse simulator_description from LLM response (robust to extra text).
        """
        if not llm_response:
            return ""
        
        # Try to extract JSON block first
        first_brace = llm_response.find('{')
        last_brace = llm_response.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = llm_response[first_brace:last_brace+1]
            try:
                obj = json.loads(json_str)
                desc = obj.get("simulator_description") or obj.get("description")
                if isinstance(desc, str):
                    return desc.strip()
            except Exception:
                pass
        
        # Fallback: use raw response (trimmed)
        return llm_response.strip()
    
    def _generate_simulator_description(self, code: str, task_spec: Dict[str, Any]) -> str:
        """
        Generate simulator_description once per iteration (before self-loop modifications).
        """
        prompt = self._build_simulator_description_prompt(code, task_spec)
        llm_response = self._call_llm(prompt, reasoning={"effort": "low"})
        return self._parse_simulator_description(llm_response)
    
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
        # Use low effort for syntax fixing (relatively simple task)
        llm_response = self._call_llm(prompt, reasoning={"effort": "low"})
        
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
        mode: str = "full",
        playbook: Optional[Dict[str, Any]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
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
        # Use the prompt template loaded from configuration via BaseAgent
        prompt_template = self.prompt_template
        
        # If no template is loaded, provide a fallback
        if not prompt_template:
            self.logger.warning("No prompt template loaded, using fallback template")
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
            # Extract blueprint from data_analysis_result (excluding file_summaries)
            blueprint = {k: v for k, v in task_spec.get("data_analysis_result", {}).items() if k != "file_summaries"}
            blueprint_str = json.dumps(blueprint, indent=2) if blueprint else "No blueprint provided"
            
            # Extract file_summaries from task_spec
            file_summaries = task_spec.get("file_summaries", [])
            file_summaries_str = json.dumps(file_summaries, indent=2) if file_summaries else "No file summaries available"
            
            # Format playbook as string (only for ACE/ALPHA mode)
            playbook_str = "No playbook provided"
            if mode in ["ace", "alpha"] and playbook:
                try:
                    playbook_str = json.dumps(playbook, indent=2, ensure_ascii=False)
                except TypeError:
                    # Fallback if playbook contains non-serializable objects
                    playbook_str = str(playbook)
            
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
            
            # For ACE/ALPHA mode, use template with blue_print, file_summaries, and playbook placeholders
            if mode in ["ace", "alpha"]:
                # Task description (both raw and lower-cased for matching)
                task_description_raw = task_spec.get('description', '')
                task_description = task_description_raw.lower()
                
                # Decide coding_patch content for ACE / ALPHA
                coding_patch_content = ""
                
                # Alpha-specific patch for COVID SIR calibration tasks
                if mode == "alpha" and "covid sir" in task_description:
                    self.logger.info("Alpha mode: injecting COVID SIR SBI calibration patch into {coding_patch} placeholder")
                    try:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        template_path = os.path.join(project_root, "templates", "gsim_sir_patch_prompt.txt")
                        with open(template_path, 'r', encoding='utf-8') as f:
                            coding_patch_content = f.read().strip()
                        self.logger.debug(f"Successfully loaded COVID SIR patch from {template_path}")
                    except Exception as e:
                        self.logger.error(f"Error loading gsim_sir_patch_prompt.txt: {e}")
                        coding_patch_content = ""
                # Alpha-specific patch for Three-disease Hospital calibration tasks
                elif mode == "alpha" and "three-disease hospital" in task_description:
                    self.logger.info("Alpha mode: injecting Three-disease Hospital SBI calibration patch into {coding_patch} placeholder")
                    try:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        template_path = os.path.join(project_root, "templates", "gsim_hosp_patch_prompt.txt")
                        with open(template_path, 'r', encoding='utf-8') as f:
                            coding_patch_content = f.read().strip()
                        self.logger.debug(f"Successfully loaded Three-disease Hospital patch from {template_path}")
                    except Exception as e:
                        self.logger.error(f"Error loading gsim_hosp_patch_prompt.txt: {e}")
                        coding_patch_content = ""
                # Alpha-specific patch for Beer Game (SUPPLY) calibration tasks
                elif mode == "alpha" and "beer game (supply)" in task_description:
                    self.logger.info("Alpha mode: injecting Beer Game (SUPPLY) SBI calibration patch into {coding_patch} placeholder")
                    try:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        template_path = os.path.join(project_root, "templates", "gsim_supply_patch_prompt.txt")
                        with open(template_path, 'r', encoding='utf-8') as f:
                            coding_patch_content = f.read().strip()
                        self.logger.debug(f"Successfully loaded Beer Game (SUPPLY) patch from {template_path}")
                    except Exception as e:
                        self.logger.error(f"Error loading gsim_supply_patch_prompt.txt: {e}")
                        coding_patch_content = ""
                # Existing LLMOB patch for daily mobility trajectories
                elif "daily mobility trajectories" in task_description:
                    self.logger.info("Loading llmob patch content for {coding_patch} placeholder (iteration 0)")
                    try:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        template_path = os.path.join(project_root, "templates", "llmob_patch_prompt.txt")
                        with open(template_path, 'r', encoding='utf-8') as f:
                            coding_patch_content = f.read()
                    except Exception as e:
                        self.logger.error(f"Error loading llmob_patch_prompt.txt: {e}")
                
                # Replace {coding_patch} placeholder before formatting other placeholders
                prompt_template_with_patch = prompt_template.replace("{coding_patch}", coding_patch_content)
                
                # Fill in the ACE template with all placeholders
                prompt = prompt_template_with_patch.format(
                    blue_print=blueprint_str,
                    file_summaries=file_summaries_str,
                    playbook=playbook_str
                )
            else:
                # For other modes, use the original format (without file_summaries and playbook)
                # Replace {coding_patch} placeholder first (even if empty) to avoid KeyError
                coding_patch_content = ""
                task_description = task_spec.get('description', '').lower()
                if 'daily mobility trajectories' in task_description:
                    self.logger.info("Loading llmob patch content for {coding_patch} placeholder (iteration 0, non-ACE mode)")
                    try:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        template_path = os.path.join(project_root, "templates", "llmob_patch_prompt.txt")
                        with open(template_path, 'r', encoding='utf-8') as f:
                            coding_patch_content = f.read()
                    except Exception as e:
                        self.logger.error(f"Error loading llmob_patch_prompt.txt: {e}")
                
                # Replace {coding_patch} placeholder before formatting other placeholders
                prompt_template_with_patch = prompt_template.replace("{coding_patch}", coding_patch_content)
                
                prompt = prompt_template_with_patch.format(
                    blue_print=blueprint_str,
                    model_plan=model_plan_str,
                    data_analysis=data_analysis_str,
                    feedback=feedback_str,
                    previous_code=previous_code_str,
                    data_path=data_path_str
                )
                
                # Add mask adoption patch if task description contains mask-wearing
                task_description = task_spec.get('description', '').lower()
                if 'mask-wearing' in task_description:
                    self.logger.info("Adding mask adoption temporal holdout patch to prompt")
                    try:
                        # Get project root directory (3 levels up from agents/code_generation_ace/agent.py)
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        template_path = os.path.join(project_root, "templates", "mask_adoption_patch.txt")
                        with open(template_path, 'r', encoding='utf-8') as f:
                            mask_adoption_patch = f"\n\n{f.read()}"
                        prompt += mask_adoption_patch
                    except Exception as e:
                        self.logger.error(f"Error loading mask_adoption_patch.txt: {e}")
                        # Continue without the patch if file cannot be loaded
                elif 'user rates' in task_description or 'daily mobility trajectories' in task_description:
                    self.logger.info("Adding use modelling llm calling patch to prompt")
                    try:
                        # Get project root directory (3 levels up from agents/code_generation_ace/agent.py)
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        template_path = os.path.join(project_root, "templates", "llm_api_call_patch_prompt.txt")
                        with open(template_path, 'r', encoding='utf-8') as f:
                            llm_calling_patch = f"\n\n{f.read()}"
                        prompt += llm_calling_patch
                    except Exception as e:
                        self.logger.error(f"Error loading llm_api_call_patch_prompt.txt: {e}")
                        # Continue without the patch if file cannot be loaded
        
        return prompt
    
    def _transform_playbook_for_prompt(self, playbook: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transform playbook format for prompt usage:
        - Remove playbook_metadata
        - Filter to only include strategies with status="in_progress"
          (These are strategies that were selected via select_strategies_for_prompt())
        - Remove meta_info from each strategy
        - Flatten reflection content to be the direct content of each strategy
        
        Status lifecycle:
        - open: New or reactivated strategy, waiting for selection
        - queued: Not selected due to budget, waiting in queue
        - in_progress: Selected for current prompt (E_selected event)
        - resolved: Successfully resolved
        
        Args:
            playbook: Original playbook dictionary
            
        Returns:
            Transformed playbook dictionary with only in_progress strategies (reflection content flattened)
        """
        if not playbook or not isinstance(playbook, dict):
            return {"strategies": {}}
        
        transformed = {"strategies": {}}
        
        # Valid statuses for inclusion in prompt (strategies that were selected)
        SELECTABLE_STATUSES = ["in_progress"]
        
        strategies = playbook.get("strategies", {})
        for strategy_id, strategy_data in strategies.items():
            if not isinstance(strategy_data, dict):
                continue
            
            # Check status: only include strategies with in_progress status
            # Status is stored in meta_info.status
            meta_info = strategy_data.get("meta_info", {})
            if not isinstance(meta_info, dict):
                # Skip if meta_info is missing or not a dict
                self.logger.debug(f"Skipping strategy '{strategy_id}' - missing or invalid meta_info")
                continue
            
            status = meta_info.get("status", "open")
            
            # Skip if status is not in_progress
            if status not in SELECTABLE_STATUSES:
                self.logger.debug(f"Skipping strategy '{strategy_id}' with status '{status}' (only in_progress strategies included)")
                continue
            
            # Get reflection content (flatten it)
            reflection = strategy_data.get("reflection", {})
            
            # If reflection is empty or not a dict, skip this strategy
            if not reflection or not isinstance(reflection, dict):
                continue
            
            # Flatten reflection: reflection fields become direct fields of the strategy
            transformed["strategies"][strategy_id] = dict(reflection)
        
        self.logger.debug(f"Transformed playbook: {len(transformed['strategies'])} in_progress strategies included")
        return transformed
    
    def _build_patch_prompt(
        self,
        task_spec: Dict[str, Any],
        previous_code: Optional[Dict[str, str]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
        playbook: Optional[Dict[str, Any]] = None,
        best_simulator_info: Optional[Dict[str, Any]] = None,
        simulation_info_history: Optional[List[Dict[str, Any]]] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """
        Build a patch-level prompt for code generation (iteration >= 1).
        
        Args:
            task_spec: Task specification containing blueprint
            previous_code: Code from the previous iteration
            simulation_results: Results from simulation execution
            playbook: Playbook dictionary (will be transformed)
            best_simulator_info: Best simulator info for alpha mode (optional)
        
        Returns:
            Formatted prompt string
        """
        # Hardcoded patch prompt template
#         prompt_template = """SYSTEM:
# You are the Code Patch Agent in a system that generates social simulations.
# Your job is to FIX and IMPROVE the existing simulator code to better satisfy the Blueprint, while keeping changes minimal.
# You MUST use the Playbook as an external tool: read it first, then selectively apply only the relevant strategies.
# Treat the Playbook as a tool. Use relevant parts, do NOT force irrelevant parts into the solution.
# If the Playbook conflicts with the Blueprint, ALWAYS follow the Blueprint.
#
# CRITICAL "PATCH-LEVEL" RULES (HARD CONSTRAINTS):
# - You must NOT rewrite the entire program from scratch.
# - You must output a single updated standalone Python program (full code) based on PREVIOUS_CODE.
# - Make minimal, targeted edits to PREVIOUS_CODE: prefer editing only the functions/classes implicated by Playbook code_refs, error logs, or blueprint mismatches.
# - Preserve stable public interfaces and the orchestrator pipeline unless Blueprint explicitly requires change:
#   parse_cli() (optional) → load_data() → build_network_and_agents() → holdout_split()
#   → calibrator.fit() → simulator.rollout() → evaluator.compute_metrics() → save_results()
# - Do NOT rename existing public functions/classes unless absolutely necessary.
# - Do NOT reorder major code blocks unless necessary.
# - Do NOT delete required steps in main(). You may add small helper functions/classes if needed.
# - Ensure deterministic behavior via a global random seed and keep existing seeding approach if present.
#
# USER:
# You will be given:
# (1) A BLUEPRINT (ground truth requirements)
# (2) PREVIOUS_CODE (the simulator code from the previous iteration)
# (3) SIMULATION RESULTS (tracebacks, errors, metrics) and/or REFLECTOR_ISSUES (structured issues)
# (4) A PLAYBOOK (reusable best practices, pitfalls, templates; structured "strategies");
#
# Your job: patch the PREVIOUS_CODE to resolve the failures/underperformance, aligned with BLUEPRINT, leveraging PLAYBOOK strategies.
#
# ========================
# BLUEPRINT (AUTHORITATIVE)
# ========================
# {blue_print}
#
# ========================
# PREVIOUS_CODE (BASELINE TO PATCH)
# ========================
# {previous_code}
#
# ========================
# SIMULATION RESULTS
# ========================
# {simulation_results}
#
# ========================
# PLAYBOOK (READ FIRST)
# ========================
# {playbook}
#
# ========================
# ADDITIONAL CONSTRAINTS
# ========================
# 1. Global Requirements:
#  - Write clean, modular, PEP-8 compliant code with complete docstrings (triple-quoted, not truncated).
#  - Provide full class/function bodies (no stubs).
#  - The output must be a single standalone Python program that runs end-to-end without manual edits.
#  - Validate all inputs; raise clear exceptions with actionable messages.
#  - Avoid unnecessary refactors. Only touch code necessary for fixes.
#
# 2. Path Handling Instructions (MUST PRESERVE):
# Keep the path setup exactly:
# ```python
# import os
# PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
# DATA_PATH = os.environ.get("DATA_PATH")
# DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
#
# 3. Orchestrator (MUST PRESERVE MAIN FLOW):
#  - main() must remain non-empty and executable end-to-end.
#  - main() must call, in order: parse_cli() (optional) → load_data() → build_network_and_agents() → holdout_split()
# → calibrator.fit() → simulator.rollout() → evaluator.compute_metrics() → save_results().
#  - You may add logging/checks, but do not remove steps.
#
# ========================
# PLAYBOOK USAGE POLICY (MANDATORY)
#  - The Playbook is a JSON object with top-level key "strategies".
#  - Each strategy contains fields like: issue_type, severity, blueprint_refs, code_refs, correct_approach.
#  - Prioritize applying strategies in this order: blocker > high > medium > low.
#  - Select only strategies that are relevant to:
#     - current Blueprint requirements, OR
#     - implicated symbols/lines in PREVIOUS_CODE, OR
#     - SIMULATION_RESULTS / REFLECTOR_ISSUES symptoms.
#  - If a selected strategy suggests a change, you must implement it in code.
#  - If a strategy seems relevant but you do NOT implement it, you must explain why in the change summary.
#
# ========================
# OUTPUT FORMAT (STRICT)
# 1. First line: a Python comment block starting with:
# PLAYBOOK_USAGE_JSON = '''<one-line valid JSON enclosed in triple quotes>'''
# JSON format MUST be:
# {{"used_bullets":[{{"id":"<strategy_id>","why":"<one-line relevance>"}}]}}
# CRITICAL: Wrap the entire JSON content in triple quotes (''') to make it a Python string literal.
# Example: PLAYBOOK_USAGE_JSON = '''{{"used_bullets":[{{"id":"strategy-1","why":"reason"}}]}}'''
#
# 2. Second line: a Python comment block starting with:
# CHANGE_SUMMARY_JSON = '''<one-line valid JSON enclosed in triple quotes>'''
# JSON format MUST be:
# {{
# "touched_symbols":[{{"symbol":"<func/class>","reason":"<why changed>"}}],
# "applied_strategies":[{{"id":"<strategy_id>","applied":true|false,"note":"<short>"}}]
# }}
# CRITICAL: Wrap the entire JSON content in triple quotes (''') to make it a Python string literal.
# Keep it one line JSON. No extra commentary.
#
# 3. Then output PURE PYTHON CODE ONLY (no markdown fences).
# The code must be the full updated program after applying patch-level edits.
#
# DO NOT output anything else.
# """

        # Use new prompt template for alpha mode with simulation_info_history, otherwise use original
        if simulation_info_history is not None:
            # New prompt template for alpha mode
            prompt_template = """SYSTEM:
        # ROLE & OBJECTIVE
        You are the Code Patch Agent for a social simulation system.
        Your Goal: FIX and IMPROVE the `PREVIOUS_CODE` and resolve `SIMULATION_RESULTS`, with the specific aim of achieving a LOWER validation error.
        You MUST use the Playbook as an external tool: read it first, then selectively apply only the relevant strategies.
        Treat the Playbook as a tool. Use relevant parts, do NOT force irrelevant parts into the solution.

        # CRITICAL "PATCH-LEVEL" PROTOCOL (HARD CONSTRAINTS)
        1. **Refinement, Not Rewrite**: Do NOT rewrite the entire program from scratch. Make minimal, targeted edits.
        2. **Single File**: You must output a single updated standalone Python program based on PREVIOUS_CODE.
        3. **Preserve Interfaces / Skeleton**:
           - You MUST NOT change the existing code skeleton, or input variables.
           - Do NOT rename existing public functions/classes or reorder major blocks unless Blueprint explicitly requires it.
           - Keep function signatures and the orchestrator flow intact.
        4. **You MUST generate code**: You cannot give an empty-string answer. You must output runnable Python code.

        # DO-NOT-CHANGE GUARANTEES (STRICT)
        A) **Do NOT modify any output files, filenames, output paths, output schemas, or output formats** produced by the program.
           - Keep the exact same files written to disk as in PREVIOUS_CODE.
           - Keep the exact same CSV/JSON structure, column names, key names, and serialization format.
        B) **Do NOT modify metric computation**:
           - Keep metric definitions, aggregation, and reporting exactly the same as PREVIOUS_CODE.
           - Do NOT change how training/validation/test metrics are computed, named, or logged.
        C) **Do NOT change integration-required path handling** (must copy exactly as provided below).

        # OUTPUT REQUIREMENT (STRICT, but position-aware):
        You must output the following two variables as triple-quoted JSON strings somewhere near the top of the file, but do NOT break Python syntax.
        1. `PLAYBOOK_USAGE_JSON = '''...'''` (Triple-quoted JSON string)
        Schema: {{"used_bullets":[{{"id":"<strategy_id>","why":"<relevance>"}}]}}
        2. `CHANGE_SUMMARY_JSON = '''...'''` (Triple-quoted JSON string)
        Schema: {{"touched_symbols":[{{"symbol":"<name>","reason":"<why>"}}], "applied_strategies":[{{"id":"<id>","applied":true}}]}}
        Placement rules (to avoid SyntaxError):
           - If the program includes any from __future__ import ... statements (e.g., from __future__ import annotations), those future-import lines MUST appear before any other executable statements. Therefore, place the two JSON variables immediately after the future-import line(s) and after the module docstring (if any), but before the rest of the code.
           - If there is no future-import, place the two JSON variables at the top of the file after an optional module docstring.
           - Do not wrap these JSON strings in Markdown fences. The rest of the output must be pure Python code.

        # CORE FUNCTIONAL REQUIREMENT (MERGED; HIGH PRIORITY)
        Please now regenerate the code function(s) / implementation where needed, with the aim to improve the code to achieve a lower validation error.
        - Use the feedback where applicable (Simulation Results + Playbook + Blueprint).
        - When you are unsure about something, take your best guess.
        - You have to generate code, and cannot give an empty string answer.
        - You cannot change the code skeleton, or input variables.
        - You MUST preserve the program's external behavior: output files and formats, and metric computation, must remain unchanged.

        USER:
        Here is the context for the current iteration.

        # PART 1: CONTEXT & DIAGNOSTICS (Reference Material)
        ========================
        PREVIOUS_CODE (Baseline to Patch)
        ========================
        {previous_code}

        ========================
        SIMULATION RESULTS (Symptoms & Metrics)
        ========================
        {simulation_results}

        # PART 2: IMPLEMENTATION CONSTRAINTS (MUST FOLLOW)
        *Attention: The following rules determine the correctness of the system integration.*
        1. **Path Handling Instructions (COPY EXACTLY)**:
           Keep the path setup exactly as follows:
           ```python
           PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
           DATA_PATH = os.environ.get("DATA_PATH")
           DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
           ```
           Note: Do NOT add `import os` inside functions if `os` is already imported at the module level.
           Only use the path setup code above.

        2. Global Requirements:
            - Write clean, modular, PEP-8 compliant code.
            - Complete docstrings (triple-quoted).
            - Full class/function bodies (NO stubs, NO pass without reason).
            - Validate all inputs; raise clear exceptions.

        3. Coding patch:
            {coding_patch}

        # PART 3: TARGET & STRATEGY (High-Attention Region)
        CRITICAL: Read the Blueprint and Playbook immediately before coding.
        ========================
        BLUEPRINT (AUTHORITATIVE)
        ========================
        {blue_print}

        ========================
        PLAYBOOK (The Solution Strategy)
        Usage Policy:
        - Prioritize: blocker > high > medium > low.
        - Select only strategies relevant to the current Blueprint or Simulation Results.
        - If a strategy suggests a change, you MUST implement it.
        ========================
        {playbook}

        # FINAL INSTRUCTION:
        1. Review the Playbook and Blueprint above.
        2. Check the Simulation Results to identify what broke and what causes high validation error.
        3. Apply the Playbook strategies to fix PREVIOUS_CODE while strictly adhering to the Implementation Constraints (Path & Orchestrator).
        4. Improve validation error by fixing mechanisms/logic bugs/mismatches, but do NOT change:
           - output files, filenames, output schemas/formats
           - metric computation and metric reporting
           - code skeleton, input variables, or public interfaces

        Generate the response now, starting strictly with the first line:
        PLAYBOOK_USAGE_JSON = '''
        """
        else:
            # Original prompt template for ACE mode
            prompt_template = """SYSTEM:
        # ROLE & OBJECTIVE
        You are the Code Patch Agent for a social simulation system.
        Your Goal: FIX and IMPROVE the `PREVIOUS_CODE` to satisfy the `BLUEPRINT` and resolve `SIMULATION_RESULTS`.
        You MUST use the Playbook as an external tool: read it first, then selectively apply only the relevant strategies.
        Treat the Playbook as a tool. Use relevant parts, do NOT force irrelevant parts into the solution.

        # CRITICAL "PATCH-LEVEL" PROTOCOL (HARD CONSTRAINTS)
        1. **Refinement, Not Rewrite**: Do NOT rewrite the entire program from scratch. Make minimal, targeted edits.
        2. **Single File**: You must output a single updated standalone Python program based on PREVIOUS_CODE.
        3. **Preserve Interfaces**: Do NOT rename existing public functions/classes or reorder major blocks unless Blueprint explicitly requires it.
        4. **Deterministic**: Ensure deterministic behavior via a global random seed.

        # OUTPUT REQUIREMENT (STRICT, but position-aware):
        You must output the following two variables as triple-quoted JSON strings somewhere near the top of the file, but do NOT break Python syntax.
        1. `PLAYBOOK_USAGE_JSON = '''...'''` (Triple-quoted JSON string)
        Schema: {{"used_bullets":[{{"id":"<strategy_id>","why":"<relevance>"}}]}}
        2. `CHANGE_SUMMARY_JSON = '''...'''` (Triple-quoted JSON string)
        Schema: {{"touched_symbols":[{{"symbol":"<name>","reason":"<why>"}}], "applied_strategies":[{{"id":"<id>","applied":true}}]}}
        Placement rules (to avoid SyntaxError):
           - If the program includes any from __future__ import ... statements (e.g., from __future__ import annotations), those future-import lines MUST appear before any other executable statements. Therefore, place the two JSON variables immediately after the future-import line(s) and after the module docstring (if any), but before the rest of the code.
           - If there is no future-import, place the two JSON variables at the top of the file after an optional module docstring.
           - Do not wrap these JSON strings in Markdown fences. The rest of the output must be pure Python code.

        USER:
        Here is the context for the current iteration.
        
        # PART 1: CONTEXT & DIAGNOSTICS (Reference Material)
        ========================
        PREVIOUS_CODE (Baseline to Patch)
        ========================
        {previous_code}
        
        ========================
        SIMULATION RESULTS (Symptoms & Metrics)
        ========================
        {simulation_results}
        
        
        # PART 2: IMPLEMENTATION CONSTRAINTS (MUST FOLLOW)
        *Attention: The following rules determine the correctness of the system integration.*
        1. **Path Handling Instructions (COPY EXACTLY)**:
           Keep the path setup exactly as follows:
           ```python
           import os
           PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
           DATA_PATH = os.environ.get("DATA_PATH")
           DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
        
        2. Orchestrator Pipeline (MUST PRESERVE MAIN FLOW):
            main() must call, in strict order: parse_cli() (optional) → load_data() → build_network_and_agents() → holdout_split() → calibrator.fit() → simulator.rollout() → evaluator.compute_metrics() → save_results()
            You may add logging/checks, but do not remove these steps.
        
        3. Global Requirements:
            Write clean, modular, PEP-8 compliant code.
            Complete docstrings (triple-quoted).
            Full class/function bodies (NO stubs, NO pass without reason).
            Validate all inputs; raise clear exceptions.
            
        4. Coding patch:
            {coding_patch}
        
        
        # PART 3: TARGET & STRATEGY (High-Attention Region)
        CRITICAL: Read the Blueprint and Playbook immediately before coding.
        ========================
        BLUEPRINT (AUTHORITATIVE)
        ========================
        {blue_print}

        ========================
        PLAYBOOK (The Solution Strategy)
        Usage Policy:
        Prioritize: blocker > high > medium > low.
        Select only strategies relevant to the current Blueprint or Simulation Results.
        If a strategy suggests a change, you MUST implement it.
        ========================
        {playbook}


        # FINAL INSTRUCTION:
        1. Review the Playbook and Blueprint above.
        2. Check the Simulation Results to identify what broke.
        3. Apply the Playbook strategies to fix PREVIOUS_CODE while strictly adhering to the Implementation Constraints (Path & Orchestrator).
        
        Generate the response now, starting strictly with the first line: PLAYBOOK_USAGE_JSON = ''' 
        """
        
        # Extract blueprint from task_spec (excluding file_summaries)
        blueprint = {k: v for k, v in task_spec.get("data_analysis_result", {}).items() if k != "file_summaries"}
        blueprint_str = json.dumps(blueprint, indent=2, ensure_ascii=False) if blueprint else "No blueprint provided"
        
        # Format previous code and simulation results
        # Priority order:
        # 1. If simulation_info_history exists: use current iteration's data from history
        # 2. Else if best_simulator_info exists: use best_simulator_info data
        # 3. Else: use previous_code and simulation_results parameters (ACE mode)
        
        # First, try to get PREVIOUS iteration info from simulation_info_history
        # For iteration k (k >= 1), we want to patch based on iteration k-1
        prev_iteration_info = None
        if simulation_info_history is not None and iteration is not None and iteration > 0:
            prev_iter = iteration - 1
            for hist_item in simulation_info_history:
                if hist_item.get("iteration") == prev_iter:
                    prev_iteration_info = hist_item
                    break
        
        if prev_iteration_info is not None:
            # Use previous iteration's data from simulation_info_history
            previous_code_str = prev_iteration_info.get("code", "") or "No previous code available"
            prev_results_json = prev_iteration_info.get("results_json", {}) or {}
            simulation_results_str = json.dumps(prev_results_json, indent=2, default=str, ensure_ascii=False) if prev_results_json else "No simulation results provided"
            self.logger.info(f"Alpha mode: Using previous iteration {prev_iter} data from simulation_info_history for patch prompt")
        elif best_simulator_info is not None:
            # Fallback: Use best_simulator_info for alpha mode
            previous_code_str = best_simulator_info.get("code", "") or "No previous code available"
            results_json = best_simulator_info.get("results_json", {}) or {}
            simulation_results_str = json.dumps(results_json, indent=2, default=str, ensure_ascii=False) if results_json else "No simulation results provided"
            self.logger.info(f"Alpha mode: Using best_simulator_info from iteration {best_simulator_info.get('iteration', 'N/A')} with val_loss {best_simulator_info.get('val_loss', 'N/A')}")
        else:
            # Use previous_code and simulation_results parameters (ACE mode)
            previous_code_str = "No previous code available"
            if previous_code:
                if isinstance(previous_code, dict):
                    # Get the first code file (usually there's only one)
                    for filename, code in previous_code.items():
                        previous_code_str = code
                        break
                elif isinstance(previous_code, str):
                    previous_code_str = previous_code
            
            # Format simulation results
            simulation_results_str = json.dumps(simulation_results, indent=2, default=str, ensure_ascii=False) if simulation_results else "No simulation results provided"
        
        # Transform and format playbook (always follow ACE-mode behavior):
        # only include strategies that were selected for the current prompt (status="in_progress").
        transformed_playbook = self._transform_playbook_for_prompt(playbook)
        playbook_str = json.dumps(transformed_playbook, indent=2, ensure_ascii=False)
        
        # For ACE mode: check if task description contains "daily mobility trajectories"
        # For alpha mode with best_simulator_info: check if task description contains "COVID SIR"
        coding_patch_content = ""
        task_description = task_spec.get('description', '').lower()
        
        if best_simulator_info is None:
            # ACE mode: check for "daily mobility trajectories"
            if 'daily mobility trajectories' in task_description:
                self.logger.info("Loading llmob patch content for {coding_patch} placeholder (iteration >= 1)")
                try:
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "llmob_patch_prompt.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        coding_patch_content = f.read()
                except Exception as e:
                    self.logger.error(f"Error loading llmob_patch_prompt.txt: {e}")
        else:
            # Alpha mode: check for task-specific patches
            if "covid sir" in task_description:
                self.logger.info("Alpha mode: Loading COVID SIR patch content for {coding_patch} placeholder (iteration >= 1)")
                try:
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "gsim_sir_patch_prompt.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        coding_patch_content = f.read().strip()
                    self.logger.debug(f"Successfully loaded COVID SIR patch from {template_path}")
                except Exception as e:
                    self.logger.error(f"Error loading gsim_sir_patch_prompt.txt: {e}")
                    coding_patch_content = ""
            elif "three-disease hospital" in task_description:
                self.logger.info("Alpha mode: Loading Three-disease Hospital patch content for {coding_patch} placeholder (iteration >= 1)")
                try:
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "gsim_hosp_patch_prompt.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        coding_patch_content = f.read().strip()
                    self.logger.debug(f"Successfully loaded Three-disease Hospital patch from {template_path}")
                except Exception as e:
                    self.logger.error(f"Error loading gsim_hosp_patch_prompt.txt: {e}")
                    coding_patch_content = ""
            elif "beer game (supply)" in task_description:
                self.logger.info("Alpha mode: Loading Beer Game (SUPPLY) patch content for {coding_patch} placeholder (iteration >= 1)")
                try:
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    template_path = os.path.join(project_root, "templates", "gsim_supply_patch_prompt.txt")
                    with open(template_path, 'r', encoding='utf-8') as f:
                        coding_patch_content = f.read().strip()
                    self.logger.debug(f"Successfully loaded Beer Game (SUPPLY) patch from {template_path}")
                except Exception as e:
                    self.logger.error(f"Error loading gsim_supply_patch_prompt.txt: {e}")
                    coding_patch_content = ""
        
        # Replace {coding_patch} placeholder first
        prompt_template_with_patch = prompt_template.replace("{coding_patch}", coding_patch_content)
        
        # Replace other placeholders
        prompt = prompt_template_with_patch.replace("{blue_print}", blueprint_str)
        prompt = prompt.replace("{previous_code}", previous_code_str)
        prompt = prompt.replace("{simulation_results}", simulation_results_str)
        prompt = prompt.replace("{playbook}", playbook_str)
        
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
            code += f"            entity = {entity_name}(f'{entity_name.lower()}_{{i}}')\n"
            
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
    
    def _update_blueprint_from_generated_code(self, blueprint, result, task_spec):
        """
        Update blueprint based on generated code and metadata.
        
        Args:
            blueprint: Blueprint object to update
            result: Generated code result containing code and metadata
            task_spec: Task specification
        """
        try:
            # Store code generation result
            blueprint.set("code_generated", True)
            blueprint.set("code_length", len(result.get('code', '')))
            
            # Extract and store metadata from result
            if "metadata" in result:
                metadata = result["metadata"]
                blueprint.set("code_metadata", metadata)
                
                # Store specific metadata fields
                if "design_patterns" in metadata:
                    blueprint.set("design_patterns", metadata["design_patterns"])
                
                if "main_class" in metadata:
                    blueprint.set("main_class", metadata["main_class"])
                
                if "imports" in metadata:
                    blueprint.set("imports", metadata["imports"])
                
                if "classes" in metadata:
                    blueprint.set("classes", metadata["classes"])
                
                if "functions" in metadata:
                    blueprint.set("functions", metadata["functions"])
            
            # Store task-specific information
            if task_spec and "objective" in task_spec:
                blueprint.set("objective", task_spec["objective"])
            
            self.logger.debug("Blueprint updated from generated code")
            
        except Exception as e:
            self.logger.error(f"Error updating blueprint from generated code: {e}")

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

    # Fixed system-role content used by _call_llm_with_functions.
    # Exposed as a class constant so orchestrators can reconstruct the same
    # message without duplicating the string.
    SYSTEM_CONTENT: str = (
        "Objective: Write code to create an accurate and realistic simulator for a given task in NumPy.\n"
        "Please note that the code should be fully functional. No placeholders.\n"
        "You must act autonomously and you will receive no human input at any stage. "
        "You have to return as output the complete code for completing this task, and correctly "
        "improve the code to create the most accurate and realistic simulator possible.\n"
        "You always write out the code contents. You always indent code with tabs.\n"
        "You cannot visualize any graphical output. You exist within a machine. "
        "The code can include black box multi-layer perceptions where required.\n"
        "Use the functions provided. When calling any helper function, only provide a "
        "RFC8259 compliant JSON request (no additional text or formatting)."
    )

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
        simulation_results: Optional[Dict[str, Any]] = None,
        best_simulator_info: Optional[Dict[str, Any]] = None,
        simulation_info_history: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
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
            messages: Shared conversation message list (list[dict]). When provided,
                the [system, user] messages used for this LLM call are appended
                in-place so callers can track the full conversation.
        
        Returns:
            Dictionary containing the generated code and metadata
        """
        self.logger.info("Generating simulation code")
        
        # Log blueprint usage if available
        if blueprint is not None:
            self.logger.info("Using blueprint for code generation in blueprint mode")
            self.logger.debug(f"Blueprint contains {len(blueprint)} items")
        
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
            "simulation_results": simulation_results,
            "iteration": iteration,
        }
        
        # Use patch prompt for iteration >= 1 (second iteration and beyond)
        # For ACE mode: use patch prompt
        # For ALPHA mode: use patch prompt if simulation_info_history is available
        # For GSIM mode (iter >= 1): use _build_gsim_patch_prompt which feeds
        #   best_simulator_info['code'] into the gsim-specific patch template and
        #   calls the LLM via Responses API with the shared messages conversation.
        if iteration is not None and iteration >= 1:
            if mode == "gsim":
                self.logger.info(
                    f"Using gsim patch prompt for iteration {iteration} (GSIM mode)"
                )
                llm_response, simulator_description_from_llm = self._build_gsim_patch_prompt(
                    task_spec=task_spec,
                    previous_code=previous_code,
                    simulation_results=simulation_results,
                    best_simulator_info=best_simulator_info,
                    simulation_info_history=simulation_info_history,
                    iteration=iteration,
                    messages=messages,
                )
                # After getting refined code, trim the shared messages back to
                # the original [system, user] pair so the next iteration starts
                # from a clean two-message context.
                if messages is not None and len(messages) > 2:
                    del messages[2:]
                    self.logger.info(
                        "GSIM patch: trimmed messages back to [system, user] (2 entries)"
                    )
                # llm_response already IS the code string; skip the standard LLM call below
                code = self._extract_code(llm_response)
                code = self._strip_markdown_fences(code)
                if feedback and isinstance(feedback, dict) and 'code_snippets' in feedback:
                    for snippet in feedback['code_snippets']:
                        before = snippet.get('before', '')
                        after = snippet.get('after', '')
                        if before and after and before in code:
                            self.logger.info(f"Applying feedback snippet from {snippet.get('file')}")
                            code = code.replace(before, f"# FIXED: Applied feedback snippet from {snippet.get('file')}\n{after}")
                code = self._fix_unclosed_docstrings(code)
                if model_plan is None:
                    model_plan = {}
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
                simulator_description = simulator_description_from_llm
                if not simulator_description:
                    try:
                        simulator_description = self._generate_simulator_description(code, task_spec)
                    except Exception as e:
                        self.logger.warning(f"Failed to generate simulator_description: {e}")
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
                self.logger.info("Code generation (gsim patch) completed")
                if blueprint is not None:
                    self._update_blueprint_from_generated_code(blueprint, result, task_spec)
                return result
            elif mode == "ace":
                self.logger.info(f"Using patch prompt for iteration {iteration} (ACE mode)")
                prompt = self._build_patch_prompt(
                    task_spec=task_spec,
                    previous_code=previous_code,
                    simulation_results=simulation_results,
                )
            elif mode == "alpha" and simulation_info_history is not None:
                self.logger.info(f"Using patch prompt for iteration {iteration} (ALPHA mode with simulation_info_history)")
                prompt = self._build_patch_prompt(
                    task_spec=task_spec,
                    previous_code=previous_code,
                    simulation_results=simulation_results,
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
        llm_response, simulator_description_from_llm, _built_messages = self._call_llm_with_functions(
            prompt, reasoning={"effort": "medium"}
        )
        # Append the [system, user] messages used for this call to the shared
        # conversation list so the orchestrator can track the full transcript.
        if messages is not None and _built_messages:
            messages.extend(_built_messages)
        
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

        # Use the description returned by the LLM tool call; fall back to a
        # separate generation call only when the tool call returned nothing.
        simulator_description = simulator_description_from_llm
        if not simulator_description:
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

    def _call_llm_with_functions(
        self, prompt: str, reasoning: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """
        Call the LLM using structured messages + OpenAI Responses API tool_choice to
        force structured code output.

        Builds two messages:
          - system: high-level objective / role instructions  (from SYSTEM_CONTENT)
          - user:   the full prompt string

        Forces the model to call `complete_SimStep_code` via tool_choice, then
        extracts both `SimulatorStep_code` and `simulator_description_and_reasoning`
        from the tool call arguments.

        Compatible with reasoning models (o-series) via the Responses API.
        Falls back to the standard `_call_llm` for non-OpenAI providers or on
        any API error.

        Args:
            prompt:    The prompt string (used as user message content)
            reasoning: Optional reasoning parameters forwarded to responses.create

        Returns:
            Tuple[str, str, List[dict]]:
              (SimulatorStep_code, simulator_description_and_reasoning, messages_sent).
            On fallback paths the description is an empty string and messages_sent is [].
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_CONTENT},
            {"role": "user", "content": prompt},
        ]

        tool_spec = {
            "type": "function",
            "name": "complete_SimStep_code",
            "description": "Write out the code for the pytorch simulator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "simulator_description_and_reasoning": {
                        "type": "string",
                        "description": "A concise description and reasoning of the code model.",
                    },
                    "SimulatorStep_code": {
                        "type": "string",
                        "description": (
                            "Code for the pytorch simulator, inclusive of the simulator definition. "
                            "If you are unsure, take your best guess. This must be a nonempty string."
                        ),
                    },
                },
                "required": ["simulator_description_and_reasoning", "SimulatorStep_code"],
            },
        }

        # Determine which provider is active
        try:
            import yaml
            with open("config.yaml", "r") as f:
                global_config = yaml.safe_load(f)
            provider_name = global_config.get("llm", {}).get("provider", "mock").lower()
        except Exception as cfg_err:
            self.logger.error(f"Could not read config.yaml for provider detection: {cfg_err}")
            provider_name = "mock"

        if provider_name != "openai":
            self.logger.info(
                f"Provider '{provider_name}' does not support tool_choice; "
                "falling back to standard _call_llm"
            )
            return self._call_llm(prompt, reasoning=reasoning), "", []

        try:
            from openai import OpenAI
            from utils.llm_utils import load_api_key

            api_key = load_api_key("OPENAI_API_KEY")
            if not api_key:
                api_key = global_config.get("llm_providers", {}).get("openai", {}).get("api_key")
            if not api_key:
                self.logger.warning("OpenAI API key not found; falling back to _call_llm")
                return self._call_llm(prompt, reasoning=reasoning), "", []

            client = OpenAI(api_key=api_key)
            provider_cfg = global_config.get("llm_providers", {}).get("openai", {})
            model = provider_cfg.get("model", "gpt-4o")
            max_output_tokens = provider_cfg.get("max_output_tokens") or provider_cfg.get("max_tokens", 100000)

            responses_kwargs: Dict[str, Any] = {
                "model": model,
                "input": messages,
                "tools": [tool_spec],
                "tool_choice": {"type": "function", "name": "complete_SimStep_code"},
                "max_output_tokens": max_output_tokens,
            }
            if reasoning:
                responses_kwargs["reasoning"] = reasoning

            self.logger.info(
                f"Calling OpenAI Responses API with tool_choice=complete_SimStep_code "
                f"(model={model}, reasoning={reasoning})"
            )
            resp = client.responses.create(**responses_kwargs)

            # Extract function call arguments from the response output list
            for item in getattr(resp, "output", []):
                item_type = getattr(item, "type", None)
                if item_type == "function_call":
                    raw_args = getattr(item, "arguments", None)
                    if raw_args:
                        try:
                            args = json.loads(raw_args)
                            code = args.get("SimulatorStep_code", "")
                            description = args.get("simulator_description_and_reasoning", "")
                            if code:
                                self.logger.info(
                                    "Successfully extracted SimulatorStep_code from tool_call response"
                                )
                                return code, description, messages
                            self.logger.warning(
                                "SimulatorStep_code is empty in tool_call arguments; "
                                "returning raw arguments string"
                            )
                            return raw_args, description, messages
                        except json.JSONDecodeError as parse_err:
                            self.logger.error(
                                f"Failed to parse tool_call arguments as JSON: {parse_err}; "
                                "returning raw arguments string"
                            )
                            return raw_args, "", messages

            # Fallback: try output_text helper (plain text response)
            output_text = getattr(resp, "output_text", None)
            if output_text:
                self.logger.warning(
                    "Model did not return a tool_call; using output_text fallback"
                )
                return output_text, "", messages

            self.logger.warning("No usable output found in Responses API response")
            return "", "", messages

        except Exception as exc:
            self.logger.error(
                f"Error in _call_llm_with_functions: {exc}; falling back to _call_llm"
            )
            return self._call_llm(prompt, reasoning=reasoning), "", []

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
            
            # For ACE/ALPHA/GSIM mode, use template with blue_print, file_summaries placeholders
            if mode in ["ace", "alpha", "gsim"]:
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
                
                # Replace {coding_patch} and {playbook} placeholders before formatting
                prompt_template_with_patch = prompt_template.replace("{coding_patch}", coding_patch_content)
                prompt_template_with_patch = prompt_template_with_patch.replace("{playbook}", "")
                
                # Fill in the template with blueprint and file_summaries placeholders
                prompt = prompt_template_with_patch.format(
                    blue_print=blueprint_str,
                    file_summaries=file_summaries_str,
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
    
    def _build_gsim_patch_prompt(
        self,
        task_spec: Dict[str, Any],
        previous_code: Optional[Dict[str, str]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
        best_simulator_info: Optional[Dict[str, Any]] = None,
        simulation_info_history: Optional[List[Dict[str, Any]]] = None,
        iteration: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> tuple:
        """
        Build a gsim-specific patch prompt (iteration >= 1) and call the LLM via
        the Responses API using the shared messages conversation.

        Template: templates/code_generation_gsim_patch.txt
        Placeholder: {completions} → best_simulator_info['code']

        Appends a new user message (with function_call hint) to `messages`, calls
        client.responses.create with tool_choice=complete_SimStep_code, then
        returns (code_str, description_str).  The caller is responsible for
        trimming messages back to [0,1] after this returns.

        Args:
            Same signature as _build_patch_prompt plus `messages`.

        Returns:
            Tuple[str, str]: (SimulatorStep_code, simulator_description_and_reasoning)
        """
        # Load the gsim patch template
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            patch_template_path = os.path.join(project_root, "templates", "code_generation_gsim_patch.txt")
            with open(patch_template_path, "r", encoding="utf-8") as f:
                patch_template = f.read()
        except Exception as e:
            self.logger.error(f"Failed to load code_generation_gsim_patch.txt: {e}")
            raise

        # Fill {completions} with best_simulator_info['code']
        best_code = ""
        if best_simulator_info and best_simulator_info.get("code"):
            best_code = best_simulator_info["code"]
        else:
            self.logger.warning("_build_gsim_patch_prompt: best_simulator_info has no code; using empty string")
        prompt = patch_template.replace("{completions}", best_code)

        # Build the tool spec (same as _call_llm_with_functions)
        tool_spec = {
            "type": "function",
            "name": "complete_SimStep_code",
            "description": "Write out the code for the pytorch simulator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "simulator_description_and_reasoning": {
                        "type": "string",
                        "description": "A concise description and reasoning of the code model.",
                    },
                    "SimulatorStep_code": {
                        "type": "string",
                        "description": (
                            "Code for the pytorch simulator, inclusive of the simulator definition. "
                            "If you are unsure, take your best guess. This must be a nonempty string."
                        ),
                    },
                },
                "required": ["simulator_description_and_reasoning", "SimulatorStep_code"],
            },
        }

        # Append the patch user-turn to the shared messages list
        patch_message = {
            "role": "user",
            "content": prompt,
            "function_call": {"name": "complete_SimStep_code"},
        }
        if messages is not None:
            messages.append(patch_message)
            call_messages = messages
            self.logger.info(
                f"GSIM patch: appended user message; calling Responses API with "
                f"{len(call_messages)} messages"
            )
        else:
            call_messages = [
                {"role": "system", "content": self.SYSTEM_CONTENT},
                patch_message,
            ]
            self.logger.info("GSIM patch: no shared messages list; building standalone call")

        # Call OpenAI Responses API
        try:
            import yaml as _yaml
            with open("config.yaml", "r") as _f:
                _global_cfg = _yaml.safe_load(_f)
            provider_name = _global_cfg.get("llm", {}).get("provider", "openai").lower()
        except Exception as cfg_err:
            self.logger.error(f"Could not read config.yaml: {cfg_err}")
            provider_name = "mock"

        if provider_name != "openai":
            self.logger.warning(
                f"GSIM patch: provider '{provider_name}' not openai; falling back to _call_llm"
            )
            return self._call_llm(prompt, reasoning={"effort": "medium"}), ""

        try:
            from openai import OpenAI
            from utils.llm_utils import load_api_key

            api_key = load_api_key("OPENAI_API_KEY") or \
                      _global_cfg.get("llm_providers", {}).get("openai", {}).get("api_key")
            if not api_key:
                self.logger.warning("GSIM patch: OpenAI API key not found; falling back to _call_llm")
                return self._call_llm(prompt, reasoning={"effort": "medium"}), ""

            client = OpenAI(api_key=api_key)
            provider_cfg = _global_cfg.get("llm_providers", {}).get("openai", {})
            model = provider_cfg.get("model", "gpt-4o")
            max_output_tokens = provider_cfg.get("max_output_tokens") or provider_cfg.get("max_tokens", 100000)

            responses_kwargs: Dict[str, Any] = {
                "model": model,
                "input": call_messages,
                "tools": [tool_spec],
                "tool_choice": {"type": "function", "name": "complete_SimStep_code"},
                "max_output_tokens": max_output_tokens,
                "reasoning": {"effort": "medium"},
            }
            resp = client.responses.create(**responses_kwargs)

            for item in getattr(resp, "output", []):
                if getattr(item, "type", None) == "function_call":
                    raw_args = getattr(item, "arguments", None)
                    if raw_args:
                        try:
                            args_dict = json.loads(raw_args)
                            code_out = args_dict.get("SimulatorStep_code", "")
                            desc_out = args_dict.get("simulator_description_and_reasoning", "")
                            if code_out:
                                self.logger.info(
                                    "GSIM patch: successfully extracted SimulatorStep_code"
                                )
                                return code_out, desc_out
                            return raw_args, desc_out
                        except json.JSONDecodeError:
                            return raw_args, ""

            output_text = getattr(resp, "output_text", None)
            if output_text:
                self.logger.warning("GSIM patch: model returned plain text, not a tool_call")
                return output_text, ""

            self.logger.warning("GSIM patch: no usable output from Responses API")
            return "", ""

        except Exception as exc:
            self.logger.error(f"GSIM patch: Responses API error ({exc}); falling back to _call_llm")
            return self._call_llm(prompt, reasoning={"effort": "medium"}), ""

    def _build_patch_prompt(
        self,
        task_spec: Dict[str, Any],
        previous_code: Optional[Dict[str, str]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
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
            best_simulator_info: Best simulator info for alpha/gsim mode (optional)
        
        Returns:
            Formatted prompt string
        """

        # Use the prompt template loaded from configuration via BaseAgent
        prompt_template = self.prompt_template
        if not prompt_template:
            self.logger.error("No patch prompt template loaded from config")
            raise ValueError("Patch prompt template not available. Check config.yaml for code_generation_gsim.prompt_template")
        
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
        
        # Replace placeholders (strip {playbook} since gsim does not use playbook)
        prompt_template_with_patch = prompt_template.replace("{coding_patch}", coding_patch_content)
        prompt_template_with_patch = prompt_template_with_patch.replace("{playbook}", "")
        
        prompt = prompt_template_with_patch.replace("{blue_print}", blueprint_str)
        prompt = prompt.replace("{previous_code}", previous_code_str)
        prompt = prompt.replace("{simulation_results}", simulation_results_str)
        
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

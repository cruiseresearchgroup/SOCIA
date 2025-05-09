"""
FeedbackGenerationAgent: Generates feedback for improving the simulation based on verification and evaluation results.
"""

import logging
import os
import difflib
import json
import re
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
        generated_code: Optional[Dict[str, Any]] = None,
        current_code: Optional[str] = None,
        previous_code: Optional[str] = None,
        iteration: Optional[int] = 0,
        historical_fix_log: Optional[Dict[str, Any]] = None
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
            current_code: Current iteration's code (optional)
            previous_code: Previous iteration's code (optional)
            iteration: Current iteration number
            historical_fix_log: Log of historical issues and their fix status (optional)
        
        Returns:
            Dictionary containing feedback for improvement
        """
        self.logger.info("Generating feedback for improvement")
        
        # Generate code diff if both previous and current code are available
        code_diff = None
        if previous_code and current_code:
            self.logger.info(f"Generating code diff between iterations {iteration-1} and {iteration}")
            prev_lines = previous_code.splitlines(keepends=True)
            curr_lines = current_code.splitlines(keepends=True)
            
            diff = difflib.unified_diff(
                prev_lines,
                curr_lines,
                fromfile=f"simulation_code_iter_{iteration-1}.py",
                tofile=f"simulation_code_iter_{iteration}.py",
                n=3  # Context lines
            )
            code_diff = "".join(diff)
            self.logger.info(f"Generated code diff with {len(code_diff)} characters")
        else:
            self.logger.info("No previous code available for diff generation")
        
        # First check if we need to update the historical fix log with fixed issues
        if iteration > 0 and historical_fix_log and current_code:
            self._check_fixed_issues(historical_fix_log, current_code, iteration)
        
        # Build prompt for LLM to generate feedback
        prompt = self._build_prompt(
            task_spec=task_spec,
            verification_results=verification_results,
            simulation_results=simulation_results,
            evaluation_results=evaluation_results,
            model_plan=model_plan,
            generated_code=generated_code,
            code_content=current_code,
            code_diff=code_diff
        )
        
        # Call LLM to generate feedback
        llm_response = self._call_llm(prompt)
        
        # Parse the LLM response and validate required schema fields
        parsed = self._parse_llm_response(llm_response)
        # Fallback to placeholder if feedback is missing or invalid
        if not isinstance(parsed, dict) or "summary" not in parsed:
            self.logger.warning("LLM feedback format is invalid, using placeholder feedback")
            feedback = self._create_placeholder_feedback()
        else:
            feedback = parsed
        
        self.logger.info("Feedback generation completed")
        return feedback
    
    def _check_fixed_issues(self, historical_fix_log: Dict[str, Any], current_code: str, current_iteration: int) -> None:
        """
        Check if previous issues have been fixed in the current code.
        
        Args:
            historical_fix_log: Log of historical issues and their fix status
            current_code: Current iteration's code
            current_iteration: Current iteration number
        """
        try:
            self.logger.info("Checking if previous issues have been fixed in the current code")
            
            # 创建prompt用于检查已修复的问题
            prompt = self._build_fix_check_prompt(historical_fix_log, current_code, current_iteration)
            
            # 设置最大重试次数
            max_retries = 2
            for retry in range(max_retries + 1):
                try:
                    # 调用LLM检查已修复的问题
                    llm_response = self._call_llm(prompt)
                    
                    # 添加简单验证和预处理
                    llm_response = llm_response.strip()
                    
                    # 如果响应不是以{开头，但包含迭代键，尝试修复
                    if not llm_response.startswith('{') and '"iteration_' in llm_response:
                        self.logger.warning("Response does not start with {, attempting to fix")
                        corrected_response = "{"
                        start_pos = llm_response.find('"iteration_')
                        if start_pos >= 0:
                            corrected_response += llm_response[start_pos:]
                            if not corrected_response.rstrip().endswith('}'):
                                corrected_response += "}"
                            llm_response = corrected_response
                    
                    # 解析LLM响应
                    fixed_issues = self._parse_fix_check_response(llm_response)
                    
                    # 如果解析失败并且还有重试机会，则重试
                    if not fixed_issues and retry < max_retries:
                        self.logger.warning(f"Empty response or parsing failed. Retrying ({retry+1}/{max_retries})...")
                        # 在下一次重试中添加更明确的指示
                        prompt += "\n\nIMPORTANT: Please format your response as a VALID JSON object without any preceding text or formatting. Start with '{' and end with '}'."
                        continue
                    
                    # 更新historical fix log中已修复的问题
                    if fixed_issues and isinstance(fixed_issues, dict):
                        updated_count = 0
                        for iteration_key, issues in fixed_issues.items():
                            if iteration_key in historical_fix_log:
                                for i, log_issue in enumerate(historical_fix_log[iteration_key]):
                                    # 查找对应的已修复问题
                                    issue_description = log_issue.get("issue", "")
                                    for fixed_issue in issues:
                                        if fixed_issue.get("issue") == issue_description and fixed_issue.get("status") == "fixed":
                                            historical_fix_log[iteration_key][i]["status"] = "fixed"
                                            historical_fix_log[iteration_key][i]["fixed_log"] = fixed_issue.get("fixed_log", "")
                                            self.logger.info(f"Marked issue as fixed: {issue_description}")
                                            updated_count += 1
                        self.logger.info(f"Successfully updated {updated_count} fixed issues in the historical log")
                        # 如果成功更新了问题，则可以跳出循环
                        break
                except json.JSONDecodeError as json_err:
                    if retry < max_retries:
                        self.logger.warning(f"JSON parsing error in fix check response (attempt {retry+1}/{max_retries}): {json_err}")
                        self.logger.warning(f"Response snippet: {llm_response[:200] if llm_response else 'None'}")
                    else:
                        self.logger.error(f"Final JSON parsing error in fix check response: {json_err}")
                        self.logger.error(f"Response snippet: {llm_response[:200] if llm_response else 'None'}")
                except Exception as e:
                    if retry < max_retries:
                        self.logger.warning(f"Error processing fix check response (attempt {retry+1}/{max_retries}): {e}")
                    else:
                        self.logger.error(f"Final error processing fix check response: {e}")
        except Exception as e:
            # 捕获所有异常，防止工作流崩溃
            self.logger.error(f"Error checking fixed issues: {e}")
            # 记录异常栈跟踪以便调试
            import traceback
            self.logger.error(f"Exception traceback: {traceback.format_exc()}")
            self.logger.error("Continuing without checking fixed issues")
    
    def _build_fix_check_prompt(self, historical_fix_log: Dict[str, Any], current_code: str, current_iteration: int) -> str:
        """
        Build a prompt for the LLM to check fixed issues.
        
        Args:
            historical_fix_log: Log of historical issues and their fix status
            current_code: Current iteration's code
            current_iteration: Current iteration number
            
        Returns:
            Prompt for the LLM
        """
        # Get previous iterations (all except the current one)
        previous_iterations = [k for k in historical_fix_log.keys() if k != f"iteration_{current_iteration}"]
        
        # If no previous iterations found, return empty prompt
        if not previous_iterations:
            self.logger.warning("No previous iterations found in historical fix log")
            return ""
        
        # Build historical issues string for the prompt
        historical_issues_str = ""
        for iteration_key in previous_iterations:
            # Extract iteration number from the key
            try:
                iteration_num = int(iteration_key.split("_")[1])
            except (IndexError, ValueError):
                self.logger.warning(f"Invalid iteration key format: {iteration_key}")
                continue
                
            historical_issues_str += f"Iteration {iteration_num}:\n"
            for i, issue in enumerate(historical_fix_log[iteration_key]):
                historical_issues_str += f"{i+1}. {issue.get('issue', 'Unknown issue')} (Status: {issue.get('status', 'unknown')})\n"
            historical_issues_str += "\n"
                
        # Load the fix check prompt template
        try:
            with open(os.path.join(self.template_dir, "fix_check_prompt.txt"), "r") as f:
                prompt_template = f.read()
                
            # Format the prompt - 注意这里需要将双花括号替换为单花括号，因为模板中已经使用双花括号作为转义
            prompt_template = prompt_template.replace("{{", "{").replace("}}", "}")
                
            prompt = prompt_template.format(
                historical_issues=historical_issues_str,
                code_content=current_code
            )
            
            # 安全检查，确保没有未替换的占位符
            if "{" in prompt and "}" in prompt:
                placeholders = re.findall(r'{(.*?)}', prompt)
                if placeholders:
                    self.logger.warning(f"Found unused placeholders in prompt: {placeholders}")
                    # 尝试移除未使用的占位符
                    for ph in placeholders:
                        prompt = prompt.replace(f"{{{ph}}}", "")
            
            return prompt
        except Exception as e:
            self.logger.error(f"Error building fix check prompt: {e}")
            return ""
    
    def _parse_fix_check_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the response from the LLM for fixed issues.
        
        Args:
            llm_response: Response from the LLM
            
        Returns:
            Dictionary containing fixed issues information
        """
        try:
            # 预处理：去除所有前导和尾随空白字符
            llm_response = llm_response.strip()
            
            # 提取JSON响应
            first_brace = llm_response.find('{')
            last_brace = llm_response.rfind('}')
            
            if first_brace == -1 or last_brace == -1:
                self.logger.warning("No valid JSON found in LLM response for fix check")
                # 尝试查找迭代关键字，可能JSON格式不完整
                if '"iteration_' in llm_response:
                    self.logger.info("Found iteration key but JSON is malformed, attempting to correct")
                    # 尝试构建有效的JSON
                    corrected_json = "{"
                    start_pos = llm_response.find('"iteration_')
                    if start_pos >= 0:
                        corrected_json += llm_response[start_pos:]
                        # 确保JSON以}结尾
                        if not corrected_json.rstrip().endswith('}'):
                            corrected_json += "}"
                        llm_response = corrected_json
                        first_brace = 0
                        last_brace = len(llm_response) - 1
                    else:
                        return {}
                else:
                    return {}
            
            json_str = llm_response[first_brace:last_brace+1]
            
            # 增强的JSON清理
            # 1. 删除属性之间换行和不必要的空格
            cleaned_json = re.sub(r',\s*\n\s*"', ', "', json_str)
            cleaned_json = re.sub(r'{\s*\n\s*"', '{ "', cleaned_json)
            # 2. 修复属性值之前的空格
            cleaned_json = re.sub(r'"\s*:', '": ', cleaned_json)
            # 3. 修复可能的结尾问题
            cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
            # 4. 确保布尔值正确，处理true/false的大小写
            cleaned_json = re.sub(r':\s*True', ': true', cleaned_json)
            cleaned_json = re.sub(r':\s*False', ': false', cleaned_json)
            
            # 记录清理后的JSON以便于调试
            self.logger.debug(f"Cleaned JSON: {cleaned_json[:100]}...")
            
            try:
                # 尝试解析清理后的JSON
                fixed_issues = json.loads(cleaned_json)
                
                # 过滤掉与入口点相关的问题，如果它们被标记为已修复
                # 这可以防止关于直接main()调用模式的不正确修复
                for iteration_key, issues in fixed_issues.items():
                    fixed_issues[iteration_key] = [
                        issue for issue in issues 
                        if not (
                            issue.get("status") == "fixed" and 
                            (
                                "if __name__ ==" in issue.get("issue", "") or 
                                "entry point" in issue.get("issue", "").lower() or
                                "main function" in issue.get("issue", "").lower() or
                                "main()" in issue.get("issue", "")
                            ) and
                            "guard" in issue.get("issue", "").lower()
                        )
                    ]
                
                return fixed_issues
            except json.JSONDecodeError as json_err:
                self.logger.warning(f"First JSON decode attempt failed: {json_err}")
                
                # 尝试更激进的清理
                # 1. 移除可能的多余引号和转义符
                aggressive_cleaned = re.sub(r'\\+(["\'])', r'\1', cleaned_json)
                # 2. 确保数组和对象正确嵌套
                aggressive_cleaned = re.sub(r'\]\s*\[', '], [', aggressive_cleaned)
                aggressive_cleaned = re.sub(r'}\s*{', '}, {', aggressive_cleaned)
                
                try:
                    fixed_issues = json.loads(aggressive_cleaned)
                    
                    # 过滤掉与入口点相关的问题，如果它们被标记为已修复
                    for iteration_key, issues in fixed_issues.items():
                        fixed_issues[iteration_key] = [
                            issue for issue in issues 
                            if not (
                                issue.get("status") == "fixed" and 
                                (
                                    "if __name__ ==" in issue.get("issue", "") or 
                                    "entry point" in issue.get("issue", "").lower() or
                                    "main function" in issue.get("issue", "").lower() or
                                    "main()" in issue.get("issue", "")
                                ) and
                                "guard" in issue.get("issue", "").lower()
                            )
                        ]
                    
                    return fixed_issues
                except json.JSONDecodeError:
                    # 最后的尝试 - 如果系统支持json5（更宽松的JSON解析器），可以尝试使用
                    self.logger.warning("Second JSON decode attempt failed, returning empty dict")
                    # 记录问题JSON以便调试
                    self.logger.error(f"Problematic JSON: {cleaned_json[:200]}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error parsing fix check response: {e}")
            self.logger.error(f"Original response snippet: {llm_response[:200]}")
            return {}
    
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
            "additional_comments": "Overall, the simulation shows promise but needs refinement in key areas",
            "code_snippets": []
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
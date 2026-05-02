#!/usr/bin/env python3
"""
Test script for DataAnalysisOddAgent
Tests the data analysis odd functionality by running task understanding, data analysis, and model planning steps.
"""

import argparse
import logging
import os
import re
import sys
import yaml
import json
from typing import Dict, Any, Optional
from orchestration.container import AgentContainer
from utils.llm_utils import load_api_key
from dependency_injector.wiring import Provide, inject
from core.playbook_manager import PlaybookManager

def setup_logging(output_path: Optional[str] = None, debug: bool = False):
    """Configure logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create handlers list
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # If output path is provided, add a file handler
    if output_path:
        try:
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            # Create log file path
            log_file_path = os.path.join(output_path, "test_data_analysis_odd.log")
            
            # Add file handler to handlers list
            handlers.append(logging.FileHandler(log_file_path))
            print(f"Logging to file: {log_file_path}")
        except Exception as e:
            print(f"Warning: Could not set up logging to file: {e}")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('SOCIA.DataAnalysisOddTest')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test script for SOCIA DataAnalysisOddAgent')
    parser.add_argument('--task', type=str, required=True, help='Description of the simulation task')
    parser.add_argument('--task-file', type=str, help='Path to task description JSON file')
    parser.add_argument('--output', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--mode', type=str, default='persona', choices=['lite', 'medium', 'persona', 'blueprint', 'odd', 'ace', 'alpha', 'gsim'], help='Workflow mode')
    parser.add_argument('--selfloop', type=int, default=3, help='Number of self-checking loop attempts for code generation')
    parser.add_argument('--persisted-data-analysis-file', type=str, help='Path to persisted data analysis file (task_spec.json) to skip data analysis phase')
    parser.add_argument('--persisted-code-file', type=str, help='Path to persisted code file (simulation_code_iter_N.py) to skip data analysis and initial code generation')
    parser.add_argument('--auto', action='store_true', default=False, help='Enable automatic mode; when False, user will be prompted to input feedback manually in each iteration')
    parser.add_argument('--iterations', type=int, default=3, help='Maximum number of iterations')
    
    return parser.parse_args()

def setup_container(config_path: str) -> AgentContainer:
    """Set up and configure the dependency injection container."""
    container = AgentContainer()
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            container.config.from_dict(config)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        # Use minimal default configuration
        container.config.from_dict({
            "system": {"name": "SOCIA", "version": "0.1.0"},
            "agents": {
                "task_understanding": {"prompt_template": "templates/task_understanding_prompt.txt"},
                "data_analysis_odd": {"prompt_template": "templates/data_analysis_odd_prompt.txt"},
                "code_generation_odd": {"prompt_template": "templates/code_generation_odd_prompt.txt"},
                "model_planning": {"prompt_template": "templates/model_planning_prompt.txt"},
                "code_verification": {"prompt_template": "templates/code_verification_prompt.txt"},
                "simulation_execution": {"prompt_template": "templates/simulation_execution_prompt.txt"},
                "result_evaluation": {"prompt_template": "templates/result_evaluation_prompt.txt"},
                "feedback_generation": {"prompt_template": "templates/feedback_generation_prompt.txt"},
                "feedback_generation_odd": {"prompt_template": "templates/feedback_generation_prompt.txt"},
                "iteration_control": {"prompt_template": "templates/iteration_control_prompt.txt"}
            }
        })
    
    # Wire the container for dependency injection
    container.wire(modules=[
        sys.modules[__name__],
        "agents.task_understanding.agent",
        "agents.data_analysis_odd.agent",
        "agents.data_analysis_ace.agent",
        "agents.code_generation_odd.agent",
        "agents.code_generation_ace.agent",
        "agents.code_generation_alpha.agent",
        "agents.model_planning.agent",
        "agents.code_verification.agent",
        "agents.simulation_execution.agent",
        "agents.simulation_execution_alpha.agent",
        "agents.simulation_execution_gsim.agent",
        "agents.result_evaluation.agent",
        "agents.feedback_generation.agent",
        "agents.feedback_generation_odd.agent",
        "agents.feedback_generation_alpha.agent",
        "agents.iteration_control.agent",
        "agents.iteration_control_ace.agent",
        "agents.base_agent",
        "utils.llm_utils"
    ])
    
    return container

def load_task_data(task_file_path: str) -> Optional[Dict[str, Any]]:
    """Load task data from JSON file."""
    if not task_file_path or not os.path.exists(task_file_path):
        return None
    
    try:
        with open(task_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading task file {task_file_path}: {e}")
        return None

def extract_data_path_from_task_file(task_file_path: str, task_data: Dict[str, Any]) -> Optional[str]:
    """Extract data path from task file, resolving relative paths."""
    if not task_data or "data_folder" not in task_data:
        return None
    
    data_folder = task_data["data_folder"]
    
    # If data_folder is relative, make it relative to the task file directory
    if not os.path.isabs(data_folder):
        task_file_dir = os.path.dirname(os.path.abspath(task_file_path))
        project_root = os.getcwd()  # Assuming we're running from project root
        data_path = os.path.join(project_root, data_folder)
    else:
        data_path = data_folder
    
    return data_path

def save_artifact(output_path: str, name: str, data: Any, iteration: Optional[int] = None):
    """Save an artifact to the output directory.
    
    Args:
        output_path: Directory to save the artifact
        name: Base name for the artifact (without iteration suffix)
        data: Data to save
        iteration: Optional iteration number. If None and name doesn't contain _iter_, defaults to 0.
                   If name already contains _iter_X, uses that iteration number.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        
        # Check if name already contains _iter_ pattern
        iter_pattern = r'_iter_(\d+)$'
        match = re.search(iter_pattern, name)
        
        if match:
            # Name already contains iteration number, use it directly
            file_path = os.path.join(output_path, f"{name}.json")
        elif iteration is not None:
            # Use provided iteration number
            file_path = os.path.join(output_path, f"{name}_iter_{iteration}.json")
        else:
            # Default to iter_0
            file_path = os.path.join(output_path, f"{name}_iter_0.json")
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logging.info(f"Saved {name} to {file_path}")
    except Exception as e:
        logging.error(f"Error saving {name}: {e}")

def save_generated_code(output_path: str, generated_code: Dict[str, Any], iteration: int = 0):
    """Save generated code using dual persistence mechanism (JSON + Python file)."""
    logger = logging.getLogger()
    
    try:
        os.makedirs(output_path, exist_ok=True)
        
        # 1. Save complete generated code data as JSON (includes metadata)
        json_file_path = os.path.join(output_path, f"generated_code_iter_{iteration}.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(generated_code, f, indent=2, default=str)
        logger.info(f"Saved generated code JSON to {json_file_path}")
        
        # 2. Save pure Python code as executable .py file
        if "code" in generated_code:
            code_content = generated_code["code"]
            code_file_path = os.path.join(output_path, f"simulation_code_iter_{iteration}.py")
            with open(code_file_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
            logger.info(f"Saved Python code to {code_file_path}")
        else:
            logger.warning("No 'code' field found in generated_code data")
            
    except Exception as e:
        logger.error(f"Error saving generated code: {e}")
        raise

def load_persisted_data_analysis(file_path: str) -> Dict[str, Any]:
    """Load persisted data analysis from JSON file."""
    logger = logging.getLogger()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            task_spec = json.load(f)
        logger.info(f"Loaded persisted data analysis from: {file_path}")
        return task_spec
    except Exception as e:
        logger.error(f"Error loading persisted data analysis file {file_path}: {e}")
        raise

def load_persisted_code(file_path: str) -> str:
    """Load persisted code from Python file."""
    logger = logging.getLogger()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        logger.info(f"Loaded persisted code from: {file_path}")
        return code
    except Exception as e:
        logger.error(f"Error loading persisted code file {file_path}: {e}")
        raise

def regenerate_blueprint_from_task_file(
    data_analysis_agent,
    task_spec: Dict[str, Any],
    task_file_path: str,
    logger
) -> Dict[str, Any]:
    """
    Rebuild the data analysis blueprint by feeding the full task file content
    into the personality-specific prompt template.
    """
    if not task_file_path or not os.path.exists(task_file_path):
        raise FileNotFoundError(f"Task file not found: {task_file_path}")
    
    template_path = data_analysis_agent.config.get(
        "prompt_template",
        "templates/data_analysis_odd_prompt_personality.txt"
    )
    if not os.path.isabs(template_path):
        template_path = os.path.abspath(template_path)
    
    logger.info("Full mode: Regenerating blueprint using personality template")
    logger.info(f"  → Template: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    with open(task_file_path, 'r', encoding='utf-8') as f:
        task_file_content = f.read()
    
    file_summaries = task_spec.get("file_summaries", [])
    if file_summaries:
        file_summaries_text = json.dumps(file_summaries, indent=2, default=str)
    else:
        file_summaries_text = "No file summaries available"
    
    prompt = prompt_template.replace("{task_file}", task_file_content)
    prompt = prompt.replace("{file_summaries_text}", file_summaries_text)
    
    llm_response = data_analysis_agent._call_llm(prompt, reasoning={"effort": "medium"})
    logger.info("Full mode: Received blueprint response from LLM")
    
    analysis_results = data_analysis_agent._parse_llm_analysis(llm_response)
    if not isinstance(analysis_results, dict) or not analysis_results:
        raise ValueError("Blueprint regeneration returned empty analysis results")
    
    return analysis_results


def get_user_feedback(
    logger,
    iteration: int,
    verification_results: Optional[Dict[str, Any]] = None,
    simulation_results: Optional[Dict[str, Any]] = None,
    evaluation_results: Optional[Dict[str, Any]] = None,
    generated_code: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prompt user for manual feedback input, showing current iteration summary first.
    """
    # Check if running in an interactive terminal
    import sys
    is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    
    if not is_interactive:
        logger.warning("=" * 80)
        logger.warning("WARNING: Running in non-interactive mode - cannot prompt for user feedback")
        logger.warning("=" * 80)
        logger.warning("The program is running in a non-interactive environment (e.g., background process, redirected output).")
        logger.warning("User feedback input is disabled. Using system-generated feedback only.")
        logger.warning("=" * 80)
        logger.warning("To enable user feedback, run the script in an interactive terminal.")
        logger.warning("=" * 80)
        return ""
    
    # First show the iteration summary
    display_iteration_summary(logger, iteration, verification_results, simulation_results, evaluation_results, generated_code)
    
    # Ensure output is flushed before waiting for input
    sys.stdout.flush()
    sys.stderr.flush()
    
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
    
    # Ensure output is flushed before waiting for input
    sys.stdout.flush()
    
    feedback_lines = []
    empty_line_count = 0
    
    try:
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
            except EOFError:
                # EOFError means no input is available (non-interactive environment)
                logger.warning("EOFError: No input available - non-interactive environment detected")
                logger.warning("Skipping user feedback input. Using system-generated feedback only.")
                break
            except KeyboardInterrupt:
                print("\nFeedback input interrupted by user.")
                logger.info("User interrupted feedback input")
                break
    except Exception as e:
        logger.error(f"Unexpected error during feedback input: {e}")
        logger.error("Skipping user feedback input. Using system-generated feedback only.")
    
    user_feedback = "\n".join(feedback_lines).strip()
    
    if user_feedback:
        print("-"*80)
        print("Your feedback has been recorded:")
        print(user_feedback)
        print("="*80)
        logger.info(f"User feedback received: {len(user_feedback)} characters")
    else:
        print("-"*80)
        print("No feedback provided. Using system-generated feedback only.")
        print("="*80)
        logger.info("No user feedback provided - using system feedback only")
    
    return user_feedback

def get_blueprint_feedback(
    logger,
    iteration: int,
    blueprint: Dict[str, Any]
) -> str:
    """
    Prompt user for blueprint feedback input.
    
    Args:
        logger: Logger instance
        iteration: Current iteration number
        blueprint: Current blueprint dictionary
    
    Returns:
        Blueprint feedback text (empty string if user skips)
    """
    # Check if running in an interactive terminal
    import sys
    is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    
    if not is_interactive:
        logger.warning("=" * 80)
        logger.warning("WARNING: Running in non-interactive mode - cannot prompt for blueprint feedback")
        logger.warning("=" * 80)
        logger.warning("Blueprint feedback input is disabled. Using current blueprint as-is.")
        logger.warning("=" * 80)
        return ""
    
    # Display current blueprint summary
    print("\n" + "="*80)
    print("CURRENT BLUEPRINT SUMMARY")
    print("="*80)
    blueprint_str = json.dumps(blueprint, indent=2, ensure_ascii=False)
    # Truncate if too long
    if len(blueprint_str) > 2000:
        blueprint_str = blueprint_str[:2000] + "\n... (truncated, full blueprint is available in task_spec)"
    print(blueprint_str)
    print("="*80)
    
    # Ensure output is flushed before waiting for input
    sys.stdout.flush()
    sys.stderr.flush()
    
    print("\n" + "="*80)
    print("BLUEPRINT FEEDBACK INPUT")
    print("="*80)
    print("Do you think there are issues with the current simulator structure (blueprint design)?")
    print("If you want to modify the blueprint, please enter your feedback below.")
    print("Otherwise, just press Enter twice to skip (keep current blueprint).")
    print("\nYour feedback should describe:")
    print("- What aspects of the blueprint need to be changed")
    print("- How the blueprint should be modified")
    print("- Any new requirements or constraints")
    print("\nEnter your blueprint feedback (press Enter twice to finish):")
    print("-"*80)
    
    # Ensure output is flushed before waiting for input
    sys.stdout.flush()
    
    feedback_lines = []
    empty_line_count = 0
    
    try:
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
            except EOFError:
                logger.warning("EOFError: No input available - non-interactive environment detected")
                logger.warning("Skipping blueprint feedback input. Using current blueprint as-is.")
                break
            except KeyboardInterrupt:
                print("\nBlueprint feedback input interrupted by user.")
                logger.info("User interrupted blueprint feedback input")
                break
    except Exception as e:
        logger.error(f"Unexpected error during blueprint feedback input: {e}")
        logger.error("Skipping blueprint feedback input. Using current blueprint as-is.")
    
    blueprint_feedback = "\n".join(feedback_lines).strip()
    
    if blueprint_feedback:
        print("-"*80)
        print("Your blueprint feedback has been recorded:")
        print(blueprint_feedback)
        print("="*80)
        logger.info(f"Blueprint feedback received: {len(blueprint_feedback)} characters")
    else:
        print("-"*80)
        print("No blueprint feedback provided. Keeping current blueprint unchanged.")
        print("="*80)
        logger.info("No blueprint feedback provided - keeping current blueprint")
    
    return blueprint_feedback

def update_blueprint_from_feedback(
    logger,
    blueprint: Dict[str, Any],
    blueprint_feedback: str,
    agent_container: AgentContainer
) -> Dict[str, Any]:
    """
    Update blueprint based on user feedback using LLM.
    
    Args:
        logger: Logger instance
        blueprint: Current blueprint dictionary
        blueprint_feedback: User's blueprint feedback text
        agent_container: Agent container for accessing LLM utilities
    
    Returns:
        Updated blueprint dictionary
    """
    logger.info("Updating blueprint based on user feedback")
    
    # Build the blueprint update prompt
    blueprint_str = json.dumps(blueprint, indent=2, ensure_ascii=False)
    
    # Get blueprint schema from template
    blueprint_schema = """{
  "overall_simulation_design": {
    "objective": "what is this simulation about",
    "initialization": {
      "description": "How the simulation starts (states, seeds, signals)",
      "source_type": "data_derived|designed|mixed|unknown",
      "data_reference": {"file": "filename", "fields": ["field_a"], "derivation": "how derived"}
    },
    "execution": "How to tune parameters through data calibration",
    "outputs": ["calibrated_parameters", "evaluation_results_on_validation"]
  },
  "scale_granularity": {
    "time_step": {"value": "seconds|minutes|hours|days", "source_type": "data_derived|designed|mixed|unknown", "data_reference": null},
    "spatial_resolution": {"value": "non-spatial|grid|POI|road network|other:<...>", "source_type": "data_derived|designed|mixed|unknown", "data_reference": null},
    "population_size": {"value": "integer or description tied to data", "source_type": "data_derived|designed|mixed|unknown", "data_reference": null},
    "rationale": "Why these scales are appropriate"
  },
  "agent_archetypes": {
    "unit": "e.g., simulated_entity|user|household|device",
    "roles": [
      {
        "name": "role_name_derived_from_task",
        "static_attributes": [
          {
            "name": "attr_name",
            "type": "int|float|string|enum|vector|other",
            "source_type": "data_derived|designed|mixed|unknown",
            "data_reference": {"file": "filename", "fields": ["field_name"], "derivation": "mapping/aggregation rule"}
          }
        ],
        "dynamic_states": [
          {
            "name": "state_name",
            "type": "int|float|string|enum|vector|other",
            "update_rule": "how it updates",
            "source_type": "data_derived|designed|mixed|unknown",
            "data_reference": {"file": "filename", "fields": ["field_name"], "derivation": "mapping/aggregation rule"}
          }
        ],
        "construction_from_data": "Explain construction; if designed, explain assumption",
        "update_from_data": "Explain updates; if designed, explain assumption"
      }
    ]
  },
  "interaction_topology": {
    "topology": "graph|hybrid|broadcast|platform-level",
    "layers": [
      {
        "name": "layer_name",
        "from_role": "role_name",
        "to_role": "role_name",
        "edge_rule": "how nodes/edges/events are formed",
        "source_type": "data_derived|designed|mixed|unknown",
        "data_reference": {"file": "filename", "fields": ["src_id", "dst_id"], "derivation": "edge construction rule"}
      }
    ],
    "protocol": "describe message passing/order among these roles"
  },
  "information_propagation": {
    "exists": true,
    "topology": "which roles/channels are used for diffusion",
    "mechanism": "how diffusion works",
    "source_type": "data_derived|designed|mixed|unknown",
    "data_reference": {"file": "filename", "fields": ["field_name"], "derivation": "drive intensity/schedule"}
  },
  "exogenous_signals": [
    {
      "name": "signal_name",
      "effect_on_agents": "how it enters decision function",
      "bounds": "[low, high] or rationale",
      "source_type": "data_derived|designed|mixed|unknown",
      "data_reference": {"file": "filename", "fields": ["field_name"], "derivation": "how derived"}
    }
  ],
  "action_decision_policy": {
    "by_role": [
      {
        "role": "role_name_derived_from_task",
        "inputs": [
          {
            "name": "obs_or_signal_name",
            "source_type": "data_derived|designed|mixed|unknown",
            "data_reference": {"file": "filename", "fields": ["field_name"], "derivation": "how computed"}
          }
        ],
        "policy_form": "policy description",
        "parameters": ["list of per-role parameter names (if any)"]
      }
    ]
  },
  "capability_realization": [
    {
      "role": "role_name_derived_from_task",
      "modes": ["heuristic_rules", "tool_calls", "llm_calls"],
      "llm_prompt_skeleton": "If llm_calls is used, provide a high-level prompt template with placeholders",
      "tool_dependencies": ["list of tool/data lookup functions (only cite files/fields when data-derived)"],
      "fallback_strategy": "what to do if LLM/tool is unavailable",
      "logging": "what intermediate results are written back to shared memory/log"
    }
  ],
  "holdout_plan": {
    "method": "temporal_holdout|random_split|rolling_backtest",
    "time_ordering": {
      "source_type": "data_derived|designed|mixed|unknown",
      "data_reference": {"file": "filename", "fields": ["time_field"], "derivation": "ordering rule"}
    },
    "train_range": "rule to compute train set",
    "validation_range": "rule to compute validation set",
    "notes": "split notes"
  },
  "simulation_evaluation": {
    "metrics": [
      {
        "name": "metric_name",
        "definition": "how to compute it",
        "ground_truth": {
          "source_type": "data_derived|designed|mixed|unknown",
          "data_reference": {"file": "filename", "fields": ["target_field"], "derivation": "target extraction"}
        }
      }
    ],
    "comparison_method": "how to compute metrics on validation set and report"
  },
  "calibratable_parameters": [
    {
      "name": "parameter_name",
      "range_bounds": "[low, high] or rationale",
      "source": "constrained by data or modeling",
      "notes": "tie to interaction or decision speed"
    }
  ],
  "llm_and_tool_specs": {
    "llm_required": true,
    "llm_calls": [
      {
        "name": "llm_call_name_derived_from_role",
        "inputs": ["list of fields/placeholders to inject"],
        "output": "expected JSON/structured output"
      }
    ],
    "tool_wrappers": [
      {
        "name": "tool_function_name",
        "input_type": "string|object",
        "output_type": "structured_json"
      }
    ]
  }
}"""
    
    prompt = f"""SYSTEM:
You are the Blueprint Reflector in a system that generates and iteratively improves social simulation code. You are an expert simulation designer and algorithm/specification specialist.

Hard rules:
- BLUEPRINT FEEDBACK is authoritative for how the BLUEPRINT should be revised. If it conflicts with the current BLUEPRINT, you MUST update the BLUEPRINT to match the feedback.
- Do NOT invent new data files/fields/APIs unless BLUEPRINT FEEDBACK explicitly asks for them or the current BLUEPRINT already implies them.
- If BLUEPRINT FEEDBACK is ambiguous or underspecified, make the smallest change that plausibly satisfies it and mark affected parts with source_type="unknown" and include a brief assumption in existing text fields (e.g., description/update_rule/definition).
- Preserve the exact BLUEPRINT schema and overall structure. Output MUST be valid JSON only. No extra top-level keys.

INPUTS:
========================
BLUEPRINT (CURRENT, baseline)
========================
{blueprint_str}

========================
BLUEPRINT FEEDBACK (AUTHORITATIVE)
========================
{blueprint_feedback}

TASK:
Revise the BLUEPRINT according to BLUEPRINT FEEDBACK and output the updated BLUEPRINT in the same schema.

You MUST do the following in order:

Step 0) Parse and normalize the feedback (MANDATORY)
- Break BLUEPRINT FEEDBACK into a list of concrete change requests.
- For each change request, decide whether it affects:
  - overall_simulation_design
  - scale_granularity
  - agent_archetypes (roles/attributes/states)
  - interaction_topology
  - information_propagation
  - exogenous_signals
  - action_decision_policy
  - capability_realization
  - holdout_plan
  - simulation_evaluation
  - calibratable_parameters
  - llm_and_tool_specs

Step 1) Apply minimal edits (MANDATORY)
- Update only the necessary fields to satisfy the feedback.
- Keep unrelated parts unchanged.
- Maintain internal consistency across sections:
  * overall_simulation_design ↔ roles ↔ topology ↔ policies ↔ evaluation ↔ calibratable_parameters ↔ llm/tool specs

Step 2) Source typing for modified elements (MANDATORY)
- For any element you modify (attribute/state/signal/metric/initialization/time ordering):
  - If directly supported by existing blueprint mappings, set source_type="data_derived" or "mixed" and keep/adjust data_reference.
  - If requested by feedback but not grounded in existing mappings, set source_type="designed" and set data_reference=null, with a short assumption in the closest existing text field (description/update_rule/definition).
  - If uncertain due to missing info, set source_type="unknown" and add a short assumption.

Step 3) Validation checks (MANDATORY)
- Ensure the output is valid JSON.
- Ensure the output strictly conforms to the schema below.
- Do NOT add any extra top-level keys (no "decision", no "analysis", no "diff").
- Do NOT include any text outside JSON.

OUTPUT REQUIREMENT:
- Output ONLY the updated BLUEPRINT JSON object using the schema below.
- If no changes are needed, output the original BLUEPRINT unchanged (but still valid JSON).

========================
OUTPUT SCHEMA (valid JSON only)
========================
{blueprint_schema}

Return only valid JSON that can be parsed. Do not include any other explanation or text outside the JSON."""
    
    # Use BaseAgent to call LLM
    from agents.base_agent import BaseAgent
    temp_agent = BaseAgent(config={"prompt_template": "", "output_format": "json"})
    
    try:
        # Call LLM
        llm_response = temp_agent._call_llm(prompt)
        
        # Parse JSON response
        llm_response = llm_response.strip()
        
        # Extract JSON from response
        first_brace = llm_response.find('{')
        last_brace = llm_response.rfind('}')
        
        if first_brace == -1 or last_brace == -1:
            logger.error("No valid JSON found in LLM response for blueprint update")
            logger.error(f"Response snippet: {llm_response[:500]}")
            return blueprint  # Return original blueprint if parsing fails
        
        json_str = llm_response[first_brace:last_brace+1]
        updated_blueprint = json.loads(json_str)
        
        logger.info("Blueprint successfully updated based on user feedback")
        return updated_blueprint
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in blueprint update response: {e}")
        logger.error(f"Response snippet: {llm_response[:500] if 'llm_response' in locals() else 'N/A'}")
        return blueprint  # Return original blueprint if parsing fails
    except Exception as e:
        logger.error(f"Error updating blueprint from feedback: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return blueprint  # Return original blueprint on error

def display_iteration_summary(
    logger,
    iteration: int,
    verification_results: Optional[Dict[str, Any]] = None,
    simulation_results: Optional[Dict[str, Any]] = None,
    evaluation_results: Optional[Dict[str, Any]] = None,
    generated_code: Optional[Dict[str, Any]] = None
):
    """Display a summary of the current iteration's execution status."""
    print("\n" + "="*80)
    print(f"ITERATION {iteration + 1} EXECUTION SUMMARY")
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
        if evaluation_results.get("placeholder"):
            print("   ⚠️  No real evaluation data available (placeholder).")
        else:
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

@inject
def run_data_analysis_test(
    args,
    logger,
    agent_container: AgentContainer = Provide[AgentContainer]
):
    """Run the data analysis test with multi-iteration support (ODD/ACE mode + lite workflow)."""
    
    logger.info(f"Starting DataAnalysisOddAgent test in {args.mode.upper()} mode")
    logger.info(f"Auto mode: {args.auto}, Max iterations: {args.iterations}")
    
    try:
        # Set up output path in container
        agent_container.output_path.override(args.output)

        # Initialize playbook for ACE/ODD-style workflows
        # Playbook is shared across all tasks, stored in project root /playbook_storage
        playbook = None
        playbook_manager = None
        if getattr(args, "mode", None) in ["odd", "ace", "alpha"]:
            # Use default storage root (project_root/playbook_storage)
            playbook_manager = PlaybookManager()
            playbook = playbook_manager.playbook
            logger.info(f"Playbook loaded/initialized at: {playbook_manager.playbook_path}")
        
        # Get agent instances - include all agents needed for iterative workflow
        # Select agents based on mode
        if args.mode == "ace":
            data_analysis_agent = agent_container.data_analysis_ace_agent()
            code_generation_agent = agent_container.code_generation_ace_agent()
            simulation_execution_agent = agent_container.simulation_execution_ace_agent()
            feedback_generation_agent = agent_container.feedback_generation_ace_agent()
            iteration_control_agent = agent_container.iteration_control_ace_agent()
        elif args.mode == "alpha":
            data_analysis_agent = agent_container.data_analysis_ace_agent()
            # For alpha mode, override the prompt template to use data_analysis_alpha_prompt.txt
            data_analysis_agent.config["prompt_template"] = "templates/data_analysis_alpha_prompt.txt"
            # Reload the prompt template
            data_analysis_agent.prompt_template = data_analysis_agent._load_prompt_template()
            logger.info("Alpha mode: Using data_analysis_alpha_prompt.txt for data_analysis_ace_agent")
            code_generation_agent = agent_container.code_generation_alpha_agent()
            logger.info("Alpha mode: Using code_generation_alpha_agent for code generation")
            simulation_execution_agent = agent_container.simulation_execution_alpha_agent()
            logger.info("Alpha mode: Using simulation_execution_alpha_agent for execution")
            feedback_generation_agent = agent_container.feedback_generation_alpha_agent()
            logger.info("Alpha mode: Using feedback_generation_alpha_agent for feedback generation")
            iteration_control_agent = agent_container.iteration_control_ace_agent()
        elif args.mode == "gsim":
            data_analysis_agent = agent_container.data_analysis_gsim_agent()
            logger.info("Gsim mode: Using data_analysis_gsim_agent for data analysis")
            code_generation_agent = agent_container.code_generation_gsim_agent()
            logger.info("Gsim mode: Using code_generation_gsim_agent for code generation")
            simulation_execution_agent = agent_container.simulation_execution_gsim_agent()
            logger.info("Gsim mode: Using simulation_execution_gsim_agent for execution")
            feedback_generation_agent = agent_container.feedback_generation_gsim_agent()
            logger.info("Gsim mode: Using feedback_generation_gsim_agent for feedback generation")
            iteration_control_agent = agent_container.iteration_control_ace_agent()
        else:
            # Default to odd agents for odd mode and other modes
            data_analysis_agent = agent_container.data_analysis_odd_agent()
            code_generation_agent = agent_container.code_generation_odd_agent()
            simulation_execution_agent = agent_container.simulation_execution_agent()
            feedback_generation_agent = agent_container.feedback_generation_odd_agent()  # Use odd agent for odd mode
            iteration_control_agent = agent_container.iteration_control_agent()

        agents = {
            "data_analysis": data_analysis_agent,
            "code_generation": code_generation_agent,
            "code_verification": agent_container.code_verification_agent(),
            "simulation_execution": simulation_execution_agent,
            "result_evaluation": agent_container.result_evaluation_agent(),
            "feedback_generation": feedback_generation_agent,
            "iteration_control": iteration_control_agent
        }
        
        # Initialize state management (similar to workflow_manager)
        state = {
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
        code_memory = {}
        state["code_memory"] = code_memory
        
        # Initialize code description memory to store simulator_description per iteration
        # Mirrors code_memory shape: {iteration: {artifact_name: content}}
        code_description_memory = {}
        state["code_description_memory"] = code_description_memory
        
        # Initialize historical fix log
        historical_fix_log = {}
        
        # Initialize current iteration counter
        current_iteration = 0
        
        # Variables to track if we're skipping initial phases
        skip_data_analysis = False
        skip_initial_code_generation = False
        persisted_iteration_number = None  # Track which iteration was loaded from persisted code
        
        # ==================================================
        # PHASE 1: Data Analysis or Load Persisted Data
        # ==================================================
        
        # Check if we should load persisted code (skip data analysis AND initial code generation)
        if hasattr(args, 'persisted_code_file') and getattr(args, 'persisted_code_file', None):
            logger.info("=" * 50)
            logger.info("LOADING PERSISTED CODE")
            logger.info("=" * 50)
            
            # Extract iteration number from filename (e.g., simulation_code_iter_1.py -> 1)
            code_filename = os.path.basename(args.persisted_code_file)
            iteration_match = re.search(r'simulation_code_iter_(\d+)\.py', code_filename)
            if iteration_match:
                persisted_iteration = int(iteration_match.group(1))
                logger.info(f"Detected iteration number {persisted_iteration} from filename: {code_filename}")
            else:
                # Default to iteration 0 if pattern doesn't match
                persisted_iteration = 0
                logger.warning(f"Could not extract iteration number from filename '{code_filename}', defaulting to iteration 0")
            
            # Load the persisted code
            persisted_code = load_persisted_code(args.persisted_code_file)
            
            # Also need to load task_spec
            if hasattr(args, 'persisted_data_analysis_file') and getattr(args, 'persisted_data_analysis_file', None):
                task_spec = load_persisted_data_analysis(args.persisted_data_analysis_file)
            else:
                # Try to find task_spec in the same directory as persisted code
                code_dir = os.path.dirname(args.persisted_code_file)
                # Try to find task_spec for the same iteration first, then fall back to iter_0
                task_spec_file = os.path.join(code_dir, f"task_spec_iter_{persisted_iteration}.json")
                if not os.path.exists(task_spec_file):
                    task_spec_file = os.path.join(code_dir, "task_spec_iter_0.json")
                if os.path.exists(task_spec_file):
                    task_spec = load_persisted_data_analysis(task_spec_file)
                    logger.info(f"Auto-loaded task_spec from: {task_spec_file}")
                else:
                    logger.error("Cannot find task_spec file. Please provide --persisted-data-analysis-file")
                    return {"status": "failed", "error": "Missing task_spec", "output_path": args.output}
            
            # Extract data path
            data_path = None
            if "data_folder" in task_spec:
                data_folder = task_spec["data_folder"]
                if os.path.isabs(data_folder):
                    data_path = data_folder
                else:
                    data_path = os.path.abspath(os.path.join(os.getcwd(), data_folder))
                logger.info(f"Data path from persisted file: {data_path}")
            
            # Create initial generated_code dict from persisted code
            state["generated_code"] = {
                "code": persisted_code,
                "code_summary": f"Loaded persisted code ({len(persisted_code)} characters)",
                "metadata": {
                    "model_type": args.mode if hasattr(args, 'mode') else "odd",
                    "mode": args.mode if hasattr(args, 'mode') else "odd",
                    "source": "persisted",
                    "iteration": persisted_iteration
                }
            }
            
            # For persisted start, load simulator_description from generated_code_iter_<persisted_iteration>.json
            # instead of regenerating it from code.
            if args.mode in ["alpha", "gsim"]:
                persisted_generated_json = os.path.join(
                    args.output, f"generated_code_iter_{persisted_iteration}.json"
                )
                persisted_sim_desc = ""
                if os.path.exists(persisted_generated_json):
                    try:
                        with open(persisted_generated_json, "r", encoding="utf-8") as gf:
                            persisted_generated_payload = json.load(gf)
                        persisted_sim_desc = (
                            persisted_generated_payload.get("simulator_description", "") or ""
                        )
                        logger.info(
                            f"Loaded simulator_description from: {persisted_generated_json}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load simulator_description from {persisted_generated_json}: {e}"
                        )
                else:
                    logger.warning(
                        f"generated_code JSON not found for persisted iteration: {persisted_generated_json}"
                    )
                state["generated_code"]["simulator_description"] = persisted_sim_desc
            
            # Store in code_memory at the correct iteration number
            code_memory[persisted_iteration] = {f"simulation_code_iter_{persisted_iteration}.py": persisted_code}
            persisted_sim_desc = state["generated_code"].get("simulator_description", "")
            code_description_memory[persisted_iteration] = {
                f"simulator_description_iter_{persisted_iteration}.txt": persisted_sim_desc
            }
            
            # Save the loaded code to output directory (if it doesn't exist already)
            persisted_code_path = os.path.join(args.output, f"simulation_code_iter_{persisted_iteration}.py")
            if not os.path.exists(persisted_code_path):
                save_generated_code(args.output, state["generated_code"], iteration=persisted_iteration)
                logger.info(f"Saved persisted code to: {persisted_code_path}")
            
            skip_data_analysis = True
            skip_initial_code_generation = True
            current_iteration = persisted_iteration  # Use the iteration number from filename
            persisted_iteration_number = persisted_iteration  # Record which iteration was loaded
            
            # ==========================================
            # Restore necessary state for continuation
            # ==========================================
            
            # 1. Load previous iterations' code (for feedback generation comparison)
            # This is CRITICAL: feedback generation needs to compare current vs previous code
            if persisted_iteration > 0:
                logger.info(f"Loading code from previous iterations (0 to {persisted_iteration-1}) for state restoration...")
                for prev_iter in range(persisted_iteration):
                    prev_code_file = os.path.join(args.output, f"simulation_code_iter_{prev_iter}.py")
                    if os.path.exists(prev_code_file):
                        try:
                            with open(prev_code_file, 'r', encoding='utf-8') as f:
                                prev_code = f.read()
                            code_memory[prev_iter] = {f"simulation_code_iter_{prev_iter}.py": prev_code}
                            # Restore simulator_description memory (best-effort) from generated_code artifact.
                            prev_generated_json = os.path.join(args.output, f"generated_code_iter_{prev_iter}.json")
                            prev_sim_desc = ""
                            if os.path.exists(prev_generated_json):
                                try:
                                    with open(prev_generated_json, 'r', encoding='utf-8') as gf:
                                        prev_generated_payload = json.load(gf)
                                    prev_sim_desc = prev_generated_payload.get("simulator_description", "") or ""
                                except Exception as desc_err:
                                    logger.warning(f"  ✗ Failed to load simulator_description from iteration {prev_iter}: {desc_err}")
                            code_description_memory[prev_iter] = {
                                f"simulator_description_iter_{prev_iter}.txt": prev_sim_desc
                            }
                            logger.info(f"  ✓ Loaded code from iteration {prev_iter}")
                        except Exception as e:
                            logger.warning(f"  ✗ Failed to load code from iteration {prev_iter}: {e}")
                    else:
                        logger.warning(f"  ✗ Code file not found: {prev_code_file}")
            
            # 2. Load historical fix log (for future iterations' code generation)
            # This helps avoid repeating the same mistakes in subsequent iterations
            historical_fix_log_file = os.path.join(args.output, "historical_fix_log.json")
            if os.path.exists(historical_fix_log_file):
                try:
                    with open(historical_fix_log_file, 'r', encoding='utf-8') as f:
                        historical_fix_log = json.load(f)
                    logger.info(f"  ✓ Loaded historical fix log: {len(historical_fix_log)} entries")
                except Exception as e:
                    logger.warning(f"  ✗ Failed to load historical fix log: {e}")
                    historical_fix_log = {}
            else:
                logger.info(f"  ℹ No historical fix log found, starting with empty log")
                historical_fix_log = {}
            
            logger.info(f"State restoration complete. Ready to continue from iteration {persisted_iteration}.")
            logger.info(f"Persisted code loaded successfully as iter_{persisted_iteration}, will skip code generation and continue with simulation execution and feedback generation")
            
        elif hasattr(args, 'persisted_data_analysis_file') and getattr(args, 'persisted_data_analysis_file', None):
            # Skip only data analysis, but still do initial code generation
            logger.info("=" * 50)
            logger.info("LOADING PERSISTED DATA ANALYSIS")
            logger.info("=" * 50)
            
            task_spec = load_persisted_data_analysis(args.persisted_data_analysis_file)
            
            # Extract data path from task_spec if available
            data_path = None
            if "data_folder" in task_spec:
                data_folder = task_spec["data_folder"]
                if os.path.isabs(data_folder):
                    data_path = data_folder
                else:
                    data_path = os.path.abspath(os.path.join(os.getcwd(), data_folder))
                logger.info(f"Data path from persisted file: {data_path}")
                
            skip_data_analysis = True
            logger.info("Persisted data analysis loaded successfully")
            
        else:
            # Regular data analysis workflow
            task_data = None
            data_path = None
            
            if args.task_file:
                task_data = load_task_data(args.task_file)
                if task_data:
                    data_path = extract_data_path_from_task_file(args.task_file, task_data)
                    logger.info(f"Loaded task data from: {args.task_file}")
                    logger.info(f"Data path resolved to: {data_path}")
                else:
                    logger.warning(f"Could not load task data from: {args.task_file}")
            
            # Data Analysis (generates task_spec)
            logger.info("=" * 50)
            logger.info("DATA ANALYSIS WITH TASK SPEC GENERATION")
            logger.info("=" * 50)
            
            if task_data:
                task_spec = agents["data_analysis"].process(
                    task_description=args.task,
                    task_data=task_data,
                    mode=args.mode
                )
            else:
                task_spec = agents["data_analysis"].process(
                    task_description=args.task,
                    mode=args.mode
                )
        
        logger.info("Data analysis and task spec generation completed successfully")
        
        # Save task_spec (after all branches)
        state["task_spec"] = task_spec
        if not skip_data_analysis or current_iteration == 0:
            save_artifact(args.output, "task_spec", task_spec)
        
        # Log key task spec information
        logger.info("Task Specification Summary:")
        logger.info(f"  - Description: {task_spec.get('description', 'N/A')[:100]}...")
        logger.info(f"  - Data folder: {task_spec.get('data_folder', 'N/A')}")
        logger.info(f"  - Data files: {list(task_spec.get('data_files', {}).keys())}")
        
        # Log data analysis results if present
        data_analysis_result = task_spec.get('data_analysis_result', {})
        if data_analysis_result:
            logger.info("Data Analysis Summary:")
            logger.info(f"  - File summaries: {len(data_analysis_result.get('file_summaries', []))} files analyzed")
            logger.info(f"  - Overall simulation design: {bool(data_analysis_result.get('overall_simulation_design', {}))}")
            logger.info(f"  - Agent archetypes: {bool(data_analysis_result.get('agent_archetypes', {}))}")
            logger.info(f"  - Interaction topology: {bool(data_analysis_result.get('interaction_topology', {}))}")
            logger.info(f"  - Calibratable parameters: {len(data_analysis_result.get('calibratable_parameters', []))}")
        
        # ==================================================
        # BLUEPRINT FEEDBACK AFTER DATA ANALYSIS (ACE/ALPHA mode only)
        # ==================================================
        if args.mode in ["ace", "alpha", "gsim"]:
            logger.info("=" * 50)
            logger.info("BLUEPRINT FEEDBACK AFTER DATA ANALYSIS (ACE mode)")
            logger.info("=" * 50)
            
            # Extract current blueprint from task_spec
            current_blueprint = {k: v for k, v in task_spec.get("data_analysis_result", {}).items() if k != "file_summaries"}
            
            if current_blueprint:
                # Collect blueprint feedback from user (only if not in auto mode)
                blueprint_feedback = None
                if not args.auto:
                    blueprint_feedback = get_blueprint_feedback(
                        logger,
                        0,  # Initial iteration (before main loop)
                        current_blueprint
                    )
                
                # Update blueprint if feedback is provided
                if blueprint_feedback and blueprint_feedback.strip():
                    logger.info("Blueprint feedback received, updating blueprint...")
                    updated_blueprint = update_blueprint_from_feedback(
                        logger,
                        current_blueprint,
                        blueprint_feedback,
                        agent_container
                    )
                    
                    # Update task_spec with new blueprint
                    task_spec["data_analysis_result"] = {
                        **updated_blueprint,
                        "file_summaries": task_spec.get("data_analysis_result", {}).get("file_summaries", [])
                    }
                    
                    # Save updated task_spec
                    save_artifact(args.output, "task_spec", task_spec, iteration=0)
                    logger.info("Blueprint updated and saved to task_spec after data analysis")
                else:
                    logger.info("No blueprint feedback provided after data analysis, keeping initial blueprint")
            else:
                logger.warning("No blueprint found in task_spec after data analysis, skipping blueprint feedback step")
        
        # ==================================================
        # MAIN ITERATION LOOP (Lite-style workflow)
        # ==================================================
        
        # Initialize best_simulator_info and simulation_info_history for alpha mode (track best code across iterations)
        best_simulator_info = None
        simulation_info_history = []
        if args.mode in ["alpha", "gsim"]:
            logger.info("Alpha mode: Initializing best_simulator_info tracker and simulation_info_history")
            
            # Try to load simulation_info_history from previous run
            simulation_info_history_path = os.path.join(args.output, "simulation_info_history.json")
            if os.path.exists(simulation_info_history_path):
                try:
                    with open(simulation_info_history_path, 'r', encoding='utf-8') as f:
                        simulation_info_history = json.load(f)
                    logger.info(f"Alpha mode: Loaded simulation_info_history from {simulation_info_history_path} ({len(simulation_info_history)} iterations)")
                    
                    # Restore best_simulator_info from history if available
                    if simulation_info_history:
                        # Find the iteration with minimum val_loss (filter out None and invalid values)
                        valid_history = []
                        for x in simulation_info_history:
                            val_loss = x.get("val_loss")
                            # Check if val_loss is a valid number (not None, not inf, not nan)
                            if val_loss is not None and isinstance(val_loss, (int, float)):
                                # Check for inf/nan (JSON may serialize inf as "Infinity" string)
                                val_loss_str = str(val_loss).lower()
                                if val_loss_str not in ['inf', 'infinity', 'nan', '-inf', '-infinity']:
                                    try:
                                        val_loss_float = float(val_loss)
                                        if val_loss_float != float('inf') and val_loss_float != float('-inf') and not (val_loss_float != val_loss_float):  # not NaN
                                            valid_history.append(x)
                                    except (ValueError, TypeError):
                                        pass
                        
                        if valid_history:
                            best_iter_info = min(valid_history, key=lambda x: float(x.get("val_loss")))
                            best_simulator_info = {
                                "iteration": best_iter_info.get("iteration"),
                                "val_loss": float(best_iter_info.get("val_loss")),
                                "code": best_iter_info.get("code"),
                                "simulator_description": best_iter_info.get("simulator_description"),
                                "results_json": best_iter_info.get("results_json")
                            }
                            logger.info(f"Alpha mode: Restored best_simulator_info from history (iteration {best_simulator_info['iteration']}, val_loss: {best_simulator_info['val_loss']:.4f})")
                        else:
                            logger.info("Alpha mode: No valid iterations found in history (all have invalid val_loss)")
                except Exception as e:
                    logger.warning(f"Alpha mode: Failed to load simulation_info_history from {simulation_info_history_path}: {e}. Starting fresh.")
                    simulation_info_history = []
            
            # Initialize best_simulator_info if not loaded from history
            if best_simulator_info is None:
                best_simulator_info = {
                    "iteration": None,
                    "val_loss": float('inf'),  # Start with infinity, lower is better
                    "code": None,
                    "simulator_description": None,
                    "results_json": None  # Will store the dict that constitutes output_iter_*/results.json
                }
                logger.info("Alpha mode: Initialized new best_simulator_info (no previous history found)")
        
        while current_iteration < args.iterations:
            logger.info("=" * 50)
            logger.info(f"STARTING ITERATION {current_iteration + 1}/{args.iterations}")
            logger.info("=" * 50)
        
            # --------------------------------------------------
            # STEP 1: Code Generation
            # --------------------------------------------------
            # Skip code generation if:
            # 1. skip_initial_code_generation is True AND current_iteration equals the persisted iteration number, OR
            # 2. We're in the first iteration and skip_initial_code_generation is True
            should_skip_code_gen = (
                skip_initial_code_generation and 
                (persisted_iteration_number is not None and current_iteration == persisted_iteration_number)
            )
            
            if not should_skip_code_gen:
                logger.info("CODE GENERATION")
                
                # Select strategies for prompt BEFORE code generation (open/queued -> in_progress)
                if playbook_manager and args.mode in ["ace", "alpha"] and current_iteration > 0:
                    logger.info("Selecting playbook strategies for code patch prompt...")
                    selected_ids = playbook_manager.select_strategies_for_prompt_simple(
                        max_count=None,  # Select all open/queued strategies
                        iteration=current_iteration,
                    )
                    # Refresh playbook reference after selection (which modifies and saves playbook)
                    playbook = playbook_manager.playbook
                    logger.info(f"✓ Selected {len(selected_ids)} strategies for prompt (status -> in_progress): {selected_ids}")
                
                # Get previous code if exists
                prev_code = None
                if current_iteration > 0 and current_iteration - 1 in code_memory:
                    prev_code_dict = code_memory[current_iteration - 1]
                    prev_code_filename = f"simulation_code_iter_{current_iteration - 1}.py"
                    if isinstance(prev_code_dict, dict) and prev_code_filename in prev_code_dict:
                        prev_code = prev_code_dict[prev_code_filename]
                    elif isinstance(prev_code_dict, str):
                        prev_code = prev_code_dict
                
                # Get previous iteration's simulation results for patch prompt (iteration >= 1)
                prev_simulation_results = None
                if current_iteration >= 1 and current_iteration - 1 in code_memory:
                    # Try to load previous simulation results
                    prev_results_file = os.path.join(args.output, f"simulation_results_iter_{current_iteration - 1}.json")
                    if os.path.exists(prev_results_file):
                        try:
                            with open(prev_results_file, 'r', encoding='utf-8') as f:
                                prev_simulation_results = json.load(f)
                            logger.debug(f"Loaded previous simulation results from iteration {current_iteration - 1}")
                        except Exception as e:
                            logger.warning(f"Failed to load previous simulation results: {e}")
                
                # Generate code
                process_kwargs = {
                    "task_spec": task_spec,
                    "data_analysis": None,  # Not used in odd/ace mode
                    "model_plan": None,     # Not used in odd/ace mode
                    "feedback": state["feedback"],  # Will be None in first iteration
                    "data_path": data_path,
                    "previous_code": prev_code,
                    "historical_fix_log": historical_fix_log,
                    "mode": args.mode,
                    "selfloop": args.selfloop,
                    "blueprint": None,
                    "output_dir": args.output,
                    "iteration": current_iteration,
                    "simulation_results": prev_simulation_results  # Pass previous iteration's results for patch prompt
                }
                if args.mode in ["ace", "alpha"]:
                    process_kwargs["playbook"] = playbook
                
                # Add best_simulator_info and simulation_info_history for alpha mode
                if args.mode in ["alpha", "gsim"]:
                    if best_simulator_info is not None:
                        process_kwargs["best_simulator_info"] = best_simulator_info
                        if best_simulator_info.get("iteration") is not None:
                            logger.info(f"Alpha mode: Passing best_simulator_info to code generation (best iteration: {best_simulator_info['iteration']}, val_loss: {best_simulator_info.get('val_loss', 'N/A')})")
                        else:
                            logger.info("Alpha mode: Passing best_simulator_info to code generation (no best iteration found yet)")
                    
                    # Pass simulation_info_history to code generation
                    process_kwargs["simulation_info_history"] = simulation_info_history
                    logger.info(f"Alpha mode: Passing simulation_info_history to code generation ({len(simulation_info_history) if simulation_info_history else 0} iterations)")
                
                state["generated_code"] = agents["code_generation"].process(**process_kwargs)
                
                # Save generated code (use current_iteration as file number: iter_0, iter_1, etc.)
                save_generated_code(args.output, state["generated_code"], iteration=current_iteration)
                
                # Store in code_memory (key: current_iteration, value: {filename: code})
                gen_code_dict = state["generated_code"]["code"]
                code_memory[current_iteration] = {f"simulation_code_iter_{current_iteration}.py": gen_code_dict}
                gen_sim_desc = state["generated_code"].get("simulator_description", "")
                code_description_memory[current_iteration] = {
                    f"simulator_description_iter_{current_iteration}.txt": gen_sim_desc
                }
                
                logger.info(f"Code generation completed for iteration {current_iteration} (saved as iter_{current_iteration})")
            else:
                logger.info(f"Skipping code generation for iteration {current_iteration} (using persisted code)")
                # Reset skip_initial_code_generation after processing the persisted iteration
                if persisted_iteration_number is not None and current_iteration == persisted_iteration_number:
                    skip_initial_code_generation = False  # Reset for next iteration
                    logger.info(f"Reset skip_initial_code_generation flag - next iteration ({current_iteration + 1}) will generate code")
            
            # --------------------------------------------------
            # STEP 2: Code Verification
            # --------------------------------------------------
            # Determine code file path based on whether code was generated or loaded
            # Always use current_iteration to determine the code file path
            code_file_path = os.path.join(args.output, f"simulation_code_iter_{current_iteration}.py")
            
            # ODD mode: Skip verification, simulation, and evaluation, use placeholders
            if args.mode == "odd":
                logger.info(f"{args.mode.upper()} mode: Skipping verification, simulation, and evaluation - using placeholders")
                state["verification_results"] = {
                    "placeholder": True,
                    "note": f"Verification is not executed in {args.mode.upper()} mode",
                    "passed": True
                }
                state["simulation_results"] = {
                    "placeholder": True,
                    "note": f"Simulation execution is not performed in {args.mode.upper()} mode",
                    "execution_status": "skipped"
                }
                state["evaluation_results"] = {
                    "placeholder": True,
                    "note": f"Evaluation is not performed in {args.mode.upper()} mode"
                }
                save_artifact(args.output, f"verification_results_iter_{current_iteration}", state["verification_results"])
                save_artifact(args.output, f"simulation_results_iter_{current_iteration}", state["simulation_results"])
                save_artifact(args.output, f"evaluation_results_iter_{current_iteration}", state["evaluation_results"])
                logger.info(f"{args.mode.upper()} mode: Placeholders created for verification, simulation, and evaluation results")
            # ACE/ALPHA mode: Skip verification, but execute simulation
            elif args.mode in ["ace", "alpha", "gsim"]:
                logger.info(f"{args.mode.upper()} mode: Skipping verification, but executing simulation")
                state["verification_results"] = {
                    "placeholder": True,
                    "note": f"Verification is not executed in {args.mode.upper()} mode",
                    "passed": True
                }
                save_artifact(args.output, f"verification_results_iter_{current_iteration}", state["verification_results"])
                
                # Execute simulation (ACE/ALPHA/GSIM: subprocess path in each agent when mode matches)
                logger.info("SIMULATION EXECUTION (ACE/GSIM/Alpha path)")
                
                # Extract environment variables and output directory for subprocess execution
                project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                
                state["simulation_results"] = agents["simulation_execution"].process(
                    code_path=code_file_path,
                    task_spec=task_spec,
                    data_path=data_path,
                    mode=args.mode,  # ACE/ALPHA mode uses subprocess execution (same as lite mode)
                    output_dir=args.output,
                    iteration=current_iteration,
                    project_root=project_root,
                    openai_api_key=openai_api_key
                )
                save_artifact(args.output, f"simulation_results_iter_{current_iteration}", state["simulation_results"])
                
                if state["simulation_results"] and state["simulation_results"].get("execution_status") == "success":
                    logger.info(f"✅ Simulation execution completed successfully")
                    
                    # Update simulation_info_history and best_simulator_info for alpha mode
                    if args.mode in ["alpha", "gsim"]:
                        # Extract val_loss from simulation_results
                        simulation_metrics = state["simulation_results"].get("simulation_metrics", {})
                        current_val_loss = simulation_metrics.get("val_loss")
                        
                        if current_val_loss is not None:
                            # Get current iteration info
                            current_code = None
                            if current_iteration in code_memory:
                                code_dict = code_memory[current_iteration]
                                code_filename = f"simulation_code_iter_{current_iteration}.py"
                                if isinstance(code_dict, dict) and code_filename in code_dict:
                                    current_code = code_dict[code_filename]
                                elif isinstance(code_dict, str):
                                    current_code = code_dict
                            
                            current_simulator_description = None
                            if state.get("generated_code") and "simulator_description" in state["generated_code"]:
                                current_simulator_description = state["generated_code"]["simulator_description"]
                            
                            # Read results.json from output_iter_* directory
                            current_results_json = None
                            output_iter_dir = os.path.join(args.output, f"output_iter_{current_iteration}")
                            results_json_path = os.path.join(output_iter_dir, "results.json")
                            if os.path.exists(results_json_path):
                                try:
                                    with open(results_json_path, 'r', encoding='utf-8') as f:
                                        current_results_json = json.load(f)
                                    logger.debug(f"Alpha mode: Loaded results.json from {results_json_path}")
                                except Exception as e:
                                    logger.warning(f"Alpha mode: Failed to load results.json from {results_json_path}: {e}")
                            else:
                                logger.debug(f"Alpha mode: results.json not found at {results_json_path}")
                            
                            # Add current iteration info to simulation_info_history
                            current_iteration_info = {
                                "iteration": current_iteration,
                                "val_loss": current_val_loss,
                                "code": current_code,
                                "simulator_description": current_simulator_description,
                                "results_json": current_results_json
                            }
                            simulation_info_history.append(current_iteration_info)
                            logger.info(f"Alpha mode: Added iteration {current_iteration} to simulation_info_history (val_loss: {current_val_loss:.4f})")
                            
                            # Persist simulation_info_history to disk
                            try:
                                simulation_info_history_path = os.path.join(args.output, "simulation_info_history.json")
                                os.makedirs(args.output, exist_ok=True)
                                with open(simulation_info_history_path, 'w', encoding='utf-8') as f:
                                    json.dump(simulation_info_history, f, indent=2, ensure_ascii=False)
                                logger.info(f"Alpha mode: Persisted simulation_info_history to {simulation_info_history_path}")
                            except Exception as e:
                                logger.warning(f"Alpha mode: Failed to persist simulation_info_history: {e}")
                            
                            # Update best_simulator_info if current iteration is better
                            if best_simulator_info is not None:
                                if current_val_loss < best_simulator_info["val_loss"]:
                                    logger.info(f"Alpha mode: New best iteration found! val_loss: {best_simulator_info['val_loss']:.4f} -> {current_val_loss:.4f}")
                                    
                                    # Update best_simulator_info
                                    best_simulator_info["iteration"] = current_iteration
                                    best_simulator_info["val_loss"] = current_val_loss
                                    best_simulator_info["code"] = current_code
                                    best_simulator_info["simulator_description"] = current_simulator_description
                                    best_simulator_info["results_json"] = current_results_json
                                else:
                                    logger.info(f"Alpha mode: Current iteration val_loss ({current_val_loss:.4f}) >= best ({best_simulator_info['val_loss']:.4f}), keeping best from iteration {best_simulator_info['iteration']}")
                        else:
                            logger.warning("Alpha mode: val_loss not found in simulation_metrics, skipping simulation_info_history and best_simulator_info update")
                else:
                    logger.warning(f"❌ Simulation execution failed")
                
                # Skip evaluation in ACE mode
                state["evaluation_results"] = {
                    "placeholder": True,
                    "note": f"Evaluation is not performed in {args.mode.upper()} mode"
                }
                save_artifact(args.output, f"evaluation_results_iter_{current_iteration}", state["evaluation_results"])
            else:
                # Full mode: Execute verification, simulation, and evaluation
                logger.info("CODE VERIFICATION")
                # Get code from generated_code or code_memory
                if state.get("generated_code") and state["generated_code"].get("code"):
                    code_to_verify = state["generated_code"]["code"]
                elif current_iteration in code_memory:
                    code_dict = code_memory[current_iteration]
                    code_filename = f"simulation_code_iter_{current_iteration}.py"
                    code_to_verify = code_dict.get(code_filename) if isinstance(code_dict, dict) else code_dict
                else:
                    logger.warning("No code available for verification")
                    code_to_verify = ""
                
                state["verification_results"] = agents["code_verification"].process(
                    code=code_to_verify,
                    task_spec=task_spec,
                    data_path=data_path,
                    use_sandbox=False  # Lightweight verification
                )
                save_artifact(args.output, f"verification_results_iter_{current_iteration}", state["verification_results"])
                
                if state["verification_results"]["passed"]:
                    logger.info(f"✅ Code verification PASSED for iteration {current_iteration}")
                else:
                    logger.warning(f"❌ Code verification FAILED for iteration {current_iteration}")
                
                # --------------------------------------------------
                # STEP 3 & 4: Simulation Execution & Evaluation
                # --------------------------------------------------
                if state["verification_results"]["passed"]:
                    logger.info("SIMULATION EXECUTION")
                    
                    # Extract environment variables and output directory for subprocess execution
                    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    
                    state["simulation_results"] = agents["simulation_execution"].process(
                        code_path=code_file_path,
                        task_spec=task_spec,
                        data_path=data_path,
                        mode="lite",  # Use lite mode for subprocess execution
                        output_dir=args.output,
                        iteration=current_iteration,
                        project_root=project_root,
                        openai_api_key=openai_api_key
                    )
                    save_artifact(args.output, f"simulation_results_iter_{current_iteration}", state["simulation_results"])
                    
                    if state["simulation_results"] and state["simulation_results"].get("execution_status") == "success":
                        logger.info(f"✅ Simulation execution completed successfully")
                    else:
                        logger.warning(f"❌ Simulation execution failed")
                    
                    logger.info("RESULT EVALUATION")
                    state["evaluation_results"] = agents["result_evaluation"].process(
                        simulation_results=state["simulation_results"],
                        task_spec=task_spec,
                        data_analysis=None  # No data analysis in odd/ace mode
                    )
                    save_artifact(args.output, f"evaluation_results_iter_{current_iteration}", state["evaluation_results"])
                else:
                    logger.warning("Skipping execution and evaluation due to verification failure")
                    state["simulation_results"] = None
                    state["evaluation_results"] = None
            
            # --------------------------------------------------
            # STEP 5: Blueprint Feedback (ACE/ALPHA mode only)
            # --------------------------------------------------
            if args.mode in ["ace", "alpha", "gsim"]:
                logger.info("BLUEPRINT FEEDBACK (ACE/ALPHA mode)")
                
                # Extract current blueprint from task_spec
                current_blueprint = {k: v for k, v in task_spec.get("data_analysis_result", {}).items() if k != "file_summaries"}
                
                if current_blueprint:
                    # Collect blueprint feedback from user (only if not in auto mode)
                    blueprint_feedback = None
                    if not args.auto:
                        blueprint_feedback = get_blueprint_feedback(
                            logger,
                            current_iteration,
                            current_blueprint
                        )
                    
                    # Update blueprint if feedback is provided
                    if blueprint_feedback and blueprint_feedback.strip():
                        logger.info("Blueprint feedback received, updating blueprint...")
                        updated_blueprint = update_blueprint_from_feedback(
                            logger,
                            current_blueprint,
                            blueprint_feedback,
                            agent_container
                        )
                        
                        # Update task_spec with new blueprint
                        task_spec["data_analysis_result"] = {
                            **updated_blueprint,
                            "file_summaries": task_spec.get("data_analysis_result", {}).get("file_summaries", [])
                        }
                        
                        # Save updated task_spec
                        save_artifact(args.output, "task_spec", task_spec, iteration=current_iteration)
                        logger.info("Blueprint updated and saved to task_spec")
                    else:
                        logger.info("No blueprint feedback provided, keeping current blueprint")
                else:
                    logger.warning("No blueprint found in task_spec, skipping blueprint feedback step")
            
            # --------------------------------------------------
            # STEP 6: Feedback Generation
            # --------------------------------------------------
            logger.info("FEEDBACK GENERATION")
            
            # Get current and previous code for feedback
            # Use the code from current iteration (whether persisted or generated)
            current_code_dict = code_memory[current_iteration]
            current_code_filename = f"simulation_code_iter_{current_iteration}.py"
            current_code = current_code_dict[current_code_filename]
            
            # Get previous code if exists
            previous_code = None
            if current_iteration > 0 and current_iteration - 1 in code_memory:
                prev_code_dict = code_memory[current_iteration - 1]
                prev_code_filename = f"simulation_code_iter_{current_iteration - 1}.py"
                if isinstance(prev_code_dict, dict) and prev_code_filename in prev_code_dict:
                    previous_code = prev_code_dict[prev_code_filename]
                elif isinstance(prev_code_dict, str):
                    previous_code = prev_code_dict
            
            # Generate feedback (system + optional user feedback)
            # In ACE mode with interactive=True (--auto=False), the agent will collect user feedback internally
            # In other modes with manual feedback, we collect it here first
            user_feedback_text = None
            should_stop_from_user_feedback = False  # Check for #STOP# in non-ACE modes
            if args.mode != "ace" and not args.auto:
                logger.info("Manual feedback mode - prompting user for feedback before LLM generation")
                user_feedback_text = get_user_feedback(
                    logger,
                    current_iteration,
                    verification_results=state["verification_results"],
                    simulation_results=state["simulation_results"],
                    evaluation_results=state["evaluation_results"],
                    generated_code=state["generated_code"]
                )
                # Check for #STOP# command in user feedback (non-ACE modes)
                if user_feedback_text and "#STOP#" in user_feedback_text:
                    should_stop_from_user_feedback = True
                    logger.info("🛑 User requested to stop iterations via #STOP# command in user feedback")
            
            # Build process kwargs for feedback generation agent
            process_kwargs = {
                "task_spec": task_spec,
                "model_plan": None,  # Not used in odd/ace mode
                "generated_code": state["generated_code"],
                "verification_results": state["verification_results"],
                "simulation_results": state["simulation_results"],
                "evaluation_results": state["evaluation_results"],
                "current_code": current_code,
                "previous_code": previous_code,
                "iteration": current_iteration,
                "historical_fix_log": historical_fix_log,
                "mode": args.mode,
                # interactive=True only when --auto=False (manual mode)
                # In ACE mode, agent will collect user feedback internally when interactive=True
                "interactive": not args.auto
            }
            
            # Add best_simulator_info and simulation_info_history for alpha mode
            if args.mode in ["alpha", "gsim"]:
                if best_simulator_info is not None:
                    process_kwargs["best_simulator_info"] = best_simulator_info
                    if best_simulator_info.get("iteration") is not None:
                        logger.info(f"Alpha mode: Passing best_simulator_info to feedback generation (best iteration: {best_simulator_info['iteration']}, val_loss: {best_simulator_info.get('val_loss', 'N/A')})")
                    else:
                        logger.info("Alpha mode: Passing best_simulator_info to feedback generation (no best iteration found yet)")
                
                # Pass simulation_info_history to feedback generation
                process_kwargs["simulation_info_history"] = simulation_info_history
                logger.info(f"Alpha mode: Passing simulation_info_history to feedback generation ({len(simulation_info_history) if simulation_info_history else 0} iterations)")
            
            # Call feedback generation agent
            # In ACE mode with --auto=False, agent handles user feedback collection internally
            # In ACE mode with --auto=True, agent skips user feedback collection (interactive=False)
            system_feedback = agents["feedback_generation"].process(**process_kwargs)
            
            # Check if user requested stop
            # In ACE mode: check feedback from agent (set by feedback_generation_ace agent internally)
            # In non-ACE modes: check should_stop_from_user_feedback (set above after get_user_feedback)
            should_stop_from_feedback = system_feedback.get("should_stop", False) or should_stop_from_user_feedback
            
            # Combine user feedback (if any) with system feedback (non-ACE modes only)
            # ACE mode user feedback is handled internally by the agent
            if args.mode != "ace" and not args.auto:
                combined_feedback = dict(system_feedback)
                
                if user_feedback_text and not should_stop_from_user_feedback:
                    # Only add user feedback to feedback sections if not stopping
                    # (If stopping, we still want to preserve the feedback, but #STOP# should be removed from display)
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
                    
                    # Also add to summary
                    if "summary" in combined_feedback:
                        combined_feedback["summary"] = f"USER FEEDBACK: {user_feedback_text}\n\nSYSTEM FEEDBACK: {combined_feedback['summary']}"
                    else:
                        combined_feedback["summary"] = f"USER FEEDBACK: {user_feedback_text}"
                    
                    logger.info("User feedback has been integrated with system feedback")
                else:
                    logger.info("No user feedback provided - using system feedback only")
                
                state["feedback"] = combined_feedback
            else:
                # Auto mode or ACE mode - use system feedback as-is
                state["feedback"] = system_feedback
                
            # Add should_stop flag to feedback if user requested stop (for consistency)
            if should_stop_from_user_feedback:
                state["feedback"]["should_stop"] = True
                state["feedback"]["stop_reason"] = "User requested stop via #STOP# command in user feedback"
            
            save_artifact(args.output, f"feedback_iter_{current_iteration}", state["feedback"])
            
            # Convert feedback to playbook entries (ACE/ALPHA mode only)
            if args.mode in ["ace", "alpha"] and playbook_manager:
                feedback_to_convert = state["feedback"]
                
                # Check if feedback is in ACE format (dict with issue_id keys)
                # ACE feedback format: {issue_id: {issue_data}, ...}
                if isinstance(feedback_to_convert, dict):
                    # Check if it looks like ACE format (all values are dicts with issue_type)
                    is_ace_format = all(
                        isinstance(v, dict) and "issue_type" in v
                        for v in feedback_to_convert.values()
                    )
                    # todo: if there are too many feedbacks, and we assembled some of them into prompt in this iterations, how about the unsolved ones? do we need to set the window length? do we have any related work that can tell us how to set the window length? how about we calc the percentage of the issues that are solved by average if we feed all of them into the window? how long is the current prompt window (with out the playbook)?
                    if is_ace_format:
                        logger.info("Converting ACE feedback to playbook entries...")
                        
                        # Step 1: Add new feedback entries and handle merges
                        add_result = playbook_manager.add_feedback_entries(
                            feedback=feedback_to_convert,
                            iteration=current_iteration,
                        )
                        # Refresh playbook reference after add_feedback_entries (which reloads from file)
                        playbook = playbook_manager.playbook
                        
                        # Get merged strategy IDs (extract old_strategy_id from tuples)
                        merged_ids_tuples = add_result.get("merged_ids", [])
                        merged_old_strategy_ids = [old_id for (new_id, old_id) in merged_ids_tuples]
                        
                        # Step 2: Resolve in_progress strategies that were NOT merged
                        # (i.e., the issue disappeared from feedback - simple version, no metric_links check)
                        if current_iteration > 0:
                            resolved_non_merged = playbook_manager.resolve_non_merged_in_progress_simple(
                                merged_strategy_ids=merged_old_strategy_ids,
                                iteration=current_iteration,
                            )
                            if resolved_non_merged:
                                playbook = playbook_manager.playbook
                                logger.info(f"✓ Resolved {len(resolved_non_merged)} strategies (not merged in feedback)")
                        
                        logger.info(f"✓ Added {len(add_result.get('added_ids', []))} feedback entries to playbook")
                    else:
                        logger.debug("Feedback is not in ACE format, skipping playbook conversion")
                else:
                    logger.debug("Feedback is not a dictionary, skipping playbook conversion")
            
            # Reset skip_initial_code_generation after first iteration (when loading persisted code)
            # This ensures that subsequent iterations will generate code normally
            if skip_initial_code_generation and current_iteration == 0:
                skip_initial_code_generation = False
                logger.info("Reset skip_initial_code_generation flag - next iteration will generate code")
            
            # --------------------------------------------------
            # STEP 7: Iteration Control Decision
            # --------------------------------------------------
            # Check if user requested stop (unified check for both ACE and non-ACE modes)
            if should_stop_from_feedback:
                logger.info("Skipping ITERATION CONTROL DECISION (user requested stop via #STOP#)")
                # Set iteration decision to stop without calling iteration_control agent
                state["iteration_decision"] = {
                    "continue": False,
                    "reason": state["feedback"].get("stop_reason", "User requested stop via #STOP# command")
                }
                save_artifact(args.output, f"iteration_decision_iter_{current_iteration}", state["iteration_decision"])
                
                # Save iteration snapshot for playbook (checkpoint level) - this is still needed
                if playbook_manager and args.mode in ["ace", "alpha"]:
                    playbook_manager.save_iteration_snapshot(current_iteration)
                    logger.info(f"📸 Playbook iteration snapshot saved: iter_{current_iteration:03d}")
                
                logger.info(f"🛑 Stopping after {current_iteration + 1} iterations: {state['iteration_decision']['reason']}")
                break
            
            # Normal iteration control process
            logger.info("ITERATION CONTROL DECISION")
            
            # Note: user_feedback parameter is no longer needed for #STOP# checking
            # The #STOP# check is now done in the main workflow before calling iteration_control
            # ACE/ALPHA mode uses decision function with simulation_results and feedback
            # Other modes use LLM-based iteration control
            if args.mode in ["ace", "alpha", "gsim"]:
                state["iteration_decision"] = agents["iteration_control"].process(
                    current_iteration=current_iteration,
                    max_iterations=args.iterations,
                    simulation_results=state["simulation_results"],
                    feedback=state["feedback"]
                )
            else:
                state["iteration_decision"] = agents["iteration_control"].process(
                    feedback=state["feedback"],
                    verification_results=state["verification_results"],
                    evaluation_results=state["evaluation_results"],
                    current_iteration=current_iteration,
                    max_iterations=args.iterations,
                    auto_mode=args.auto
                )
            save_artifact(args.output, f"iteration_decision_iter_{current_iteration}", state["iteration_decision"])
            
            # Save iteration snapshot for playbook (checkpoint level)
            if playbook_manager and args.mode in ["ace", "alpha"]:
                playbook_manager.save_iteration_snapshot(current_iteration)
                logger.info(f"📸 Playbook iteration snapshot saved: iter_{current_iteration:03d}")
            
            # Check if we should continue
            if not state["iteration_decision"]["continue"]:
                logger.info(f"🛑 Stopping after {current_iteration + 1} iterations: {state['iteration_decision']['reason']}")
                break
            
            # Move to next iteration
            current_iteration += 1
        
        # ==================================================
        # FINAL SUMMARY
        # ==================================================
        # Note: After loop ends, current_iteration is the next iteration number
        # So the last completed iteration is current_iteration - 1
        # But if we completed N iterations (0 to N-1), current_iteration = N
        final_iteration = current_iteration  # This is the total number of iterations completed
        last_code_iteration = current_iteration - 1 if current_iteration > 0 else 0
        
        # Finalize playbook (session level)
        if playbook_manager and args.mode in ["ace", "alpha"]:
            final_archive_path = playbook_manager.finalize()
            logger.info(f"📚 Playbook finalized and archived: {final_archive_path}")
        
        logger.info("=" * 50)
        logger.info("TEST COMPLETION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Data Analysis: Completed")
        logger.info(f"✅ Code Generation: Completed ({final_iteration} iterations)")
        logger.info(f"📁 Artifacts saved to: {args.output}")
        logger.info(f"📄 Final code: simulation_code_iter_{last_code_iteration}.py")
        if playbook_manager:
            logger.info(f"📚 Playbook storage: {playbook_manager.storage_root}")
        
        return {
            "status": "success",
            "state": state,
            "task_spec": task_spec,
            "data_analysis": data_analysis_result,
            "generated_code": state["generated_code"],
            "total_iterations": final_iteration,
            "output_path": args.output
        }
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "failed",
            "error": str(e),
            "output_path": args.output
        }

def check_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    api_key = load_api_key("OPENAI_API_KEY")
    return api_key is not None

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.output, args.debug)
    
    # Check API key
    if not check_api_key():
        logger.error("OpenAI API key not found in keys.py")
        logger.info("Please set up your API key using: python main.py --setup-api-key")
        return 1
    
    # Set up the dependency injection container
    container = setup_container(args.config)
    
    # Run the test
    result = run_data_analysis_test(args, logger)
    
    if result["status"] == "success":
        logger.info("DataAnalysisOddAgent test completed successfully!")
        return 0
    else:
        logger.error("DataAnalysisOddAgent test failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
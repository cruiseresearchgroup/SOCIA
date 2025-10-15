#!/usr/bin/env python3
"""
Standalone test for Calibrasim Code Generation Agent
- Loads existing task_spec, data_analysis, and model_plan
- Forces use of Calibrasim code generation prompt
- Saves generated code and summary to files
"""

import os
import sys
import json
import logging
import yaml
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.code_generation_calibrasim.agent import CodeGenerationCalibrasimAgent

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_codegen_calibrasim.log')
        ]
    )


def load_config() -> Dict[str, Any]:
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        return {}


def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def test_codegen():
    print("=" * 60)
    print("Start Calibrasim Code Generation Agent test")
    print("=" * 60)

    setup_logging()

    cfg = load_config()
    agents_cfg = cfg.get('agents', {})

    # Prefer Calibrasim code generation config; fallback to direct prompt
    codegen_cfg = agents_cfg.get('code_generation_calibrasim', {
        'prompt_template': 'templates/Calibrasim_code_generation_prompt.txt',
        'output_format': 'python',
        'code_style': 'pep8'
    })

    # Initialize agent
    agent = CodeGenerationCalibrasimAgent(codegen_cfg)

    # Load inputs from a known Calibrasim output directory
    base = 'output/mask_adoption_calibrasim_debug_run2'
    task_spec = load_json(os.path.join(base, 'task_spec_iter_1.json'))
    data_analysis = load_json(os.path.join(base, 'data_analysis_iter_1.json'))
    model_plan = load_json(os.path.join(base, 'model_plan_iter_1.json'))

    # Basic stats
    print(f"task_spec keys: {list(task_spec.keys())[:10]}")
    print(f"data_analysis keys: {list(data_analysis.keys())[:10]}")
    print(f"model_plan keys: {list(model_plan.keys())[:10]}")

    # Generate code
    print("Generating code (Calibrasim mode)...")
    result = agent.process(
        task_spec=task_spec,
        data_analysis=data_analysis,
        model_plan=model_plan,
        feedback=None,
        data_path=task_spec.get('data_folder') or 'data_fitting/mask_adoption_data/',
        previous_code=None,
        historical_fix_log=None,
        mode='calibrasim',
        selfloop=1
    )

    # Validate and save
    code = result.get('code', '')
    summary = result.get('code_summary', '')

    out_code = 'test_codegen_calibrasim_output.py'
    out_meta = 'test_codegen_calibrasim_meta.json'

    with open(out_code, 'w', encoding='utf-8') as f:
        f.write(code)
    with open(out_meta, 'w', encoding='utf-8') as f:
        json.dump(result.get('metadata', {}), f, indent=2, ensure_ascii=False)

    print(f"Saved code to: {out_code}")
    print(f"Summary: {summary}")
    print("Metadata:")
    print(json.dumps(result.get('metadata', {}), indent=2, ensure_ascii=False))

    print("\nDone.")


if __name__ == '__main__':
    test_codegen()






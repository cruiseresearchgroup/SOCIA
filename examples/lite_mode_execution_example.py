#!/usr/bin/env python3
"""
Example demonstrating the lite mode subprocess execution in SOCIA.

This script shows how the lite mode now executes Python scripts directly
using subprocess instead of skipping execution entirely.
"""

import os
import sys
import tempfile

# Add the parent directory to the path to import SOCIA modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.simulation_execution.agent import run_python_script

def create_sample_simulation():
    """Create a sample simulation script for testing."""
    sample_code = '''
import os
import time
import random

def main():
    print("Starting sample simulation...")
    
    # Get environment variables
    project_root = os.environ.get("PROJECT_ROOT", ".")
    data_path = os.environ.get("DATA_PATH", "data")
    
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    
    # Simulate some work
    for i in range(5):
        value = random.random()
        print(f"Step {i}: Simulation value = {value:.3f}")
        time.sleep(0.1)
    
    print("Simulation completed successfully!")
    print("Results: Final simulation score = 0.842")

if __name__ == "__main__":
    main()
'''
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_code)
        return f.name

def main():
    """Demonstrate lite mode subprocess execution."""
    print("SOCIA Lite Mode Subprocess Execution Example")
    print("=" * 50)
    
    # Create a sample simulation script
    script_file = create_sample_simulation()
    print(f"Created sample script: {script_file}")
    
    try:
        # Execute the script using the run_python_script function
        print("\nExecuting script with run_python_script...")
        result = run_python_script(script_file, data_path="data/sample")
        
        print("\nExecution Results:")
        print("=" * 30)
        print("standard output（stdout）:")
        print(result["stdout"])
        print("\nerror info（stderr）:")
        print(result["stderr"])
        print(f"\nreturn code（returncode）:")
        print(result["returncode"])
        print(f"\nExecution time: {result['execution_time']:.2f} seconds")
        print(f"Success: {result['success']}")
        
    finally:
        # Clean up the temporary file
        if os.path.exists(script_file):
            os.unlink(script_file)
            print(f"\nCleaned up temporary file: {script_file}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Blueprint Mode Example for SOCIA

This example demonstrates how to use SOCIA's blueprint mode for rapid prototyping
and concept validation. Blueprint mode is identical to lite mode but provides 
a conceptually distinct mode for creating simulation blueprints.

Usage:
    python examples/blueprint_mode_example.py
"""

import subprocess
import sys
import os

def run_blueprint_mode_example():
    """Run a simple blueprint mode example."""
    
    # Task description for a simple simulation
    task_description = (
        "Create a simple ecosystem simulation where different animal species "
        "compete for food resources in a forest environment. Include predator-prey "
        "relationships and population dynamics. The blueprint should be initialized "
        "from task understanding and capture entity interactions."
    )
    
    # Output directory for the blueprint
    output_dir = "./output/blueprint_example"
    
    # Construct the command
    cmd = [
        sys.executable,
        "main.py",
        "--task", task_description,
        "--mode", "blueprint", 
        "--output", output_dir,
        "--auto",  # Use automatic mode for the example
        "--iterations", "2"
    ]
    
    print("🎯 Running SOCIA Blueprint Mode Example")
    print("="*60)
    print(f"Task: {task_description}")
    print(f"Mode: blueprint")
    print(f"Output: {output_dir}")
    print("="*60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="..")
        
        print("Command output:")
        print(result.stdout)
        
        if result.stderr:
            print("Error output:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("\n✅ Blueprint mode example completed successfully!")
            print(f"Check the output directory: {output_dir}")
        else:
            print(f"\n❌ Blueprint mode example failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running blueprint mode example: {e}")

if __name__ == "__main__":
    # Change to the SOCIA root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    socia_root = os.path.dirname(script_dir)
    os.chdir(socia_root)
    
    run_blueprint_mode_example()

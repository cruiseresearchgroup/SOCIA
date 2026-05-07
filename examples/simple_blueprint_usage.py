#!/usr/bin/env python3
"""
Simple Blueprint Usage Example

This example demonstrates how to use the simplified Blueprint class
for storing and retrieving data in blueprint mode.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.blueprint import Blueprint

def demonstrate_blueprint_usage():
    """Demonstrate basic Blueprint class usage."""
    
    print("🔵 Blueprint Class Usage Example")
    print("=" * 50)
    
    # Create a new blueprint
    blueprint = Blueprint("Create a simple ecosystem simulation")
    print(f"✅ Created blueprint: {blueprint}")
    
    # Add some data using different methods
    print("\n📝 Adding data to blueprint...")
    
    # Method 1: Using set() method
    blueprint.set("entities", ["Predator", "Prey", "Environment"])
    blueprint.set("population_size", 1000)
    
    # Method 2: Using bracket notation
    blueprint["simulation_type"] = "agent_based"
    blueprint["time_steps"] = 100
    
    # Method 3: Using update() method
    blueprint.update({
        "parameters": {
            "birth_rate": 0.1,
            "death_rate": 0.05,
            "predation_rate": 0.02
        },
        "visualization": True
    })
    
    print(f"📊 Blueprint now contains {len(blueprint)} items")
    
    # Retrieve data using different methods
    print("\n🔍 Retrieving data from blueprint...")
    
    # Method 1: Using get() method with default
    entities = blueprint.get("entities", [])
    print(f"Entities: {entities}")
    
    # Method 2: Using bracket notation
    sim_type = blueprint["simulation_type"]
    print(f"Simulation type: {sim_type}")
    
    # Method 3: Checking if key exists
    if "parameters" in blueprint:
        params = blueprint.get("parameters")
        print(f"Parameters: {params}")
    
    # Display all keys and values
    print(f"\n📋 All blueprint keys: {list(blueprint.keys())}")
    
    # Save to JSON file
    print("\n💾 Saving blueprint to JSON file...")
    blueprint.save_to_file("example_blueprint.json")
    
    # Load from JSON file
    print("📂 Loading blueprint from JSON file...")
    new_blueprint = Blueprint()
    new_blueprint.load_from_file("example_blueprint.json")
    print(f"✅ Loaded blueprint with {len(new_blueprint)} items")
    
    # Display JSON representation
    print(f"\n📄 Blueprint JSON representation:")
    print(blueprint.to_json())
    
    # Clean up
    os.remove("example_blueprint.json")
    print("\n🧹 Cleaned up example file")

def simulate_agent_interactions():
    """Simulate how different agents might interact with the blueprint."""
    
    print("\n🤖 Agent Interaction Simulation")
    print("=" * 50)
    
    # Initialize blueprint
    blueprint = Blueprint("Multi-agent ecosystem simulation")
    
    # Simulate Task Understanding Agent
    print("\n🎯 Task Understanding Agent adds initial data...")
    blueprint.update({
        "task_type": "ecosystem_simulation",
        "requirements": ["predator-prey dynamics", "population tracking"],
        "success_criteria": "stable population over time"
    })
    
    # Simulate Code Generation Agent
    print("💻 Code Generation Agent adds code metadata...")
    blueprint.set("code_generated", True)
    blueprint.set("main_classes", ["Predator", "Prey", "Environment", "Simulation"])
    blueprint.set("key_functions", ["initialize", "step", "update_populations"])
    
    # Simulate Code Verification Agent
    print("🔍 Code Verification Agent adds verification results...")
    blueprint["verification_passed"] = True
    blueprint["syntax_errors"] = []
    blueprint["warnings"] = ["Consider adding type hints"]
    
    # Simulate Simulation Execution Agent
    print("🚀 Simulation Execution Agent adds execution results...")
    blueprint.update({
        "execution_successful": True,
        "execution_time": 2.34,
        "final_populations": {
            "predators": 150,
            "prey": 800,
            "total": 950
        }
    })
    
    # Simulate Feedback Generation Agent
    print("💬 Feedback Generation Agent adds feedback...")
    blueprint.set("feedback", {
        "positive": ["Simulation runs successfully", "Good population dynamics"],
        "improvements": ["Add visualization", "Optimize performance"]
    })
    
    print(f"\n📊 Final blueprint contains {len(blueprint)} items:")
    for key in blueprint.keys():
        print(f"  - {key}")
    
    print(f"\n📄 Complete blueprint structure:")
    print(blueprint.to_json())

if __name__ == "__main__":
    demonstrate_blueprint_usage()
    simulate_agent_interactions()

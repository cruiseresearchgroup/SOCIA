#!/usr/bin/env python3
"""
Blueprint Mode Dual-Layer Prompt Example

This example demonstrates the new dual-layer prompt design in blueprint mode:
- System Prompt: Loaded from task_understanding_blueprint_prompt.txt
- User Prompt: Contains specific task content with ODD protocol structure

The example shows how tasks are processed with ODD (Overview, Design concepts, Details) protocol.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demonstrate_dual_prompt_structure():
    """Demonstrate the dual-layer prompt structure used in blueprint mode."""
    
    print("🔵 Blueprint Mode Dual-Layer Prompt Design")
    print("=" * 60)
    
    # Example System Prompt (from task_understanding_blueprint_prompt.txt)
    system_prompt_example = """
You are the Task Understanding Agent specialized in blueprint mode for generating social simulations. 
Your role is to analyze research topics and simulation scenarios, then convert them into structured 
ODD (Overview, Design concepts, Details) protocol specifications.

Your responsibilities:
1. Parse the given research topic/task into clear simulation requirements
2. Structure the analysis following ODD protocol principles
3. Extract entities, behaviors, interactions, and parameters
4. Identify evaluation metrics and validation criteria
5. Output a complete JSON specification for blueprint initialization

Analysis Framework:
- OVERVIEW: What is the simulation's purpose and main research question?
- DESIGN CONCEPTS: What are the key entities, their attributes and behaviors?
- DETAILS: What are the specific parameters, interactions, and evaluation criteria?
"""
    
    print("📋 System Prompt Structure:")
    print(system_prompt_example.strip())
    
    # Example User Prompt
    original_topic = "Urban evacuation simulation during emergency scenarios"
    task_description = "Create a simulation to model how people evacuate from urban areas during emergencies like fires or natural disasters"
    
    user_prompt_example = f"""Please develop the following selected research question and simulation scenario
into an ODD protocol for a multi-agent simulation:

ORIGINAL RESEARCH TOPIC: {original_topic}

Additional context:
- Task description: {task_description}
- Data folder: emergency_evacuation_data/
- Available data files: ['building_layouts.json', 'population_density.csv', 'exit_routes.geojson']

Please analyze this research topic and provide a complete ODD-structured JSON specification suitable for blueprint initialization."""
    
    print(f"\n📝 User Prompt Structure:")
    print(user_prompt_example)
    
    # Example Expected Output Structure
    expected_output = {
        "title": "Urban Emergency Evacuation Simulation",
        "description": "Agent-based simulation of urban evacuation during emergency scenarios",
        "simulation_type": "agent_based",
        "odd_protocol": True,
        "processing_mode": "blueprint",
        "original_research_topic": original_topic,
        "overview": {
            "purpose": "Model evacuation behavior and identify bottlenecks",
            "entities": ["Person", "Building", "Exit", "Emergency"],
            "scale": "Urban district level"
        },
        "design_concepts": {
            "basic_principles": "Panic behavior, social influence, pathfinding",
            "emergence": "Crowd dynamics and bottleneck formation",
            "adaptation": "Route selection based on congestion",
            "objectives": "Find nearest safe exit",
            "learning": "Learn from observed congestion",
            "prediction": "Anticipate crowd movements",
            "sensing": "Detect nearby people and obstacles",
            "interaction": "Social influence on route choice",
            "stochasticity": "Random initial positions and panic levels",
            "collectives": "Crowd formation and movement",
            "observation": "Evacuation times and route usage"
        },
        "details": {
            "entities": [
                {
                    "name": "Person",
                    "attributes": ["position", "destination", "speed", "panic_level"],
                    "behaviors": ["move_to_exit", "follow_crowd", "avoid_obstacles"]
                },
                {
                    "name": "Building",
                    "attributes": ["layout", "capacity", "exit_points"],
                    "behaviors": ["provide_structure"]
                }
            ],
            "interactions": [
                {
                    "name": "crowd_following",
                    "description": "People follow others toward exits",
                    "entities_involved": ["Person", "Person"]
                }
            ],
            "parameters": {
                "population_size": 1000,
                "building_area": 10000,
                "num_exits": 5,
                "emergency_start_time": 0
            }
        }
    }
    
    print(f"\n📊 Expected ODD-Structured Output:")
    import json
    print(json.dumps(expected_output, indent=2))

def run_blueprint_mode_example():
    """Run a blueprint mode example with dual-layer prompts."""
    
    print(f"\n🚀 Running Blueprint Mode Example")
    print("=" * 60)
    
    # Task description for ODD protocol analysis
    task_description = (
        "Create an urban evacuation simulation to study how people evacuate "
        "from buildings during emergency scenarios, focusing on crowd dynamics "
        "and bottleneck identification using agent-based modeling."
    )
    
    # Output directory for the blueprint
    output_dir = "./output/blueprint_odd_example"
    
    # Show the command that would be run
    cmd_example = f"""python main.py \\
    --task "{task_description}" \\
    --mode blueprint \\
    --output {output_dir} \\
    --auto \\
    --iterations 2"""
    
    print("📝 Example Command:")
    print(cmd_example)
    
    print(f"\n🎯 This will:")
    print("1. Use blueprint-specific task understanding with ODD protocol")
    print("2. Apply dual-layer prompt design (System + User prompts)")
    print("3. Initialize blueprint with ODD-structured JSON output")
    print("4. Skip data analysis and model planning steps")
    print("5. Generate code using blueprint information")
    print("6. Save blueprint data at each iteration")
    
    print(f"\n📁 Expected Output Files:")
    print(f"- {output_dir}/task_spec_iter_1.json (ODD-structured)")
    print(f"- {output_dir}/blueprint_iter_1.json (Blueprint data)")
    print(f"- {output_dir}/simulation_code_iter_1.py (Generated code)")
    print(f"- {output_dir}/socia.log (Detailed logs)")

if __name__ == "__main__":
    demonstrate_dual_prompt_structure()
    run_blueprint_mode_example()

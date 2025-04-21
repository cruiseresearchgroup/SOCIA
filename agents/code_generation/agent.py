"""
CodeGenerationAgent: Generates simulation code based on the model plan.
"""

import logging
import os
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
        model_plan: Dict[str, Any],
        data_analysis: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate simulation code based on the model plan.
        
        Args:
            task_spec: Task specification from the Task Understanding Agent
            model_plan: Model plan from the Model Planning Agent
            data_analysis: Data analysis results from the Data Analysis Agent (optional)
            feedback: Feedback from previous iterations (optional)
        
        Returns:
            Dictionary containing the generated code and metadata
        """
        self.logger.info("Generating simulation code")
        
        # Build prompt from template
        prompt = self._build_prompt(
            task_spec=task_spec,
            model_plan=model_plan,
            data_analysis=data_analysis,
            feedback=feedback
        )
        
        # Call LLM to generate code
        llm_response = self._call_llm(prompt)
        
        # Extract code from the response
        # Since code generation typically produces Python code rather than JSON,
        # we handle the response differently
        code = self._extract_code(llm_response)
        
        # Generate a summary of the code
        code_summary = self._generate_code_summary(code)
        
        result = {
            "code": code,
            "code_summary": code_summary,
            "metadata": {
                "model_type": model_plan.get("model_type", "unknown"),
                "entities": [e.get("name") for e in model_plan.get("entities", [])],
                "behaviors": [b.get("name") for b in model_plan.get("behaviors", [])]
            }
        }
        
        self.logger.info("Code generation completed")
        return result
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from the LLM response.
        
        Args:
            response: The LLM's response
        
        Returns:
            The extracted code
        """
        # Look for code blocks marked with ```python and ```
        code_start = response.find("```python")
        if code_start >= 0:
            code_start += len("```python")
            code_end = response.find("```", code_start)
            if code_end >= 0:
                return response[code_start:code_end].strip()
        
        # If no Python code blocks found, look for generic code blocks
        code_start = response.find("```")
        if code_start >= 0:
            code_start += len("```")
            code_end = response.find("```", code_start)
            if code_end >= 0:
                return response[code_start:code_end].strip()
        
        # If no code blocks found, return the entire response
        # This is not ideal, but it's a fallback
        return response
    
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
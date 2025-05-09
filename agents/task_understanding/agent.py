"""
TaskUnderstandingAgent: Analyzes user requirements and converts them into structured simulation specifications.
"""

import logging
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent

class TaskUnderstandingAgent(BaseAgent):
    """
    Task Understanding Agent analyzes user requirements and converts them into structured
    simulation specifications that other agents can use.
    
    This agent is responsible for:
    1. Extracting key simulation requirements from natural language descriptions
    2. Identifying required entities, behaviors, and interactions
    3. Determining appropriate metrics and success criteria
    4. Structuring all this information into a consistent format for downstream agents
    """
    
    def process(self, task_description: str, task_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the task description and extract structured simulation requirements.
        
        Args:
            task_description: Natural language description of the simulation task
            task_data: Task data from JSON task description file (optional)
        
        Returns:
            Dictionary containing structured simulation requirements
        """
        self.logger.info("Processing task description")
        
        # If task_data is provided, use it to enhance the specification
        if task_data:
            self.logger.info("Using provided task data from JSON file")
            
            # Extract task objective
            task_objective = task_data.get("task_objective", {})
            description = task_objective.get("description", task_description)
            simulation_focus = task_objective.get("simulation_focus", [])
            
            # Extract data information
            data_folder = task_data.get("data_folder", "")
            data_files = task_data.get("data_files", {})
            self.logger.info(f"Data folder from task file: {data_folder}")
            self.logger.info(f"Data files specified: {list(data_files.keys())}")
            
            # Extract evaluation metrics
            evaluation_metrics = task_data.get("evaluation_metrics", {})
            
            # Create task specification from JSON data
            task_spec = {
                "title": "Simulation Task",
                "description": description,
                "simulation_focus": simulation_focus,
                "data_folder": data_folder,
                "data_files": data_files,
                "evaluation_metrics": evaluation_metrics
            }
            
            # Add more structure based on task description if needed
            enhanced_prompt = self._build_prompt(
                task_description=task_description,
                task_data=task_data
            )
            
            # If LLM is available, use it to enhance the specification
            llm_response = self._call_llm(enhanced_prompt)
            enhanced_spec = self._parse_llm_response(llm_response)
            
            # If LLM provided a valid response, merge it with task_spec
            if isinstance(enhanced_spec, dict):
                for key, value in enhanced_spec.items():
                    if key not in task_spec or not task_spec[key]:
                        task_spec[key] = value
            
            return task_spec
            
        else:
            # No task data provided, proceed with normal LLM-based analysis
            self.logger.info("No task data provided, using LLM-based analysis")
            
            # Build prompt from template
            prompt = self._build_prompt(task_description=task_description)
            
            # Call LLM to analyze the task
            llm_response = self._call_llm(prompt)
            
            # Parse the response
            task_spec = self._parse_llm_response(llm_response)
            
            # In a real implementation, validate the task specification structure
            # Here, we'll just return a placeholder for now
            if isinstance(task_spec, str):
                # If the response wasn't parsed as JSON, provide a placeholder structure
                task_spec = {
                    "title": "Simulation Task",
                    "description": task_description,
                    "simulation_type": "agent_based",
                    "entities": [
                        {
                            "name": "Person",
                            "attributes": ["location", "age", "interests"],
                            "behaviors": ["move", "interact"]
                        },
                        {
                            "name": "Location",
                            "attributes": ["position", "capacity", "type"],
                            "behaviors": []
                        }
                    ],
                    "interactions": [
                        {
                            "name": "person_visits_location",
                            "description": "Person agent visits a location based on interests",
                            "entities_involved": ["Person", "Location"]
                        }
                    ],
                    "parameters": {
                        "simulation_duration": 30,
                        "time_unit": "days",
                        "population_size": 1000
                    },
                    "metrics": [
                        {
                            "name": "location_popularity",
                            "description": "Number of visits to each location"
                        },
                        {
                            "name": "travel_distance",
                            "description": "Total distance traveled by each agent"
                        }
                    ],
                    "validation_criteria": [
                        {
                            "name": "visit_frequency_distribution",
                            "description": "Distribution of visit frequencies should match real data"
                        }
                    ]
                }
            
            self.logger.info("Task understanding completed")
            return task_spec 
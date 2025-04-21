"""
TaskUnderstandingAgent: Analyzes user requirements and converts them into structured simulation specifications.
"""

import logging
from typing import Dict, Any

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
    
    def process(self, task_description: str) -> Dict[str, Any]:
        """
        Process the task description and extract structured simulation requirements.
        
        Args:
            task_description: Natural language description of the simulation task
        
        Returns:
            Dictionary containing structured simulation requirements
        """
        self.logger.info("Processing task description")
        
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
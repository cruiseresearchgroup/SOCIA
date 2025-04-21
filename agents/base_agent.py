"""
BaseAgent: Abstract base class for all agents in the SOCIA system.
"""

import logging
import os
import json
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

from utils.llm_utils import get_llm_provider

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the SOCIA system.
    
    This class provides common functionality for all agents, including
    loading prompt templates, interacting with the LLM, and processing
    inputs and outputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config
        self.logger = logging.getLogger(f"SOCIA.{self.__class__.__name__}")
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        try:
            template_path = self.config.get("prompt_template")
            if not template_path:
                self.logger.warning("No prompt template specified, using default")
                return ""
            
            with open(template_path, 'r') as f:
                template = f.read()
            return template
        except Exception as e:
            self.logger.error(f"Error loading prompt template: {e}")
            return ""
    
    def _build_prompt(self, **kwargs) -> str:
        """
        Build a prompt by filling in the template with the provided arguments.
        
        Args:
            **kwargs: Keyword arguments to fill in the template
        
        Returns:
            The filled prompt template
        """
        prompt = self.prompt_template
        for key, value in kwargs.items():
            if isinstance(value, dict) or isinstance(value, list):
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)
            prompt = prompt.replace(f"{{{key}}}", value_str)
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            The LLM's response
        """
        self.logger.debug(f"Calling LLM with prompt: {prompt[:100]}...")
        
        # Get LLM configuration from global config
        try:
            # Get global configuration
            with open("config.yaml", 'r') as f:
                global_config = yaml.safe_load(f)
            
            llm_config = global_config.get("llm", {})
        except Exception as e:
            self.logger.error(f"Error loading global LLM configuration: {e}")
            llm_config = {"provider": "mock"}
        
        # Get LLM provider
        llm_provider = get_llm_provider(llm_config)
        
        # Call the LLM
        response = llm_provider.call(prompt)
        return response
    
    def _parse_llm_response(self, response: str) -> Any:
        """
        Parse the LLM's response based on the expected output format.
        
        Args:
            response: The LLM's response
        
        Returns:
            The parsed response
        """
        output_format = self.config.get("output_format", "text")
        
        if output_format == "json":
            try:
                # Extract JSON from the response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    self.logger.warning("Could not extract JSON from response")
                    return {"error": "Could not extract JSON from response"}
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON response: {e}")
                return {"error": f"Error parsing JSON response: {e}"}
        else:
            return response
    
    @abstractmethod
    def process(self, **kwargs) -> Any:
        """
        Process the inputs and generate outputs.
        
        Args:
            **kwargs: Input arguments specific to the agent
        
        Returns:
            The agent's output
        """
        pass 
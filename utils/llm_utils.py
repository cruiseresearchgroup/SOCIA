"""
Utilities for interacting with LLMs in the SOCIA system.
"""

import os
import json
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

def load_api_key(key_name: str) -> Optional[str]:
    """
    Load API key from keys.py file
    
    Args:
        key_name: Name of the API key, e.g., "OPENAI_API_KEY"
        
    Returns:
        Optional[str]: The API key value, or None if not found
    """
    try:
        # Import the keys module to access the hardcoded API key
        import keys
        # Return the hardcoded API key
        return getattr(keys, key_name, None)
    except ImportError:
        # Return None if keys.py doesn't exist
        return None

class LLMProvider:
    """
    Base class for LLM providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM provider.
        
        Args:
            config: Configuration dictionary for the LLM provider
        """
        self.config = config
        self.logger = logging.getLogger("SOCIA.LLMProvider")
    
    def call(self, prompt: str) -> str:
        """
        Call the LLM with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            The LLM's response
        """
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIProvider(LLMProvider):
    """
    LLM provider using OpenAI's API.
    """
    
    def call(self, prompt: str) -> str:
        """
        Call OpenAI's API with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            The LLM's response
        """
        try:
            import openai
            from openai import OpenAI
            
            # Get API key from keys.py file
            api_key = load_api_key("OPENAI_API_KEY")
            
            # Use API key from config only as fallback
            if not api_key:
                api_key = self.config.get("api_key")
                
            if not api_key:
                self.logger.error("OpenAI API key not found in keys.py")
                return "Error: OpenAI API key not found in keys.py"
            
            # Initialize client
            client = OpenAI(api_key=api_key)
            
            # Configure request parameters
            model = self.config.get("model", "gpt-4")
            temperature = self.config.get("temperature", 0.7)
            max_tokens = self.config.get("max_tokens", 4000)
            
            # Call the API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response text
            return response.choices[0].message.content
        
        except ImportError:
            self.logger.error("openai package not installed")
            return "Error: openai package not installed"
        
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return f"Error: {str(e)}"


class GeminiProvider(LLMProvider):
    """
    LLM provider using Google's Gemini API.
    """
    
    def call(self, prompt: str) -> str:
        """
        Call Google's Gemini API with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            The LLM's response
        """
        try:
            import google.generativeai as genai
            
            # Get API key from keys.py file
            api_key = load_api_key("GEMINI_API_KEY")
            
            # Use API key from config only as fallback
            if not api_key:
                api_key = self.config.get("api_key")
                
            if not api_key:
                self.logger.error("Gemini API key not found in keys.py")
                return "Error: Gemini API key not found in keys.py"
            
            # Configure the Gemini API
            genai.configure(api_key=api_key)
            
            # IMPORTANT: Use the exact model name from config
            if "model" not in self.config:
                self.logger.warning("Model not specified in config, using gemini-pro as fallback")
                model = "models/gemini-pro"
            else:
                # Get model name EXACTLY as specified in config, no default value
                model = self.config.get("model")
            
            temperature = self.config.get("temperature", 0.7)
            max_tokens = self.config.get("max_tokens", 8192)
            
            # Log the full model information
            self.logger.info(f"Using Gemini model: {model}")
            
            # Initialize model
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 0
            }
            
            # Get the generative model
            model_instance = genai.GenerativeModel(model_name=model,
                                            generation_config=generation_config)
            
            # Call the API
            response = model_instance.generate_content(prompt)
            
            # Extract response text
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                return ''.join([part.text for part in response.parts if hasattr(part, 'text')])
            else:
                self.logger.warning("Unexpected response format from Gemini")
                return str(response)
        
        except ImportError:
            self.logger.error("google-generativeai package not installed")
            return "Error: google-generativeai package not installed"
        
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {e}")
            return f"Error: {str(e)}"


class AnthropicProvider(LLMProvider):
    """
    LLM provider using Anthropic's API.
    """
    
    def call(self, prompt: str) -> str:
        """
        Call Anthropic's API with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            The LLM's response
        """
        try:
            # Get API key from keys.py file
            api_key = load_api_key("ANTHROPIC_API_KEY")
            
            # Use API key from config only as fallback
            if not api_key:
                api_key = self.config.get("api_key")
                
            if not api_key:
                self.logger.error("Anthropic API key not found in keys.py")
                return "Error: Anthropic API key not found in keys.py"
            
            # Implementation for Anthropic API
            # Currently just returns a placeholder message
            self.logger.warning("Anthropic provider not yet fully implemented")
            return "Error: Anthropic provider not yet fully implemented"
            
        except ImportError:
            self.logger.error("anthropic package not installed")
            return "Error: anthropic package not installed"
            
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {e}")
            return f"Error: {str(e)}"


class LlamaProvider(LLMProvider):
    """
    LLM provider using local Llama models.
    """
    
    def call(self, prompt: str) -> str:
        """
        Call a local Llama model with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            The LLM's response
        """
        try:
            # Check model path configuration
            model_path = self.config.get("model_path")
            if not model_path:
                self.logger.error("Llama model path not found in configuration")
                return "Error: Llama model path not configured"
            
            # Implementation for Llama model
            # Currently just returns a placeholder message
            self.logger.warning("Llama provider not yet fully implemented")
            return "Error: Llama provider not yet fully implemented"
            
        except ImportError:
            self.logger.error("llama package not installed")
            return "Error: llama package not installed"
            
        except Exception as e:
            self.logger.error(f"Error calling Llama model: {e}")
            return f"Error: {str(e)}"


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing and development.
    """
    
    def call(self, prompt: str) -> str:
        """
        Return a mock response based on the prompt.
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            A mock response
        """
        self.logger.info("Using mock LLM provider")
        
        # Return a simple mock response
        return "This is a mock response from the LLM provider."


def get_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    Get an LLM provider based on the provided configuration.
    
    Args:
        config: Configuration dictionary for the LLM provider
    
    Returns:
        An LLM provider instance
    """
    # Get the provider name from the config
    provider_name = config.get("provider", "mock").lower()
    
    # Load the global config to get provider-specific settings
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            global_config = yaml.safe_load(f)
        
        providers_config = global_config.get("llm_providers", {})
        
        # Get provider-specific config if available
        if provider_name in providers_config:
            provider_config = providers_config.get(provider_name, {})
            
            # Direct debugging of the actual model name coming from config
            if provider_name == "gemini":
                model_name = provider_config.get('model', 'NO_MODEL_FOUND_IN_CONFIG')
                logging.getLogger("SOCIA.LLMProvider").info(f"Loading Gemini model from config: {model_name}")
        else:
            provider_config = {}
            
        # Create and return the appropriate provider
        if provider_name == "openai":
            return OpenAIProvider(provider_config)
        elif provider_name == "gemini":
            return GeminiProvider(provider_config)
        elif provider_name == "anthropic":
            return AnthropicProvider(provider_config)
        elif provider_name == "llama":
            return LlamaProvider(provider_config)
        else:
            return MockProvider(provider_config)
    
    except Exception as e:
        logging.getLogger("SOCIA.LLMProvider").error(f"Error loading provider config: {e}")
        # Fallback to basic provider without specific config
        if provider_name == "openai":
            return OpenAIProvider({})
        elif provider_name == "gemini":
            return GeminiProvider({})
        elif provider_name == "anthropic":
            return AnthropicProvider({})
        elif provider_name == "llama":
            return LlamaProvider({})
        else:
            return MockProvider({}) 
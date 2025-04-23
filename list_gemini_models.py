#!/usr/bin/env python3
"""
Script to list all available Gemini models
"""

import os
import sys
import json
from pathlib import Path

def load_api_key(key_name: str):
    """Load API key from keys.py file"""
    try:
        import keys
        return getattr(keys, key_name, None)
    except ImportError:
        return None

def main():
    """Main function"""
    try:
        import google.generativeai as genai
        
        # Get API key from keys.py file
        api_key = load_api_key("GEMINI_API_KEY")
        
        if not api_key:
            print("Gemini API key not found in keys.py")
            return
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # List all available models
        print("Listing all available Gemini models:")
        print("-" * 50)
        
        models = genai.list_models()
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                print(f"Model name: {model.name}")
                print(f"Display name: {model.display_name}")
                print(f"Supported methods: {model.supported_generation_methods}")
                print(f"Input token limit: {model.input_token_limit}")
                print(f"Output token limit: {model.output_token_limit}")
                print("-" * 50)
                
    except ImportError:
        print("Error: google-generativeai package not installed")
        print("Please install it with: pip install google-generativeai")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
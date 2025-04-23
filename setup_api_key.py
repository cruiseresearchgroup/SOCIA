#!/usr/bin/env python3
"""
Script to set up API keys for LLM providers in the keys.py file
"""

import os
import sys
from pathlib import Path

def print_header():
    """Print script header"""
    print("\n=============================================")
    print("         SOCIA API Key Setup Tool")
    print("=============================================\n")

def get_provider_choice():
    """Get provider choice from user"""
    print("Select the LLM provider to configure:")
    print("1. OpenAI (GPT models)")
    print("2. Google Gemini")
    print("3. Configure both")
    choice = input("Choice (1-3) > ").strip()
    return choice

def get_openai_api_key():
    """Get OpenAI API key from user input"""
    print("\nPlease enter your OpenAI API key:")
    print("This will be stored in the keys.py file.")
    api_key = input("OpenAI API key > ").strip()
    return api_key

def get_gemini_api_key():
    """Get Gemini API key from user input"""
    print("\nPlease enter your Google Gemini API key:")
    print("This will be stored in the keys.py file.")
    api_key = input("Gemini API key > ").strip()
    return api_key

def save_to_keys_py(openai_key=None, gemini_key=None):
    """Save API keys to keys.py file"""
    file_path = Path("keys.py")
    
    # If the file already exists, read content and update
    if file_path.exists():
        with open(file_path, "r") as f:
            content = f.read()
        
        # Update OpenAI API key if provided
        if openai_key:
            import re
            if "OPENAI_API_KEY" in content:
                # Replace existing key
                content = re.sub(
                    r'OPENAI_API_KEY\s*=\s*"[^"]*"',
                    f'OPENAI_API_KEY = "{openai_key}"',
                    content
                )
            else:
                # Add new key
                content += f'\n\n# OpenAI API key\nOPENAI_API_KEY = "{openai_key}"\n'
        
        # Update Gemini API key if provided
        if gemini_key:
            import re
            if "GEMINI_API_KEY" in content:
                # Replace existing key
                content = re.sub(
                    r'GEMINI_API_KEY\s*=\s*"[^"]*"',
                    f'GEMINI_API_KEY = "{gemini_key}"',
                    content
                )
            else:
                # Add new key
                content += f'\n\n# Gemini API key\nGEMINI_API_KEY = "{gemini_key}"\n'
    else:
        # Create new file
        content = '"""\nFile for storing API keys. Do not commit this file to version control.\n"""\n\n'
        if openai_key:
            content += f'# OpenAI API key\nOPENAI_API_KEY = "{openai_key}"\n\n'
        if gemini_key:
            content += f'# Gemini API key\nGEMINI_API_KEY = "{gemini_key}"\n\n'
        content += '# Other potential API keys\n# ANTHROPIC_API_KEY = "your-anthropic-api-key-here"'
    
    # Write to file
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"\n✅ API key(s) successfully saved to {file_path}")

def main():
    """Main function"""
    print_header()
    
    choice = get_provider_choice()
    
    openai_key = None
    gemini_key = None
    
    if choice == '1' or choice == '3':
        openai_key = get_openai_api_key()
    
    if choice == '2' or choice == '3':
        gemini_key = get_gemini_api_key()
    
    if not openai_key and not gemini_key:
        print("No API keys provided, exiting...")
        return
    
    save_to_keys_py(openai_key, gemini_key)
    
    print("\nThe application will now read the API key(s) from keys.py.")
    print("To test your configuration, update the provider in config.yaml and run:")
    print("python main.py --run-example")

if __name__ == "__main__":
    main() 
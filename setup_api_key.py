#!/usr/bin/env python3
"""
Script to set up the OpenAI API key in the keys.py file
"""

import os
import sys
from pathlib import Path

def print_header():
    """Print script header"""
    print("\n=============================================")
    print("         SOCIA API Key Setup Tool")
    print("=============================================\n")

def get_api_key():
    """Get API key from user input"""
    print("Please enter your OpenAI API key:")
    print("This will be stored in the keys.py file.")
    api_key = input("API key > ").strip()
    return api_key

def save_to_keys_py(api_key):
    """Save API key to keys.py file"""
    file_path = Path("keys.py")
    
    # If the file already exists, read content and update
    if file_path.exists():
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check if OPENAI_API_KEY already exists
        if "OPENAI_API_KEY" in content:
            # Replace existing key
            import re
            content = re.sub(
                r'OPENAI_API_KEY\s*=\s*"[^"]*"',
                f'OPENAI_API_KEY = "{api_key}"',
                content
            )
        else:
            # Add new key
            content += f'\n\nOPENAI_API_KEY = "{api_key}"\n'
    else:
        # Create new file
        content = f'"""\nFile for storing API keys. Do not commit this file to version control.\n"""\n\n# OpenAI API key\nOPENAI_API_KEY = "{api_key}"\n'
    
    # Write to file
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"\n✅ API key successfully saved to {file_path}")

def main():
    """Main function"""
    print_header()
    
    api_key = get_api_key()
    if not api_key:
        print("No API key provided, exiting...")
        return
    
    save_to_keys_py(api_key)
    
    print("\nThe application will now read the API key from keys.py.")
    print("To test your configuration, run:")
    print("python main.py --run-example")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""Check available Gemini models in the installed google.generativeai package."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

import google.generativeai as genai

api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not set. Available models may be limited.")

# List available models
print("=" * 80)
print("Available Gemini Models")
print("=" * 80)

try:
    models = genai.list_models()
    for model in models:
        print(f"\nModel: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        if hasattr(model, 'input_token_limit'):
            print(f"  Input token limit: {model.input_token_limit}")
        if hasattr(model, 'output_token_limit'):
            print(f"  Output token limit: {model.output_token_limit}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\n" + "=" * 80)
print("Available Classes in google.generativeai")
print("=" * 80)
print(f"genai.GenerativeModel: {hasattr(genai, 'GenerativeModel')}")
print(f"genai.TextGenerationModel: {hasattr(genai, 'TextGenerationModel')}")
print(f"genai.ChatSession: {hasattr(genai, 'ChatSession')}")

# Attempt to use gen-2 model
print("\n" + "=" * 80)
print("Testing gemini-2.0-flash")
print("=" * 80)
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Hello")
    print("✓ gemini-2.0-flash works!")
except Exception as e:
    print(f"✗ gemini-2.0-flash error: {e}")

# Attempt to use flash model
print("\n" + "=" * 80)
print("Testing gemini-1.5-flash")
print("=" * 80)
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Hello")
    print("✓ gemini-1.5-flash works!")
except Exception as e:
    print(f"✗ gemini-1.5-flash error: {e}")

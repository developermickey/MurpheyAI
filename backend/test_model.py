#!/usr/bin/env python3
"""
Test script to verify the model is loading and generating responses correctly.
"""
import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.model_service import model_service
from app.core.config import settings

async def test_model():
    """Test the model generation."""
    print("=" * 60)
    print("Testing AI Model")
    print("=" * 60)
    print(f"Model Name: {settings.MODEL_NAME}")
    print(f"Model Path: {settings.MODEL_PATH or '(Using Hugging Face)'}")
    print()
    
    test_prompts = [
        "Hello! How are you?",
        "What is artificial intelligence?",
        "Tell me a joke."
    ]
    
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        print("Response: ", end="", flush=True)
        
        try:
            response_chunks = []
            async for chunk in model_service.generate(
                prompt=prompt,
                model_name=settings.MODEL_NAME,
                temperature=0.7,
                max_tokens=100,
                stream=True
            ):
                response_chunks.append(chunk)
                print(chunk, end="", flush=True)
            
            print("\n" + "-" * 60 + "\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_model())


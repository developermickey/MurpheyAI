#!/usr/bin/env python3
"""
Test script to verify AI model responses via the API.
"""
import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_ai_response():
    """Test AI model response through the API."""
    print("=" * 60)
    print("Testing AI Model Responses")
    print("=" * 60)
    
    # First, login to get token
    print("\n1. Logging in...")
    login_response = requests.post(
        f"{API_URL}/api/v1/auth/login",
        json={"username": "admin@test.com", "password": "admin123"}
    )
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.text}")
        return
    
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Logged in successfully")
    
    # Test prompts
    test_prompts = [
        "Hello! What is your name?",
        "Explain artificial intelligence in one sentence.",
        "What can you help me with?"
    ]
    
    print("\n2. Testing AI responses...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: {prompt}")
        print("-" * 60)
        
        try:
            response = requests.post(
                f"{API_URL}/api/v1/chat/message",
                headers=headers,
                json={
                    "content": prompt,
                    "conversation_id": None,
                    "model": "gpt2",
                    "temperature": 0.7,
                    "max_tokens": 150
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "response" in result:
                    print(f"✅ Response: {result['response']}")
                elif "error" in result:
                    print(f"⚠️  Error: {result['error']}")
                else:
                    print(f"Response: {json.dumps(result, indent=2)}")
            else:
                print(f"❌ API Error ({response.status_code}): {response.text}")
        
        except requests.exceptions.Timeout:
            print("⏳ Request timed out (model might be loading for first time)")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n")
    
    print("=" * 60)
    print("Test complete!")
    print("\nNote: First request may take longer as the model downloads/loads.")
    print("Subsequent requests will be faster.")

if __name__ == "__main__":
    test_ai_response()


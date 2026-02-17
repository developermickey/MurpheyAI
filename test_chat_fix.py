#!/usr/bin/env python3
"""Test script to verify chat fix without Python crashes."""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing MurpheyAI Chat Generation Fix")
print("=" * 60)

# Step 1: Login
print("\n1. Logging in...")
login_response = requests.post(
    f"{BASE_URL}/api/v1/auth/login",
    json={"email": "admin@test.com", "password": "admin123"},
    timeout=10
)

if login_response.status_code != 200:
    print(f"❌ Login failed: {login_response.status_code}")
    print(login_response.text)
    exit(1)

token = login_response.json().get("access_token")
print(f"✅ Login successful! Token: {token[:20]}...")

headers = {"Authorization": f"Bearer {token}"}

# Step 2: Send a message
print("\n2. Sending test message to chat endpoint...")
test_prompts = [
    "Hello, how are you?",
    "What is machine learning?",
    "Tell me a joke",
]

for prompt in test_prompts:
    print(f"\n   Testing: '{prompt}'")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/message",
            json={"content": prompt, "model": "small"},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")[:100]
            error = result.get("error")
            
            if error:
                print(f"   ⚠️  Error returned (no crash): {error[:50]}...")
            else:
                print(f"   ✅ Got response: {response_text}...")
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text[:50]}")
    except Exception as e:
        print(f"   ❌ Exception: {str(e)[:100]}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)

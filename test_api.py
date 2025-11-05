import requests
import json

# Test health endpoint first
health_url = "https://medrag-ai-engine.onrender.com/health"
print("Testing health endpoint...")
try:
    health_response = requests.get(health_url, timeout=30)
    print(f"Health Status: {health_response.status_code}")
    if health_response.status_code == 200:
        print(f"Health Response: {health_response.json()}")
    else:
        print(f"Health Error: {health_response.text}")
except Exception as e:
    print(f"Health check failed: {e}")

print("\n" + "="*50 + "\n")

# Test diagnose endpoint
diagnose_url = "https://medrag-ai-engine.onrender.com/diagnose"
data = {
    "query": "Patient has chest pain, shortness of breath, and fatigue for 3 days",
    "k": 5
}

print("Testing diagnose endpoint...")
try:
    response = requests.post(diagnose_url, json=data, timeout=60)
    print(f"Diagnose Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
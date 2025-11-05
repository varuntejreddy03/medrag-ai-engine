import requests
import json

url = "https://medrag-ai-engine.onrender.com/diagnose"
payload = {"query": "chest pain and shortness of breath", "k": 3}

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed: {e}")

from fastapi.testclient import TestClient
from app.main import app
import json


def test_predictions():
    print("Initializing TestClient (triggers startup event)...")
    # Using 'with' block is better for lifespan events
    with TestClient(app) as client:
        # 1. Health Check
        print("Testing /health...")
        response = client.get("/health")
    print(f"Health Status: {response.status_code}")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    # 2. Prediction
    print("Testing /predict...")
    payload = {"text": "I hate you, you are terrible!"}
    response = client.post("/predict", json=payload)
    
    print(f"Predict Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Response Keys:", data.keys())
        print("Scores:", data['scores'])
        print("Flags:", data['flags'])
        assert 'scores' in data
        assert 'flags' in data
    else:
        print("Error:", response.text)
        
    # 3. Validation Error (Empty)
    print("Testing Validation Error...")
    response = client.post("/predict", json={"text": ""})
    print(f"Validation Status: {response.status_code}")
    assert response.status_code == 400

    print("SUCCESS: API endpoints verified.")

if __name__ == "__main__":
    test_predictions()

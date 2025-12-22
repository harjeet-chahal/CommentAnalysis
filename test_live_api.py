
import subprocess
import time
import requests
import sys
import os

def test_live_server():
    # Start server in background
    print("Starting Uvicorn...")
    server = subprocess.Popen(
        ["uvicorn", "app.main:app", "--port", "8000"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    

    # Wait for startup
    time.sleep(15)
    
    try:
        if server.poll() is not None:
            print("Server exited prematurely!")
            out, err = server.communicate()
            print("STDOUT:", out.decode())
            print("STDERR:", err.decode())
            sys.exit(1)
            
        # 1. Health
        print("Testing /health...")
        try:
            r = requests.get("http://127.0.0.1:8000/health")
            print(r.json())
            assert r.status_code == 200
        except Exception as e:
            print(f"Health Check Failed: {e}")
            sys.exit(1)
            
        # 2. Predict
        print("Testing /predict...")
        payload = {"text": "This is a test comment."}
        r = requests.post("http://127.0.0.1:8000/predict", json=payload)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print(r.json())
        else:
            print(r.text)
            
        assert r.status_code == 200
        
        print("SUCCESS: Live server verification passed.")
        
    finally:
        print("Killing server...")
        server.terminate()
        server.wait()

if __name__ == "__main__":
    test_live_server()

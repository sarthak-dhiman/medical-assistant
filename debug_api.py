import requests
import base64
import cv2
import numpy as np
import time

# Configuration
API_URL = "http://localhost:8000/predict"
# Create a dummy image (black/white)
img = np.zeros((480, 640, 3), dtype=np.uint8)
# Add some "skin" color to trigger skin detection
img[:, :] = (200, 200, 200) # BGR
_, buffer = cv2.imencode('.jpg', img)
img_b64 = base64.b64encode(buffer).decode('utf-8')

payload = {
    "image": img_b64,
    "mode": "SKIN_DISEASE"
}

print(f"Sending request to {API_URL}...")
try:
    start = time.time()
    response = requests.post(API_URL, json=payload)
    end = time.time()
    
    print(f"Status Code: {response.status_code}")
    print(f"Time: {end - start:.2f}s")
    print("Task ID:", response.json().get("task_id"))
    task_id = response.json().get("task_id")
    
    # Poll for result
    for _ in range(10):
        time.sleep(1)
        res = requests.get(f"http://localhost:8000/result/{task_id}")
        data = res.json()
        print("Poll State:", data.get("state"))
        if data.get("state") == "SUCCESS":
            print("Result:", data.get("result"))
            break
        elif data.get("state") == "FAILURE":
            print("Worker Failed:", data)
            break
except Exception as e:
    print(f"Request failed: {e}")


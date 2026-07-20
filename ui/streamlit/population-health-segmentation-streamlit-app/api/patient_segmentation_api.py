
import requests
API_URL="http://127.0.0.1:8000/api/v1/patient-segmentation"

def predict(payload):
    r=requests.post(f"{API_URL}/predict",json=payload,timeout=6000)
    r.raise_for_status()
    return r.json()

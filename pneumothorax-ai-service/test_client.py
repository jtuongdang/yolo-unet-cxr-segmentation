import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print(f"Testing health endpoint at {BASE_URL}/health ...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        print("Health status code:", r.status_code)
        print("Health response:", r.json())
    except Exception as e:
        print("Health check failed:", e)
        return

    print("\nTesting demo-cases endpoint...")
    r = requests.get(f"{BASE_URL}/api/demo-cases", timeout=10)
    print("Demo cases count:", len(r.json().get("cases", [])))

    print("\nTesting prediction on Case 01: Severe Pneumothorax...")
    payload = {"filename": "01_Severe_Large_Pneumothorax.png"}
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/api/predict/demo", json=payload, timeout=60)
    t_elapsed = round(time.time() - t0, 2)
    print(f"Prediction returned status {r.status_code} in {t_elapsed}s")
    
    if r.status_code == 200:
        data = r.json()
        print("Exam ID:", data.get("exam_id"))
        print("Triage Alert:", data.get("triage_alert"))
        print("Biomarkers:", data.get("biomarkers"))
        print("Total Lesions Found:", data.get("findings", {}).get("total_lesions_count"))
        print("Artifacts generated:", data.get("artifacts"))

if __name__ == "__main__":
    test_api()

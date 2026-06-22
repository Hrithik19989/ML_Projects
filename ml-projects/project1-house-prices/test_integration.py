import os
import pytest
import requests

# Fetch the live production URL from environment variable, fallback to local test instance
BACKEND_URL = os.environ.get("BACKEND_URL", "https://onrender.com").rstrip("/")

@pytest.fixture
def valid_payload():
    return {
        "transaction_date": 2013.25,
        "house_age": 15.0,
        "distance_to_mrt": 350.5,
        "convenience_stores": 4,
        "latitude": 24.98298,
        "longitude": 121.54024
    }

def test_production_health_check():
    """Verify backend is alive using the explicit HEAD/GET root method."""
    response = requests.get(f"{BACKEND_URL}/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_production_inference(valid_payload):
    """Verify end-to-end payload routing and ML output prediction generation."""
    response = requests.post(f"{BACKEND_URL}/predict", json=valid_payload, timeout=10)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert "features_evaluated" in data

from fastapi.testclient import TestClient
from app.main import app # Adjust based on where this file sits


client = TestClient(app)

def test_api_health_and_prediction():
    """Test if valid data returns a valid 200 HTTP response and a price."""
    payload = {
        "distance_to_mrt": 250.5,
        "num_convenience_stores": 4,
        "latitude": 24.98,
        "longitude": 121.54
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price_per_unit" in response.json()
    assert response.json()["status"] == "success"

def test_api_invalid_data():
    """Test if negative numbers fail strict validation (should return 422 error)."""
    invalid_payload = {
        "distance_to_mrt": -50.0,  # Negative distance breaks our Pydantic rule
        "num_convenience_stores": 3,
        "latitude": 24.98,
        "longitude": 121.54
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # 422 means Unprocessable Entity

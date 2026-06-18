import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import app.main

# Initialize the TestClient using the app instance
@pytest.fixture
def client():
    with TestClient(app.main.app) as c:
        yield c

def test_read_root(client):
    """Test the root endpoint returns operational status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_predict_success(client, monkeypatch):
    """Test that valid input structure yields a successful prediction response."""
    # Mocking model inference to isolate API test behavior from local pickle dependencies
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    mock_model = MagicMock()
    mock_model.predict.return_value = [42.5]
    
    
    monkeypatch.setattr(app.main, "scaler", mock_scaler)
    monkeypatch.setattr(app.main, "model", mock_model)

    # Valid payload as a direct object dictionary (fixes original array 422 error)
    json_payload = {
        "transaction_date": 2013.25,
        "house_age": 32.0,
        "distance_to_mrt": 350.5,
        "convenience_stores": 4,
        "latitude": 24.98298,
        "longitude": 121.54024
    }
    
    response = client.post("/predict", json=json_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == 42.5
    assert data["unit"] == "per unit area"

def test_api_predict_invalid_data(client):
    """Test that missing required fields correctly raises a 422 error."""
    # Missing required structural properties
    invalid_payload = {
        "distance_to_mrt": 350.5,
        "convenience_stores": 4
    }
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


if __name__ == "__main__":
    import pytest
    import sys
    # Triggers pytest programmatically on this file and prints output
    sys.exit(pytest.main([__file__, "-v"]))

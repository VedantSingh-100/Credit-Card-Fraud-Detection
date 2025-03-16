# test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app from main.py

client = TestClient(app)

def test_predict_missing_features_key():
    """
    Test the /predict endpoint when the 'features' key is missing.
    This should trigger a validation error.
    """
    payload = {}  # Missing 'features'
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"Expected status code 422, got {response.status_code}"

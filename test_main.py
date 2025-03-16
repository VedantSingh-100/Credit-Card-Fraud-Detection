# test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app from main.py

client = TestClient(app)

def test_predict_valid_input():
    """
    Test the /predict endpoint with a valid input.
    The input should be a JSON with a 'features' key containing a list of 31 float values.
    """
    payload = {"features": [0.1] * 31}  # Replace 31 with the correct input dimension if different
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    
    result = response.json()
    # Check that the result contains 'fraud_probability'
    assert "fraud_probability" in result, "Response JSON does not contain 'fraud_probability'"
    # Verify that the fraud probability is a float between 0 and 1
    prob = result["fraud_probability"]
    assert isinstance(prob, float), "fraud_probability is not a float"
    assert 0.0 <= prob <= 1.0, "fraud_probability is not between 0 and 1"

def test_predict_invalid_input_length():
    """
    Test the /predict endpoint with an input that has an incorrect number of features.
    This should trigger a validation error from FastAPI/Pydantic.
    """
    payload = {"features": [0.1] * 10}  # Incorrect length: expecting 31 values
    response = client.post("/predict", json=payload)
    # FastAPI's validation errors typically return status code 422
    assert response.status_code == 422, f"Expected status code 422, got {response.status_code}"

def test_predict_missing_features_key():
    """
    Test the /predict endpoint when the 'features' key is missing.
    This should trigger a validation error.
    """
    payload = {}  # Missing 'features'
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"Expected status code 422, got {response.status_code}"

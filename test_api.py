from fastapi.testclient import TestClient
from main import app
import requests


client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the model inference API!"}



def test_inference_less_than_50k():
    response = client.post("/inference/", json={
        "age": 25,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}



def test_inference_more_than_50k():
    # Adjusting features like education, age, and capital gain to likely push the prediction over 50K
    response = client.post("/inference/", json={
        "age": 45,  # Older, suggesting more work experience
        "workclass": "Self-emp-inc",  # Self-employed (incorporated) often have higher earnings
        "fnlgt": 123456,
        "education": "Masters",
        "education-num": 16,  # Higher educational attainment
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",  # Typically higher-paying job category
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,  # Non-zero capital gains
        "capital-loss": 0,
        "hours-per-week": 60,  # Working more hours per week
        "native-country": "United-States"
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}



def test_inference_incomplete_data():
    request_data = {
        "age": 30,
        "workclass": "Private",
        # Missing other required fields
    }
    response = client.post("/inference/", json=request_data)
    assert response.status_code == 422  # HTTP 422 Unprocessable Entity for validation errors


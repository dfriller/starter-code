from fastapi.testclient import TestClient
import json
import logging

from main import app

client = TestClient(app)

def test_root():
    """
    Test the welcome message at the root endpoint
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to the model inference API!"

def test_inference():
    """
    Test the model inference output for a sample with expected prediction '>50K'
    """
    sample = {
        'age': 50,
        'workclass': "Private",
        'fnlgt': 234721,
        'education': "Doctorate",
        'education_num': 16,
        'marital_status': "Separated",
        'occupation': "Exec-managerial",
        'relationship': "Not-in-family",
        'race': "Black",
        'sex': "Female",
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': "United-States"
    }

    response = client.post("/inference/", json=sample)

    # Validate response and output
    assert response.status_code == 200
    result = response.json()
    assert result["age"] == 50
    assert result["fnlgt"] == 234721

    # Log and assert prediction
    logging.info(f'********* prediction = {result["prediction"]} ********')
    assert result["prediction"] == '>50K'

def test_inference_class0():
    """
    Test the model inference output for a sample with expected prediction '<=50K'
    """
    sample = {
        'age': 30,
        'workclass': "Private",
        'fnlgt': 234721,
        'education': "HS-grad",
        'education_num': 1,
        'marital_status': "Separated",
        'occupation': "Handlers-cleaners",
        'relationship': "Not-in-family",
        'race': "Black",
        'sex': "Male",
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 35,
        'native_country': "United-States"
    }

    response = client.post("/inference/", json=sample)

    # Validate response and output
    assert response.status_code == 200
    result = response.json()
    assert result["age"] == 30
    assert result["fnlgt"] == 234721

    # Log and assert prediction
    logging.info(f'********* prediction = {result["prediction"]} ********')
    assert result["prediction"][0] == '<=50K'

def test_incomplete_inference_query():
    """
    Test that an incomplete sample does not generate a prediction
    """
    sample = {
        'age': 50,
        'workclass': "Private",
        'fnlgt': 234721
    }

    response = client.post("/inference/", json=sample)

    assert 'prediction' not in response.json().keys()
    logging.warning(f"The sample has {len(sample)} features. Must have 14 features.")

if __name__ == '__main__':
    test_root()
    test_inference()
    test_inference_class0()
    test_incomplete_inference_query()

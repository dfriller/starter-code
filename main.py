from fastapi import FastAPI, HTTPException
from typing import Union, Optional
from pydantic import BaseModel
import pandas as pd
import os
import pickle
import logging
from ml.data import process_data

# Configuration
savepath = 'model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Define data model
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        json_schema_extra = {
            "example": {
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
        }

# Instantiate FastAPI app
app = FastAPI(
    title="Inference API",
    description="An API that takes a sample and runs an inference",
    version="1.0.0"
)

# Load model artifacts on startup of the application to reduce latency
@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    if os.path.isfile(os.path.join(savepath, filename[0])):
        model = pickle.load(open(os.path.join(savepath, filename[0]), "rb"))
        encoder = pickle.load(open(os.path.join(savepath, filename[1]), "rb"))
        lb = pickle.load(open(os.path.join(savepath, filename[2]), "rb"))

# Define endpoints
@app.get("/")
async def greetings():
    return "Welcome to the model inference API!"

@app.post("/inference/")
async def ingest_data(inference: InputData):
    data = {
        'age': inference.age,
        'workclass': inference.workclass,
        'fnlgt': inference.fnlgt,
        'education': inference.education,
        'education-num': inference.education_num,
        'marital-status': inference.marital_status,
        'occupation': inference.occupation,
        'relationship': inference.relationship,
        'race': inference.race,
        'sex': inference.sex,
        'capital-gain': inference.capital_gain,
        'capital-loss': inference.capital_loss,
        'hours-per-week': inference.hours_per_week,
        'native-country': inference.native_country,
    }
    sample = pd.DataFrame(data, index=[0])

    # Apply transformation to sample data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    if os.path.isfile(os.path.join(savepath, filename[0])):
        model = pickle.load(open(os.path.join(savepath, filename[0]), "rb"))
        encoder = pickle.load(open(os.path.join(savepath, filename[1]), "rb"))
        lb = pickle.load(open(os.path.join(savepath, filename[2]), "rb"))

    sample, _, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Get model prediction which is a one-dim array like [1]
    prediction = model.predict(sample)

    # Convert prediction to label and add to data output
    if prediction[0] > 0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K'
    data['prediction'] = prediction

    return data

if __name__ == '__main__':
    pass












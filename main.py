import pandas as pd
# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from models import InferenceItem
from typing import List
import pickle
from ml.data import process_data
import os

app = FastAPI()
model = None  # Global variable to hold the loaded model
encoder = None
lb = None

savepath = './model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']
@app.get("/")
async def root():
    return {"message": "Welcome to the model inference API!"}


@app.on_event("startup")
async def load_model():
    global model, encoder, lb
    try:
        with open('model/trained_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('model/encoder.pkl', 'rb') as file:
            encoder = pickle.load(file)
        with open('model/labelizer.pkl', 'rb') as file:
            lb = pickle.load(file)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise



@app.post("/inference/")
async def do_inference(item: InferenceItem):
    # Convert Pydantic model instance to a dictionary
    data_dict = item.dict(by_alias=True)  # by_alias=True to use field aliases if set in the Pydantic model

    # Create a DataFrame
    data_df = pd.DataFrame([data_dict])

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

    # Process the data and predict
    # Assuming process_data and model.predict are properly defined
    # Example:
    if os.path.isfile(os.path.join(savepath,filename[0])):
        model = pickle.load(open(os.path.join(savepath,filename[0]), "rb"))
        encoder = pickle.load(open(os.path.join(savepath,filename[1]), "rb"))
        lb = pickle.load(open(os.path.join(savepath,filename[2]), "rb"))

    processed_data, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,  # you need to define this list
        training=False,
        encoder=encoder,
        lb=lb
    )


    prediction = model.predict(processed_data)
    prediction_label = '>50K' if prediction[0] > 0.5 else '<=50K'
    return {"prediction": prediction_label}

    return data


if __name__ == '__main__':
    pass


# app_fastapi.py
# Day 2, Session 3 - Deploying a model as a REST API (FastAPI)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Model API")
model = joblib.load("iris_model.pkl")

class IrisFeatures(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Iris Model API is running!"}

@app.post("/predict")
def predict(data: IrisFeatures):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

# Run with:  uvicorn app_fastapi:app --reload --port 5001
# Interactive docs: http://localhost:5001/docs

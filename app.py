from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import create_model, Field
import mlflow.sklearn
import numpy as np
import json
import uvicorn
import os

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# mlflow.set_tracking_uri("http://localhost:5000")

# Load model
model = mlflow.sklearn.load_model("models:/ChurnModel/1")

# Load feature metadata
with open("models/feature_metadata.json", "r") as f:
    feature_names = json.load(f)

# Dynamically create Pydantic model
fields = {name: (float, Field(...)) for name in feature_names}  # Field(...) makes it required
CustomerFeatures = create_model("CustomerFeatures", **fields)

@app.post("/predict")
def predict(customer: CustomerFeatures):
    features = [getattr(customer, name) for name in feature_names]
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)
    return {"churn": bool(prediction[0])}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8060, reload=True)
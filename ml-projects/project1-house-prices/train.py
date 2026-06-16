import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import PredictionInput

app = FastAPI(
    title="House Price Prediction API",
    description="Inference API for predicting house prices based on geographic and structural features.",
    version="1.0.0"
)

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")

# Load model and scaler at startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts. Ensure train.py has run successfully. Error: {e}")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the House Price Prediction API. Go to /docs for interactive testing."
    }

@app.post("/predict", summary="Predict House Price")
def predict(payload: PredictionInput):
    try:
        # Extract features in the exact order your model expects
        raw_features = np.array([[
            payload.transaction_date,
            payload.house_age,
            payload.distance_to_mrt,
            payload.convenience_stores,
            payload.latitude,
            payload.longitude
        ]])
        
        # Apply the pre-fitted scaling transformation
        scaled_features = scaler.transform(raw_features)
        
        # Generate prediction output
        prediction = model.predict(scaled_features)
        
        return {
            "prediction": float(prediction[0]),
            "unit": "per unit area"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

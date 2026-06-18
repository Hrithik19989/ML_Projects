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

# Define path mapping references
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "artifacts", "features.pkl")

def load_artifacts():
    """Safely loads model artifacts, triggering train.py if missing."""
    global model, scaler, expected_features
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        expected_features = joblib.load(FEATURES_PATH)
    except Exception:
        print("Artifacts missing. Triggering on-the-fly training pipeline...")
        try:
            import subprocess
            train_script = os.path.join(BASE_DIR, "train.py")
            subprocess.run(["python", train_script], check=True)
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            expected_features = joblib.load(FEATURES_PATH)
        except Exception as e:
            model, scaler, expected_features = None, None, None
            print(f"Critical error loading or training model artifacts: {e}")

# Trigger loading on app initialization
load_artifacts()

@app.get_route("/", methods=["GET", "HEAD"])
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the House Price Prediction API. Go to /docs for interactive testing."
    }

@app.post("/predict", summary="Predict House Price")
def predict(payload: PredictionInput):
    if not all([model, scaler, expected_features]):
        raise HTTPException(status_code=503, detail="Model artifacts are incomplete or missing on server.")
        
    try:
        # Structure the input raw features array
        raw_features = np.array([[
            payload.transaction_date,
            payload.house_age,
            payload.distance_to_mrt,
            payload.convenience_stores,
            payload.latitude,
            payload.longitude
        ]])
        
        # Checked column dimension shape indexing (.shape[1]) against expected_features length
        if raw_features.shape[1] != len(expected_features):
            raise HTTPException(
                status_code=400, 
                detail=f"Feature mismatch. Expected {len(expected_features)} inputs, got {raw_features.shape[1]}."
            )
        
        # Run preprocessing and model inference
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)
        
        return {
            "prediction": float(prediction[0]),
            "unit": "per unit area",
            "features_evaluated": expected_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

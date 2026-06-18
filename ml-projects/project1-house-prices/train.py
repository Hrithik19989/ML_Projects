import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_xgb_pipeline():
    # 1. Setup explicit project directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Real-estate1.csv")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    features_path = os.path.join(artifacts_dir, "features.pkl")

    # 2. Load the actual CSV file
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find the dataset at {csv_path}")
        
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 3. Clean up column mappings
    df.columns = df.columns.str.strip()
    
    # Identify feature column patterns matching structural dataset inputs
    feature_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'age', 'mrt', 'store', 'lat', 'long'])]
    
    if len(feature_cols) < 6:
        print("Mapping features by positional indices (Columns 1 to 6)...")
        X = df.iloc[:, 1:7]  # Drops index/ID column, takes 6 structural inputs
        y = df.iloc[:, -1]   # Takes the final house price column
    else:
        X = df[feature_cols]
        y = df.iloc[:, -1]

    print(f"Features shape: {X.shape} | Target shape: {y.shape}")
    print(f"Features used: {list(X.columns)}")

    # 4. Split and Scale dataset structures
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train the XGBoost Regressor Pipeline
    print("Training XGBoost Regressor model...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Print scoring performance
    print(f"-> Train R² Score: {xgb_model.score(X_train_scaled, y_train):.4f}")
    print(f"-> Test R² Score: {xgb_model.score(X_test_scaled, y_test):.4f}")

    # 6. Save out the artifact binary dependencies (including features list)
    feature_names = list(X.columns)
    
    joblib.dump(xgb_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)
    
    print(f"✅ Successfully exported scaler to {scaler_path}")
    print(f"✅ Successfully exported XGBoost model to {model_path}")
    print(f"✅ Successfully exported feature names list to {features_path}")

if __name__ == "__main__":
    train_xgb_pipeline()

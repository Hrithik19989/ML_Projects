import streamlit as st
import requests
import os

# Configure Streamlit page UI
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction Dashboard")
st.write("Enter the structural and geographic properties below to estimate property valuation per unit area.")

# Form inputs configured to match your Pydantic schemas layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_date = st.number_input("Transaction Year (e.g., 2013.25)", min_value=2000.0, max_value=2050.0, value=2013.25, step=0.01)
        house_age = st.number_input("House Age (Years)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        distance_to_mrt = st.number_input("Distance to MRT Station (Meters)", min_value=0.0, value=350.5, step=10.0)
        
    with col2:
        convenience_stores = st.number_input("Nearby Convenience Stores", min_value=0, max_value=20, value=4, step=1)
        latitude = st.number_input("Latitude Coordinate", min_value=-90.0, max_value=90.0, value=24.98298, format="%.5f")
        longitude = st.number_input("Longitude Coordinate", min_value=-180.0, max_value=180.0, value=121.54024, format="%.5f")
        
    submit_button = st.form_submit_button("Estimate Market Value")

if submit_button:
    # 1. READ ENVIRONMENT VARIABLE FOR PRODUCTION DESYNC
    # Falls back to local development URL if running on your local PC
    base_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
    
    # Strip any accidental trailing slashes to guarantee clean routing
    backend_url = f"{base_url.rstrip('/')}/predict"
    
    payload = {
        "transaction_date": transaction_date,
        "house_age": house_age,
        "distance_to_mrt": distance_to_mrt,
        "convenience_stores": int(convenience_stores),
        "latitude": latitude,
        "longitude": longitude
    }
    
    try:
        with st.spinner("Processing architectural metrics against ML pipeline..."):
            response = requests.post(backend_url, json=payload, timeout=15)
            
        if response.status_code == 200:
            result = response.json()
            prediction_value = round(result["prediction"], 2)
            st.success(f"### Estimated Price: **{prediction_value}** {result['unit']}")
            st.info(f"Features verified by pipeline: {', '.join(result['features_evaluated'])}")
        elif response.status_code == 503:
            st.error("Backend Error: Machine Learning model artifacts are not currently loaded.")
        else:
            st.error(f"Validation Error ({response.status_code}): {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to backend server at {backend_url}. Verify service state execution.")

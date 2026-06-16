import streamlit as st
import requests

st.set_page_config(page_title="Real Estate Valuation", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction Dashboard")
st.write("Adjust the features below to calculate the real estate value in real-time.")

# 1. Layout UI components using columns
col1, col2 = st.columns(2)

with col1:
    distance_to_mrt = st.slider("Distance to nearest MRT (meters)", 0.0, 10000.0, 500.0, step=50.0)
    num_stores = st.number_input("Number of Nearby Convenience Stores", min_value=0, max_value=20, value=3)

with col2:
    latitude = st.number_input("Latitude Coordinate", value=24.97, format="%.4f")
    longitude = st.number_input("Longitude Coordinate", value=121.53, format="%.4f")

# 2. Add Trigger Button
if st.button("Calculate Predicted Market Price", type="primary"):
    # Construct payload MATCHING your FastAPI 'PredictionInput' schema fields exactly
    payload = {
        "transaction_date": 2013.5,  # Added: Provide a fallback/average year if not in UI
        "house_age": 15.0,           # Added: Provide a fallback/average age if not in UI
        "distance_to_mrt": distance_to_mrt,
        "convenience_stores": num_stores, # Fixed key: matches backend model expectation
        "latitude": latitude,
        "longitude": longitude
    }
    
    # URL targeting your FastAPI backend (Toggle comment based on Local vs Production)
    # API_URL = "http://127.0.0"  # For Local testing
    API_URL = "https://onrender.com" # Replace with your live Render URL
    
    try:
        with st.spinner("Analyzing market configurations..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            # Fixed key: matches backend return statement {"prediction": ...}
            price = result["prediction"] 
            unit = result.get("unit", "per unit area")
            st.success(f"### 📈 Estimated Price: ${price:,.2f} {unit}")
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Prediction Server API. Is it running?")

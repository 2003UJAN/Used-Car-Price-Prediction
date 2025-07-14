import streamlit as st
import pandas as pd
import joblib
import io
import requests

st.set_page_config(page_title="Used Car Price Predictor")
st.title("🚗 Used Car Price Predictor")
st.markdown("Estimate resale value using a trained ML model hosted on Hugging Face.")

# URLs from Hugging Face
MODEL_URL = "https://huggingface.co/ujan2003/Car-Price-Prediction/resolve/main/car_price_predictor.pkl"
FEATURES_URL = "https://huggingface.co/ujan2003/Car-Price-Prediction/resolve/main/model_features.pkl"

# Load model and features from Hugging Face
@st.cache_resource
def load_model():
    model_file = requests.get(MODEL_URL).content
    model = joblib.load(io.BytesIO(model_file))
    return model

@st.cache_data
def load_features():
    features_file = requests.get(FEATURES_URL).content
    features = joblib.load(io.BytesIO(features_file))
    return features

model = load_model()
features = load_features()

# Input form
present_price = st.number_input("Present Price (in ₹ Lakh)", 0.0, 100.0, 5.0, step=0.1)
kms_driven = st.number_input("KMs Driven", 0, 500000, 30000, step=500)
mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, 18.0, step=0.1)
engine = st.number_input("Engine Capacity (CC)", 600, 5000, 1200, step=100)
max_power = st.number_input("Max Power (BHP)", 20.0, 300.0, 80.0, step=1.0)
seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9])
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
car_age = st.slider("Car Age (Years)", 0, 25, 5)

fuel_type = st.radio("Fuel Type", ["Diesel", "Petrol", "CNG"])
seller_type = st.radio("Seller Type", ["Dealer", "Individual"])
transmission = st.radio("Transmission", ["Manual", "Automatic"])

# Encoding
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0
seller_individual = 1 if seller_type == "Individual" else 0
trans_manual = 1 if transmission == "Manual" else 0

# Prepare input vector
input_data = {
    'Present_Price': present_price,
    'Kms_Driven': kms_driven,
    'mileage': mileage,
    'engine': engine,
    'max_power': max_power,
    'seats': seats,
    'Owner': owner,
    'Car_Age': car_age,
    'Fuel_Type_Diesel': fuel_diesel,
    'Fuel_Type_Petrol': fuel_petrol,
    'Seller_Type_Individual': seller_individual,
    'Transmission_Manual': trans_manual
}

input_df = pd.DataFrame([input_data])

# Fill in missing dummy features
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]  # Ensure same column order

# Prediction
if st.button("Predict Selling Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Resale Price: ₹ {prediction:.2f} Lakh")

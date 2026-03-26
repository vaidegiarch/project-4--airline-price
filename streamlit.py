import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------
# Load trained model
# ---------------------

with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)
# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Airline Ticket Price Prediction", layout="centered")
st.title("✈️ Airline Ticket Price Prediction")

st.write("""
This app predicts **Airline Ticket Prices** based on your input features.
""")

# ---------------------
# Input fields
# ---------------------
st.header("Input Features")

# Example: replace these with your dataset features
airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet", "GoAir"])
source = st.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai"])
destination = st.selectbox("Destination", ["Cochin", "Hyderabad", "New Delhi"])
total_stops = st.number_input("Total Stops", min_value=0, max_value=5, value=1)
journey_day = st.slider("Journey Day", 1, 31, 15)
journey_month = st.slider("Journey Month", 1, 12, 6)
departure_hour = st.slider("Departure Hour", 0, 23, 10)
arrival_hour = st.slider("Arrival Hour", 0, 23, 12)

# Convert inputs to dataframe for model
input_dict = {
    "Airline": [airline],
    "Source": [source],
    "Destination": [destination],
    "Total_Stops": [total_stops],
    "Journey_Day": [journey_day],
    "Journey_Month": [journey_month],
    "Dep_Hour": [departure_hour],
    "Arrival_Hour": [arrival_hour]
}

input_df = pd.DataFrame(input_dict)

# ---------------------
# Prediction button
# ---------------------
if st.button("Predict Ticket Price"):
    # Preprocess input if needed (encode categorical variables)
    # Example: One-hot encoding placeholder
    # input_df_processed = preprocess(input_df)
    
    # For simplicity, assuming your model handles categorical variables
    prediction = model.predict(input_df)[0]
    
    st.success(f"💰 Predicted Ticket Price: ₹{round(prediction, 2)}")

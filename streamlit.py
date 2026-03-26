# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

@st.cache_resource
def load_model():
    with open("/content/drive/MyDrive/xgb_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
# ---------------------
# Set page config
# ---------------------
st.set_page_config(page_title="Airline Ticket Price Prediction ✈️", layout="centered")
st.title("Airline Ticket Price Prediction")

st.write("""
Predict the price of airline tickets based on your flight details.
""")




# ---------------------
# Input Features
# ---------------------
st.header("Flight Details")

# Airline
airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet", "GoAir", "Vistara"])

# Source
source = st.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai", "Bangalore"])

# Destination
destination = st.selectbox("Destination", ["Cochin", "Hyderabad", "New Delhi", "Kolkata", "Bangalore"])

# Total Stops
total_stops = st.number_input("Total Stops", min_value=0, max_value=5, value=1)

# Journey Date (Day, Month, Year selection)
st.subheader("Journey Date")
day = st.selectbox("Day", list(range(1, 32)))
month = st.selectbox("Month", list(range(1, 13)))
year = st.selectbox("Year", [2023, 2024, 2025])  # adjust years as needed

# Departure and Arrival Hours
departure_hour = st.slider("Departure Hour (0-23)", 0, 23, 10)
arrival_hour = st.slider("Arrival Hour (0-23)", 0, 23, 12)

# ---------------------
# Prepare input DataFrame
# ---------------------
input_dict = {
    "Airline": [airline],
    "Source": [source],
    "Destination": [destination],
    "Total_Stops": [total_stops],
    "Journey_Day": [day],
    "Journey_Month": [month],
    "Journey_Year": [year],
    "Dep_Hour": [departure_hour],
    "Arrival_Hour": [arrival_hour]
}

input_df = pd.DataFrame(input_dict)

# Convert categorical columns to 'category' dtype
for col in ["Airline", "Source", "Destination"]:
    input_df[col] = input_df[col].astype("category")

# ---------------------
# Predict Button
# ---------------------
if st.button("Predict Ticket Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Ticket Price: ₹{round(prediction, 2)}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

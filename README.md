# project-4--airline-price
airline price prediction


Airline Ticket Price Prediction
Project Overview

This project predicts the price of airline tickets based on various flight features such as airline, source, destination, number of stops, journey date, and departure/arrival times. The goal is to provide users with accurate ticket price estimates to help them make informed travel decisions.

The model is trained using XGBoost Regressor, which showed the best performance among several tested algorithms.

Features Used
Airline – Airline operating the flight (e.g., IndiGo, Air India)
Source – City from which the journey starts
Destination – City where the journey ends
Total Stops – Number of stops during the journey
Date of Journey – Split into day and month features
Departure and Arrival Time – Hour of departure and arrival

All categorical features are either converted to category type or one-hot encoded, and date columns are preprocessed into numeric features for model training.

Model
Algorithm: XGBoost Regressor
Evaluation Metric: R² Score on test set
Model File: xgb_model.pkl (saved using pickle)

XGBoost was chosen for its ability to handle non-linear relationships and its high predictive accurac

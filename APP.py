
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("NBmodel.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict whether they will churn or not")

# User input
age = st.number_input("age", min_value=18, max_value=100, value=30)
tenure = st.number_input("tenure (months)", min_value=0, max_value=72, value=12)
gender = st.selectbox("gender", ["Male", "Female"])

# Convert gender to numeric
gender_num = 1 if gender == "Male" else 0

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([[age, tenure, gender_num]], 
                              columns=["age", "tenure", "gender"])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")
    
    st.write(f"Probability of Churn: {probability:.2f}")

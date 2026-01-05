import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('car_purchase_decision_tree.pkl')

st.title("🚗 Car Purchase Prediction App")

st.write("Predict whether a customer will purchase a car based on details")

# User inputs
user_id = st.number_input("User ID", min_value=1)
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=18, max_value=100)
salary = st.number_input("Annual Salary", min_value=1000)

# Convert gender to numeric
gender_encoded = 1 if gender == "Male" else 0

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[user_id, gender_encoded, age, salary]],
        columns=["User ID", "Gender", "Age", "AnnualSalary"]
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Customer is likely to purchase the car")
    else:
        st.error("❌ Customer is not likely to purchase the car")

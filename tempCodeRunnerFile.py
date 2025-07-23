import streamlit as st
import numpy as np
import joblib

# Load the model and encoders
model = joblib.load("salary_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_workclass = joblib.load("le_workclass.pkl")
le_occupation = joblib.load("le_occupation.pkl")

# Set Streamlit page config
st.set_page_config(page_title="Employee Income Prediction", layout="centered")

# Title and description
st.title("💼 Employee Income Prediction App")
st.markdown("This app predicts whether an employee's income is >50K or <=50K based on their profile.")

# Input fields
st.header("📝 Enter Employee Details")

age = st.slider("Age", min_value=17, max_value=75, value=30)

workclass = st.selectbox("Workclass", le_workclass.classes_)
educational_num = st.slider("Education Level (1=Preschool, 16=Doctorate)", min_value=1, max_value=16, value=10)
occupation = st.selectbox("Occupation", le_occupation.classes_)
gender = st.selectbox("Gender", le_gender.classes_)

capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.slider("Hours Worked Per Week", min_value=1, max_value=100, value=40)

# Encode categorical variables
gender_val = le_gender.transform([gender])[0]
workclass_val = le_workclass.transform([workclass])[0]
occupation_val = le_occupation.transform([occupation])[0]

# Prepare features in exact training order
features = np.array([[age, workclass_val, educational_num, occupation_val, gender_val, capital_gain, capital_loss, hours_per_week]])

# Predict on button click
if st.button("🔍 Predict Income"):
    prediction = model.predict(features)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"💰 Predicted Income Category: **{result}**")

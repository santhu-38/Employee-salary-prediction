import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("salary_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_workclass = joblib.load("le_workclass.pkl")
le_occupation = joblib.load("le_occupation.pkl")

# Page config
st.set_page_config(page_title="Employee Income Prediction", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    html, body {
        background: linear-gradient(to right, #eef2f7, #ffffff);
        font-family: 'Segoe UI', sans-serif;
    }

    .header {
        background-color: #1a1f36;
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        margin: 0 auto;
        max-width: 1000px;
    }

    .form-box {
        background-color: #ffffff;
        padding: 2rem 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        max-width: 1000px;
        margin: 0 auto;
    }

    .stRadio > div {
        display: flex;
        gap: 2rem;
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }

    .stButton>button {
        background-color: #0057b8;
        color: white;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.4rem 1.2rem;
        border-radius: 6px;
        transition: 0.3s ease;
        display: block;
        margin: 0 auto;
    }

    .stButton>button:hover {
        background-color: #004494;
    }

    .result-box {
        margin-top: 1.8rem;
        background-color: #e7f4ea;
        padding: 1.3rem;
        border-radius: 12px;
        color: #1b5e20;
        font-weight: bold;
        font-size: 1.6rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in-out;
    }

    .footer {
        margin-top: 2rem;
        text-align: center;
        font-size: 0.9rem;
        color: #666;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>💼 Employee Income Prediction</h1></div>', unsafe_allow_html=True)

# Form Section
st.markdown('<div class="form-box">', unsafe_allow_html=True)
st.markdown("Fill in the employee details below to predict whether the income is **>50K** or **<=50K**.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("📅 Age", min_value=17, max_value=75, value=30)
    educational_num = st.number_input("🎓 Education Level (1=Preschool, 16=Doctorate)", min_value=1, max_value=16, value=10)
    capital_gain = st.number_input("💹 Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("📉 Capital Loss", min_value=0, value=0)

with col2:
    hours_per_week = st.number_input("⏰ Hours Worked per Week", min_value=1, max_value=100, value=40)
    gender = st.radio("👤 Gender", options=["Male", "Female"], index=0, horizontal=True)
    workclass = st.selectbox("🏢 Workclass", le_workclass.classes_)
    occupation = st.selectbox("💼 Occupation", le_occupation.classes_)

# Encoding inputs
gender_val = le_gender.transform([gender])[0]
workclass_val = le_workclass.transform([workclass])[0]
occupation_val = le_occupation.transform([occupation])[0]

features = np.array([[age, workclass_val, educational_num, occupation_val, gender_val, capital_gain, capital_loss, hours_per_week]])

# Predict Button
if st.button("🔍 Predict Income"):
    prediction = model.predict(features)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.markdown(f'<div class="result-box">💰 Predicted Income Category: {result}</div>', unsafe_allow_html=True)

# Close form
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 SmartPredict AI | Crafted with precision and purpose.</div>', unsafe_allow_html=True)

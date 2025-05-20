import streamlit as st
import pandas as pd
import joblib
import os

# Dynamically determine path to the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'sales_model.pkl')

# Load model
model = joblib.load(model_path)

st.title("📈 Sales Prediction App")
st.write("Enter values to predict sales based on advertising spend.")

tv = st.slider("TV Advertisement Spend", 0, 300, 100)
radio = st.slider("Radio Advertisement Spend", 0, 50, 25)
newspaper = st.slider("Newspaper Advertisement Spend", 0, 100, 20)

if st.button("Predict Sales"):
    data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    prediction = model.predict(data)
    st.success(f"💰 Predicted Sales: {round(prediction[0], 2)} units")

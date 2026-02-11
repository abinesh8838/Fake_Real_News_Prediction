import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Fake vs Real Prediction App")

# Example input fields
feature1 = st.number_input("Enter Feature 1")
feature2 = st.number_input("Enter Feature 2")

if st.button("Predict"):
    
    input_data = np.array([[feature1, feature2]])
    input_data = scaler.transform(input_data)
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Real")
    else:
        st.error("Fake")

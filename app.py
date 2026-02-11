import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detection App")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    
    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)
    
    st.subheader("Prediction:")
    st.write(prediction[0])
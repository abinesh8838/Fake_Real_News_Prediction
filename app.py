import streamlit as st
import pickle
import os

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

st.title("📰 Fake News Detection App")

st.write("Enter a news article below to check if it is Fake or Real.")

user_input = st.text_area("News Text")

if st.button("Predict"):
    
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.subheader("Prediction:")
        st.write(prediction[0])

        st.subheader("Confidence:")
        st.write(round(max(probability[0]) * 100, 2), "%")
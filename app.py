import streamlit as st
import joblib

# Load both model and vectorizer from one file
model, vectorizer = joblib.load("model.pkl")

st.title("🧠 Deepfake Text Detector")

# Text input from user
user_input = st.text_area("Enter text to analyze:")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Transform and predict
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        label = "🟢 Real Text" if prediction == 0 else "🔴 Deepfake Text"
        st.success(f"Prediction: {label}")

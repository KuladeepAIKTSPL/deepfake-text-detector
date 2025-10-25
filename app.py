import streamlit as st
import joblib

st.title("🧠 Deepfake Text Detector")

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

user_input = st.text_area("Enter text to analyze:")

if st.button("Detect"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]
        result = "🤖 AI-Generated" if pred == 1 else "🧍 Human-Written"
        st.subheader(result)
    else:
        st.warning("Please enter some text.")

import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords

# Set page config before any Streamlit content
st.set_page_config(page_title="🧠 Reddit Sentiment Analyzer", page_icon="💬", layout="centered")

# Download stopwords (if not already)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load pre-trained components
try:
    tfidf = joblib.load("reddit_sentiment_app/model/tfidf_vectorizer.pkl")
    le = joblib.load("reddit_sentiment_app/model/label_encoder.pkl")
    model = tf.keras.models.load_model("reddit_sentiment_app/model/sentiment_model.h5")
except Exception as e:
    st.error(f"❌ Failed to load model components. Check file paths.\n\n**Error:** {e}")
    st.stop()

# Title and description
st.title("🧠 Reddit Sentiment Analysis")
st.markdown(
    """
    Welcome to the **Reddit Sentiment Analyzer**!  
    This tool uses a trained deep learning model to predict the **emotion** behind Reddit comments.  
    Simply enter a comment below and get instant insights! 🎯
    """
)

# Clean the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# User input section
user_input = st.text_area("💬 Enter a Reddit comment:")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Preprocessing
        clean_input = clean_text(user_input)
        tfidf_input = tfidf.transform([clean_input]).toarray()
        prediction = model.predict(tfidf_input)

        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = le.inverse_transform([predicted_index])[0]
        confidence = float(np.max(prediction))

        # Display prediction
        st.success(f"🎯 **Predicted Emotion:** `{predicted_label}`")
        st.info(f"📊 **Confidence Score:** `{confidence * 100:.2f}%`")

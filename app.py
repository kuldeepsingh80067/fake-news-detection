# ==========================================
# 📰 Fake News Detection PRO App
# Author: Kuldeep Singh
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
import requests
from bs4 import BeautifulSoup
from PIL import Image
import easyocr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 🧹 TEXT CLEANING
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# ==========================================
# 🌐 EXTRACT TEXT FROM URL
# ==========================================
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.text for p in soup.find_all("p")]
        return " ".join(paragraphs)
    except:
        return ""


# ==========================================
# 🖼️ OCR FROM IMAGE
# ==========================================
reader = easyocr.Reader(['en'])

def extract_text_from_image(image):
    result = reader.readtext(np.array(image))
    text = " ".join([res[1] for res in result])
    return text


# ==========================================
# 🤖 LOAD / TRAIN MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    except:
        data = {
            "text": [
                "Government releases official economic report",
                "Scientists confirm climate change effects",
                "Breaking news: major policy announced",
                "Click here to earn money instantly!!!",
                "Miracle cure discovered doctors hate it",
                "Shocking secret revealed you won't believe",
                "NASA launches new satellite successfully",
                "Fake celebrity news goes viral online"
            ],
            "label": [1,1,1,0,0,0,1,0]
        }

        df = pd.DataFrame(data)
        df["text"] = df["text"].apply(clean_text)

        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]

        model = LogisticRegression()
        model.fit(X, y)

        pickle.dump(model, open("model.pkl", "wb"))
        pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return model, vectorizer


model, vectorizer = load_model()

# ==========================================
# 🎨 UI
# ==========================================
st.set_page_config(page_title="Fake News Detector PRO", page_icon="📰")

# Sidebar (🔥 Professional Touch)
st.sidebar.title("👨‍💻 About")
st.sidebar.markdown("""
**Developer:** Kuldeep Singh  
AI/ML Enthusiast 🚀  
""")

st.title("📰 Fake News Detection PRO")
st.markdown("Analyze news via **Text, URL, or Image** 🤖")

option = st.radio("Choose Input Type:", ["Text", "URL", "Image"])

news_text = ""

# TEXT INPUT
if option == "Text":
    news_text = st.text_area("Enter News Text")

# URL INPUT
elif option == "URL":
    url = st.text_input("Enter News URL")
    if url:
        news_text = get_text_from_url(url)
        st.success("✅ Text extracted from URL")

# IMAGE INPUT
elif option == "Image":
    image_file = st.file_uploader("Upload News Image", type=["jpg", "png", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        news_text = extract_text_from_image(image)
        st.success("✅ Text extracted from Image")

# ==========================================
# 🔍 PREDICTION
# ==========================================
if st.button("Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please provide input")
    else:
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()

        st.subheader("📊 Result")

        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.write(f"Confidence: {confidence*100:.2f}%")

        if "click" in news_text.lower() or "shocking" in news_text.lower():
            st.info("⚠️ This news contains sensational words, often used in fake news.")

# Footer (🔥 Branding)
st.markdown("---")
st.markdown("### 👨‍💻 Developed by Kuldeep Singh")
st.caption("Built with ❤️ using AI, NLP & Streamlit")

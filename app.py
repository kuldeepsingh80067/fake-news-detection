# ==========================================
# 📰 Fake News Detection PRO (10/10 Version)
# Author: Kuldeep Singh
# ==========================================

import streamlit as st
import numpy as np
import re
import string
import pickle
import requests
from bs4 import BeautifulSoup
from PIL import Image

# OPTIONAL OCR (safe import)
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
except:
    reader = None

# ==========================================
# ⚙️ PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Fake News Detector PRO", page_icon="📰")

# ==========================================
# 🧹 TEXT CLEANING
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ==========================================
# 🌐 URL TEXT EXTRACTION
# ==========================================
def get_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paragraphs)
    except:
        return ""

# ==========================================
# 🖼️ IMAGE OCR
# ==========================================
def extract_text_from_image(image):
    if reader is None:
        return ""
    try:
        result = reader.readtext(np.array(image))
        return " ".join([r[1] for r in result])
    except:
        return ""

# ==========================================
# 🤖 LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except:
        st.error("❌ Model files not found! Upload model.pkl & vectorizer.pkl")
        st.stop()

model, vectorizer = load_model()

# ==========================================
# 🎨 UI
# ==========================================
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
        with st.spinner("Extracting content..."):
            news_text = get_text_from_url(url)
        if news_text:
            st.success("✅ Content extracted")
        else:
            st.error("❌ Failed to extract content")

# IMAGE INPUT
elif option == "Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Reading text from image..."):
            news_text = extract_text_from_image(image)
        if news_text:
            st.success("✅ Text extracted from image")
        else:
            st.warning("⚠️ Could not detect text")

# ==========================================
# 🔍 PREDICTION
# ==========================================
if st.button("🔍 Analyze News"):
    if not news_text.strip():
        st.warning("⚠️ Please provide valid input")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(news_text)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            confidence = np.max(proba)

        st.subheader("📊 Result")

        if prediction == 1:
            st.success(f"✅ Real News ({confidence*100:.2f}% confidence)")
        else:
            st.error(f"❌ Fake News ({confidence*100:.2f}% confidence)")

        # Insight
        if confidence < 0.6:
            st.warning("⚠️ Low confidence prediction — result may not be reliable")

# ==========================================
# 📌 FOOTER
# ==========================================
st.markdown("---")
st.markdown("### 👨‍💻 Developed by Kuldeep Singh")
st.caption("Built with ❤️ using AI, NLP & Streamlit")

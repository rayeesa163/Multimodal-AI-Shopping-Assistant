import streamlit as st
import pandas as pd
import cv2
from deepface import DeepFace
from sentence_transformers import SentenceTransformer, util
import torch

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Shopping Concierge",
    page_icon="🛍️",
    layout="wide"
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_products():
    return pd.read_csv("products.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

products = load_products()
model = load_model()

# -------------------- HEADER --------------------
st.markdown("""
    <h1 style="font-size: 48px; color: #4A90E2;">🛍️ Multimodal AI Shopping Assistant</h1>
    <h4 style="color: #6c757d;">Smart, emotional, and intent-aware shopping recommendations powered by AI 🤖</h4>
    <hr>
""", unsafe_allow_html=True)

# -------------------- EMOTION DETECTION --------------------
st.sidebar.header("🎭 Emotion Detection")

if st.sidebar.button("📸 Capture from Webcam"):
    st.sidebar.info("Opening webcam... please wait.")

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            st.sidebar.success(f"😊 Detected: **{emotion.upper()}**")
        except Exception:
            emotion = "neutral"
            st.sidebar.error("😓 Couldn’t detect face properly.")
    else:
        emotion = "neutral"
        st.sidebar.error("⚠️ Webcam access failed.")
else:
    emotion = "neutral"
    st.sidebar.markdown("Default Emotion: 😐 Neutral")

# -------------------- MAIN INPUT --------------------
st.markdown("## 📝 Tell us what you're looking for:")
user_query = st.text_input("Example: Stylish shoes under ₹3000", value="Comfy hoodie under ₹2000")

# -------------------- PRODUCT RECOMMENDATION --------------------
if st.button("✨ Show Smart Recommendations"):
    st.markdown("### 🎯 Matching Products Just for You:")

    filtered = products[products['emotion_tag'] == emotion]

    if filtered.empty:
        st.warning("No emotion-specific products found! Showing top recommendations instead.")
        filtered = products

    product_embeddings = model.encode(filtered['product_name'].tolist(), convert_to_tensor=True)
    user_embedding = model.encode(user_query, convert_to_tensor=True)

    scores = util.cos_sim(user_embedding, product_embeddings)[0]
    top_indices = torch.topk(scores, k=min(3, len(scores))).indices

    for idx in top_indices:
        product = filtered.iloc[int(idx)]
        st.markdown(f"""
        <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <h4>🛍️ {product['product_name']}</h4>
            <ul>
                <li><strong>Category:</strong> {product['category']}</li>
                <li><strong>Price:</strong> ₹{product['price']}</li>
                <li><strong>Emotion Tag:</strong> {product['emotion_tag'].capitalize()}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ by <strong>Rayeesa Tabusum</strong> & <strong>Mahera Mahin</strong> · AI Engineers · Converge 2025</p>",
    unsafe_allow_html=True
)

import streamlit as st
from PIL import Image
from model import load_model, predict_with_heatmap, LABELS
from utils import preprocess_image, overlay_heatmap
import torch

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Intelligent X-Ray Analysis System")
st.markdown("### Early detection of diseases using AI")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Project Info")
    st.info("""
    CS50 Project:
    - X-ray image analysis
    - Detection of respiratory diseases
    - Highlight areas of interest
    - Confidence score
    """)
    st.warning("⚠️ This is an experimental system and does not replace a doctor's diagnosis")

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_data(show_spinner=True)
def get_model():
    model = load_model()
    return model

model = get_model()

if model is None:
    st.error("❌ Failed to load the model. Please check the files")
    st.stop()

# ---------------------------
# Upload image
# ---------------------------
st.header("📤 Upload X-Ray Image")
uploaded_file = st.file_uploader(
    "Select an X-ray image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = preprocess_image(uploaded_file)
    if image is None:
        st.error("❌ Failed to process the image. Please check the file")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.subheader("📊 Analysis Results")
        label, confidence, heatmap = predict_with_heatmap(model, image)

        # Show diagnosis with appropriate color
        if confidence > 0.7:
            st.success(f"Diagnosis: {label}")
        elif confidence > 0.5:
            st.warning(f"Diagnosis: {label}")
        else:
            st.error(f"Diagnosis: {label}")

        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.progress(float(confidence))

        if label == "Normal":
            st.info("✅ The image appears normal")
        else:
            st.warning("⚠️ It is recommended to consult a doctor")

    # ---------------------------
    # Show heatmap
    # ---------------------------
    st.subheader("🗺️ Important Areas Map")
    overlayed = overlay_heatmap(image, heatmap)
    st.image(overlayed, caption="Areas with Highest Probability", use_column_width=True)
    st.caption("Red color indicates the most important regions")

else:
    st.info("👆 Please upload an image to start analysis")
    st.image("https://via.placeholder.com/500x300?text=Upload+X-Ray+Image",
             use_column_width=True)

st.markdown("---")
st.caption("CS50 Project - AI-based Medical Image Analysis App")

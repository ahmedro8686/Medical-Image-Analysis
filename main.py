import streamlit as st
from PIL import Image
from model import load_model, predict_with_heatmap, LABELS
from utils import preprocess_image, overlay_heatmap
import torch

# ---------------------------
# إعداد الصفحة
# ---------------------------
st.set_page_config(
    page_title="تحليل الصور الطبية",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 نظام التحليل الذكي للصور الشعاعية")
st.markdown("### الكشف المبكر عن الأمراض باستخدام الذكاء الاصطناعي")

# ---------------------------
# الشريط الجانبي
# ---------------------------
with st.sidebar:
    st.header("معلومات المشروع")
    st.info("""
    مشروع CS50:
    - تحليل صور الأشعة السينية
    - الكشف عن الأمراض التنفسية
    - عرض مناطق الاهتمام
    - نسبة ثقة في النتائج
    """)
    st.warning("⚠️ هذا نظام تجريبي ولا يغني عن تشخيص الطبيب")

# ---------------------------
# تحميل النموذج (مخزن في cache)
# ---------------------------
@st.cache_data(show_spinner=True)
def get_model():
    model = load_model()
    return model

model = get_model()

if model is None:
    st.error("❌ تعذر تحميل النموذج. يرجى التحقق من الملفات")
    st.stop()

# ---------------------------
# رفع الصورة
# ---------------------------
st.header("📤 رفع الصورة الشعاعية")
uploaded_file = st.file_uploader(
    "اختر صورة أشعة سينية (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = preprocess_image(uploaded_file)
    if image is None:
        st.error("❌ تعذر معالجة الصورة. يرجى التحقق من الملف")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 الصورة الأصلية")
        st.image(image, caption="الصورة الأصلية", use_column_width=True)

    with col2:
        st.subheader("📊 نتائج التحليل")
        label, confidence, heatmap = predict_with_heatmap(model, image)

        # عرض التشخيص مع اللون المناسب
        if confidence > 0.7:
            st.success(f"التشخيص: {label}")
        elif confidence > 0.5:
            st.warning(f"التشخيص: {label}")
        else:
            st.error(f"التشخيص: {label}")

        st.metric("نسبة الثقة", f"{confidence*100:.2f}%")
        st.progress(float(confidence))

        if label == "طبيعي":
            st.info("✅ الصورة تبدو طبيعية")
        else:
            st.warning("⚠️ يُنصح بمراجعة الطبيب")

    # ---------------------------
    # عرض heatmap
    # ---------------------------
    st.subheader("🗺️ خريطة المناطق المهمة")
    overlayed = overlay_heatmap(image, heatmap)
    st.image(overlayed, caption="المناطق ذات الاحتمالية الأعلى", use_column_width=True)
    st.caption("اللون الأحمر يشير للمناطق الأكثر أهمية")

else:
    st.info("👆 يرجى رفع صورة لبدء التحليل")
    st.image("https://via.placeholder.com/500x300?text=Upload+X-Ray+Image",
             use_column_width=True)

st.markdown("---")
st.caption("مشروع CS50 - تطبيق تحليل الصور الطبية بالذكاء الاصطناعي")

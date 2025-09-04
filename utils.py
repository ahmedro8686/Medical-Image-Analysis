

from PIL import Image
import numpy as np
import cv2

def preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        if image.size[0] < 50 or image.size[1] < 50:
            raise ValueError("الصورة صغيرة جداً")
        return image
    except Exception as e:
        print(f"خطأ في معالجة الصورة: {e}")
        return None

def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha=0.5):
    try:
        image_np = np.array(image.resize((224,224)))
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max()-heatmap.min()+1e-8)
        heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_normalized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_np, 1-alpha, heatmap_colored, alpha, 0)
        return Image.fromarray(overlay)
    except Exception as e:
        print(f"خطأ في دمج heatmap: {e}")
        return image


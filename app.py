# ----------------- FAKE CV2 FIX FOR STREAMLIT CLOUD -----------------
import sys, types
fake_cv2 = types.ModuleType("cv2")
fake_cv2.imread = lambda *a, **k: None
fake_cv2.imwrite = lambda *a, **k: None
fake_cv2.imshow = lambda *a, **k: None
fake_cv2.destroyAllWindows = lambda *a, **k: None
fake_cv2.IMREAD_COLOR = 1
sys.modules["cv2"] = fake_cv2
# --------------------------------------------------------------------

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="Pothole Detector", layout="centered")
st.title("🚧 Pothole Detection System")
st.write("Upload a road image and the model will detect potholes and draw bounding boxes.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        img_path = tmp.name

    with st.spinner("🔍 Detecting potholes... Please wait..."):
        results = model(img_path)
        output = results[0].plot()
        output = output[:, :, ::-1]  # BGR → RGB

        st.image(output, caption="✅ Detection Result", use_column_width=True)
        st.success("Detection Completed ✅")

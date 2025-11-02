import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception as e:
    st.error(f"YOLO import failed: {e}")
    st.stop()

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("❌ best.pt file not found. Upload your model file to project root.")
        st.stop()
    return YOLO(model_path)

model = load_model()

# --------------------------
# UI
# --------------------------
st.title("🚧 Pothole Detection (YOLOv8)")
st.write("Upload an image to detect potholes.")

file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(img)

    st.write("⚙️ Running model...")

    results = model(img_np, device="cpu")

    result_img = results[0].plot()
    st.image(result_img, caption="Result", use_column_width=True)

    st.subheader("Detections:")
    names = results[0].names

    for box in results[0].boxes:
        st.write(f"✔️ {names[int(box.cls)]} ({float(box.conf):.2f})")

st.caption("✅ YOLOv8 + Streamlit Cloud | CPU Mode | Latest Versions")

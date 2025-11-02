import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Prevent Qt errors in Streamlit

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Try OpenCV import safely
try:
    import cv2
except:
    cv2 = None
    st.warning("⚠️ OpenCV GUI backend disabled. Running headless mode.")

# --------------------------
# Load YOLO Model
# --------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")   # replace with your model file name
    return model

model = load_model()

# --------------------------
# Streamlit UI
# --------------------------
st.title("🚦 Object Detection App (YOLO + Streamlit)")
st.write("Upload an image to detect objects")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array
    img_np = np.array(img)

    st.write("🔍 Detecting...")
    
    # Run YOLO (Force CPU for Streamlit)
    results = model(img_np, device="cpu")

    # Plot results
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Output", use_column_width=True)

    # Show detected classes
    st.subheader("🧾 Detected Objects:")
    names = results[0].names

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        st.write(f"✔️ {names[cls]} — {conf:.2f}")

st.markdown("---")
st.caption("✅ Powered by YOLOv8 + Streamlit (Latest Versions)")

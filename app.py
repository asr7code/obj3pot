import streamlit as st

# Try importing cv2 safely
try:
    import cv2
except ImportError:
    st.write("Installing OpenCV...")
    import os
    os.system("pip install opencv-python-headless>=4.10.0")
    import cv2

# ✅ Patch OpenCV GUI functions for headless mode
if not hasattr(cv2, "IMREAD_COLOR"):
    cv2.IMREAD_COLOR = 1
if not hasattr(cv2, "IMREAD_GRAYSCALE"):
    cv2.IMREAD_GRAYSCALE = 0
if not hasattr(cv2, "IMWRITE_JPEG_QUALITY"):
    cv2.IMWRITE_JPEG_QUALITY = 1
if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *args, **kwargs: None
if not hasattr(cv2, "waitKey"):
    cv2.waitKey = lambda *args, **kwargs: None

from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.title("Pothole Detection App 🚧")
st.write("Upload an image to detect potholes")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting potholes..."):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp.name)

        results = model(temp.name)
        output_img = results[0].plot()

        st.image(output_img, caption="Detected Potholes ✅", use_column_width=True)
        st.success("Done!")

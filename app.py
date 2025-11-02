import streamlit as st
import numpy as np

# ✅ Patch OpenCV for headless environment (before importing YOLO)
import cv2
if not hasattr(cv2, "IMREAD_COLOR"):
    cv2.IMREAD_COLOR = 1
if not hasattr(cv2, "IMREAD_GRAYSCALE"):
    cv2.IMREAD_GRAYSCALE = 0
if not hasattr(cv2, "IMWRITE_JPEG_QUALITY"):
    cv2.IMWRITE_JPEG_QUALITY = 1
if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *args, **kwargs: None  # disable GUI functions
if not hasattr(cv2, "waitKey"):
    cv2.waitKey = lambda *args, **kwargs: None

# ✅ After patching, import YOLO
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Pothole Detection App 🚧")
st.write("Upload an image to detect potholes")

# Load YOLO model
model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting potholes..."):
        # Save input to temp file for YOLO
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file.name)

        results = model(temp_file.name)

        # Render result image
        result_img = results[0].plot()  # numpy array
        st.image(result_img, caption="Detected Potholes ✅", use_column_width=True)

        st.success("Detection Complete!")

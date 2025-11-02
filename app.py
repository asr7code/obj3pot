import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.title("🚧 Pothole Detection App")
st.write("Upload an image to detect potholes using YOLOv8")

# Load model
model = YOLO("best.pt")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        temp_img_path = tmp.name

    # Predict
    with st.spinner("Detecting Potholes..."):
        results = model(temp_img_path)
        output = results[0].plot()  # returns numpy array (BGR)

        # Convert BGR to RGB for Streamlit
        output_rgb = output[:, :, ::-1]

        st.image(output_rgb, caption="Detected Potholes ✅", use_column_width=True)
        st.success("Detection complete ✅")

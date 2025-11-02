import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# App config
st.set_page_config(page_title="Pothole Detection", layout="centered")
st.title("🕳️ Pothole Detection using YOLOv8")

# Model loader
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # keep file name same

model = load_model()

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        img.save(temp.name)
        temp_path = temp.name

    # run YOLO detection
    results = model(temp_path)

    # save YOLO result to file
    result_img_path = "result.jpg"
    results[0].plot(save=True, filename=result_img_path)

    # display result
    st.subheader("Detection Result:")
    st.image(result_img_path, use_column_width=True)

    # cleanup
    os.remove(temp_path)

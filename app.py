import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="Pothole Detection App", layout="wide")
st.title("🕳️ Pothole Detection System")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    with st.spinner("Detecting potholes..."):
        results = model.predict(img_array)

    result_img = results[0].plot()  # draw bounding boxes

    st.subheader("✅ Detection Result:")
    st.image(result_img, caption="Pothole Detection", use_column_width=True)

    # download output button
    result = Image.fromarray(result_img)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        result.save(tmp.name)
        st.download_button("⬇️ Download Output Image", data=open(tmp.name, "rb"), file_name="detected_potholes.jpg")

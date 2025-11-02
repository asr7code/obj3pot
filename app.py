import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import tempfile
import os

# Load model only once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # your trained model
    return model

model = load_model()

st.title("🛣️ Pothole Detection System")
st.write("Upload a road image, and the model will detect potholes and draw bounding boxes.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=500)

    # Save temp image for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        img.save(temp.name)
        temp_path = temp.name

    # Run YOLO inference
    results = model.predict(temp_path, conf=0.4)

    # Draw bounding boxes
    img_cv = cv2.imread(temp_path)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            conf = float(box.conf[0])

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_cv, f"Pothole {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Convert back to display
    img_out = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    st.image(img_out, caption="Detected Potholes ✅", width=500)

    # Download option
    result = Image.fromarray(img_out)
    result.save("result.jpg")

    with open("result.jpg", "rb") as f:
        st.download_button("📥 Download Result Image", f, "pothole_detected.jpg")

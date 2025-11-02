import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ✅ Safe OpenCV import
try:
    import cv2
except:
    cv2 = None

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ✅ Load YOLO model
model = YOLO("best.pt")

# ✅ Safe drawing (works without cv2 GUI)
def draw_boxes(result, img):
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    img = np.array(img)
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)

        img[y1:y1+3, x1:x2] = [255, 0, 0]
        img[y2:y2+3, x1:x2] = [255, 0, 0]
        img[y1:y2, x1:x1+3] = [255, 0, 0]
        img[y1:y2, x2:x2+3] = [255, 0, 0]

    return Image.fromarray(img)

# ✅ Streamlit UI
st.title("🚧 Pothole Detector — YOLOv8")
uploaded_file = st.file_uploader("Upload a road image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Detecting potholes...")
    result = model.predict(img)[0]
    
    output = draw_boxes(result, img)
    st.image(output, caption="✅ Detected Potholes", use_container_width=True)

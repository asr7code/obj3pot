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

# ✅ Load model
model = YOLO("best.pt")


# ✅ Custom safe plot function (bypass cv2 issues)
def draw_boxes(result, img):
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    img = np.array(img)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"pothole {confs[i]:.2f}"

        # draw rectangle manually (no cv2 needed)
        img[y1:y1+3, x1:x2] = [255, 0, 0]     # top
        img[y2:y2+3, x1:x2] = [255, 0, 0]     # bottom
        img[y1:y2, x1:x1+3] = [255, 0, 0]     # left
        img[y1:y2, x2:x2+3] = [255, 0, 0]     # right

    return Image.fromarray(img)


# ✅ Streamlit UI
st.title("🚧 Pothole Detection App")
st.write("Upload an image to detect potholes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Detecting potholes...")
    results = model.predict(img)

    result_img = draw_boxes(results[0], img)
    st.image(result_img, caption="✅ Potholes Detected", use_container_width=True)

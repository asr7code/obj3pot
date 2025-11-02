# ----- ✅ Fix OpenCV import crash in Streamlit Cloud -----
import sys, types

fake_cv2 = types.ModuleType("cv2")

# Add dummy functions/attributes that Ultralytics needs
fake_cv2.imshow = lambda *a, **k: None
fake_cv2.waitKey = lambda *a, **k: None
fake_cv2.destroyAllWindows = lambda *a, **k: None
fake_cv2.imread = lambda *a, **k: None
fake_cv2.imwrite = lambda *a, **k: None

sys.modules["cv2"] = fake_cv2
# --------------------------------------------------------

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw

# ✅ Load YOLO model
model = YOLO("best.pt")

def draw_boxes(result, img):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, y1), f"{conf:.2f}", fill="red")

    return img

st.title("🚧 Pothole Detection YOLOv8")
uploaded_file = st.file_uploader("Upload road image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    st.write("Detecting potholes...")
    result = model.predict(img, conf=0.25)[0]

    output = draw_boxes(result, img)
    st.image(output, caption="Detected Potholes", use_container_width=True)

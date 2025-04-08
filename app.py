import streamlit as st
from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# 🎨 Streamlit UI Setup
st.set_page_config(page_title="Safety Violation Detector", layout="wide")
st.title("🚧 Real-Time Safety Violation Detection with YOLOv11")

# 🧠 Load your trained YOLOv11 model
model = YOLO("yolov11_baseline_model.pt")  # 👈 Replace this with your model file name

# 🎥 Use your computer’s built-in webcam
cap = cv2.VideoCapture(0)  # 0 means default webcam

# 🧾 Display Start Checkbox
run = st.checkbox("▶️ Start Camera")

# 💡 Image Placeholder & Violation Counter
FRAME_WINDOW = st.image([])
violation_count = 0

# 🗂️ Create folder to save violation snapshots
if not os.path.exists("violations"):
    os.makedirs("violations")

# 🔁 Real-time detection loop
while run:
    success, frame = cap.read()
    if not success:
        st.warning("⚠️ Could not access webcam.")
        break

    # 🧠 Run detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # 🧾 Extract detected class names
    class_ids = results[0].boxes.cls.cpu().tolist()
    class_names = [model.names[int(cls)] for cls in class_ids]

    # 🚨 Define which labels are considered violations
    violation_classes = ["no_helmet", "no_vest"]  # 👈 Edit to match your model labels
    violations = [name for name in class_names if name in violation_classes]

    if violations:
        violation_count += 1
        st.error(f"🚨 Violation Detected: {', '.join(set(violations))}")
        
        # 💾 Save snapshot of violation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violations/violation_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

    # 📊 Show violation counter
    st.metric("Violations Detected", violation_count)

    # 🖼️ Convert BGR to RGB and display the annotated frame
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(rgb_frame)

# 🧹 Release camera when done
cap.release()

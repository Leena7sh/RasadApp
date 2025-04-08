import streamlit as st
from ultralytics import YOLO
import cv2

# Streamlit page config
st.set_page_config(page_title="Live Safety Violation Detection", layout="wide")
st.title("🚧 Real-Time Safety Violation Detection with YOLOv11")

# Load YOLOv11 model
model_path = "/workspaces/RasadApp/app.py"  # Change this if your file has a different name
model = YOLO(model_path)

# Streamlit checkbox to start/stop detection
run = st.checkbox('▶️ Start Camera')

# Image placeholder
FRAME_WINDOW = st.image([])

# Open webcam
cap = cv2.VideoCapture(0)

while run:
    success, frame = cap.read()
    if not success:
        st.error("❌ Could not access webcam.")
        break

    # Run inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # Convert BGR (OpenCV) to RGB (Streamlit)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display in Streamlit
    FRAME_WINDOW.image(annotated_frame)

# Release webcam when done
cap.release()


import streamlit as st
from ultralytics import YOLO
import cv2
import os
from datetime import datetime, timedelta
import torch
import pandas as pd
import altair as alt
from playsound import playsound
import threading

# ✅ FIRST Streamlit call must be here
st.set_page_config(page_title="Safety Violation Detector", layout="wide")

# 🧠 System Info
st.sidebar.subheader("🧠 System Check")
st.sidebar.write("Using GPU?", torch.cuda.is_available())
st.sidebar.write("Device name:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 🎨 UI Setup
st.set_page_config(page_title="Safety Violation Detector", layout="wide")
st.title("🚧 Real-Time Safety Violation Detection with YOLOv11")

# 🧬 Load model
model = YOLO("yolov11_baseline_model.torchscript")
model.to("cpu")

# 📁 Violation folder setup
if not os.path.exists("violations"):
    os.makedirs("violations")

# 📊 Session state setup
if "violation_log" not in st.session_state:
    st.session_state.violation_log = []

# 🎮 Start/Stop
run = st.sidebar.checkbox("▶️ Start Camera")
reset = st.sidebar.button("🔄 Reset Log")

if reset:
    st.session_state.violation_log.clear()

# 📹 Setup webcam
cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])

# 🔊 Play sound on violation (runs in background thread)
def play_alert():
    try:
        playsound("alert.mp3")  # Make sure this file exists
    except:
        pass

# Define violation classes
VIOLATION_CLASSES = ["maskoff", "no_glove", "no_hairnet"]

# Time limit for counting violations
VIOLATION_WINDOW = timedelta(seconds=30)

# ♻️ Detection Loop
if run:
    while True:
        success, frame = cap.read()
        if not success:
            st.warning("⚠️ Could not access webcam.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # Extract detections
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().tolist() if boxes is not None else []
        class_names = [model.names[int(cls)] for cls in class_ids] if class_ids else []

        # Filter for violations only
        violations = [name for name in class_names if name in VIOLATION_CLASSES]

        if violations:
            timestamp = datetime.now()
            st.error(f"🚨 Violation Detected: {', '.join(set(violations))}")

            filename = f"violations/violation_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)

            # Log violation
            st.session_state.violation_log.append({
                "Time": timestamp,
                "Type": ', '.join(set(violations))
            })

            # Play alert sound in background
            threading.Thread(target=play_alert).start()

        # Filter violations within last 30 seconds
        now = datetime.now()
        recent_violations = [v for v in st.session_state.violation_log if v["Time"] >= now - VIOLATION_WINDOW]

        # Show metrics
        col1, col2 = st.columns(2)
        col1.metric("📷 Total Detections", len(class_names))
        col2.metric("❌ Violations (30s)", len(recent_violations))

        # Show video
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame, use_column_width=True)

    cap.release()

# 📊 Violation Log + Chart
if st.session_state.violation_log:
    st.subheader("📊 Violation History")
    df_log = pd.DataFrame(st.session_state.violation_log)
    df_log["Time"] = pd.to_datetime(df_log["Time"])
    st.dataframe(df_log.sort_values("Time", ascending=False), use_container_width=True)

    st.subheader("🔺 Violation Trend (Bar Chart)")
    df_chart = df_log.copy()
    df_chart["Minute"] = df_chart["Time"].dt.floor("min")
    chart_data = df_chart.groupby(["Minute", "Type"]).size().reset_index(name="Count")

    chart = alt.Chart(chart_data).mark_bar().encode(
        x="Minute:T",
        y="Count:Q",
        color="Type:N",
        tooltip=["Minute", "Type", "Count"]
    ).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

    st.subheader("🔢 Violation Type Breakdown")
    pie_data = df_log["Type"].value_counts().reset_index()
    pie_data.columns = ["Type", "Count"]

    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta="Count",
        color="Type",
        tooltip=["Type", "Count"]
    )
    st.altair_chart(pie_chart, use_container_width=True)

else:
    st.info("No violations recorded yet.")

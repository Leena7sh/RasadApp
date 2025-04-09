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
# 🎨 UI Setup
st.set_page_config(page_title="Safety Violation Detector", layout="wide")
st.title("🚧 Real-Time Safety Violation Detection with YOLOv11")

# 🧠 System Info
st.sidebar.subheader("🧠 System Check")
st.sidebar.write("Using GPU?", torch.cuda.is_available())
st.sidebar.write("Device name:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# 🧬 Load model
model = YOLO("yolov11_baseline_model.torchscript")
model.to("cpu")

# 📁 Violation folder setup
if not os.path.exists("violations"):
    os.makedirs("violations")

# 📊 Session state setup
if "violation_log" not in st.session_state:
    st.session_state.violation_log = []
if "last_logged_time" not in st.session_state:
    st.session_state.last_logged_time = datetime.min

# 🎮 Start/Stop
run = st.sidebar.checkbox("▶️ Start Camera")
reset = st.sidebar.button("🔄 Reset Log")

if reset:
    st.session_state.violation_log.clear()
    st.session_state.last_logged_time = datetime.min

# 📹 Setup webcam
cap = cv2.VideoCapture(0)
col1, col2 = st.columns(2)
FRAME_WINDOW = col1.image([])

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

# Time limit between consecutive logs
LOG_INTERVAL = timedelta(seconds=5)

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

        now = datetime.now()
        if violations and now - st.session_state.last_logged_time > LOG_INTERVAL:
            st.session_state.last_logged_time = now
            st.error(f"🚨 Violation Detected: {', '.join(set(violations))}")

            filename = f"violations/violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)

            # Log violation
            st.session_state.violation_log.append({
                "Time": now,
                "Type": ', '.join(set(violations))
            })

            # Play alert sound in background
            threading.Thread(target=play_alert).start()

        # Filter violations within last 30 seconds
        recent_violations = [v for v in st.session_state.violation_log if v["Time"] >= now - VIOLATION_WINDOW]

        # Show metrics
        col2.metric("📷 Total Detections", len(class_names))
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

    # 📂 Daily Report Download
    today = datetime.now().strftime("%Y-%m-%d")
    csv_data = df_log.to_csv(index=False).encode('utf-8')
    st.download_button(label="📂 Download CSV Report", data=csv_data, file_name=f"violations_{today}.csv", mime='text/csv')

    # 📊 Charts
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

    # ♻️ Delete logs older than 1 day
    st.session_state.violation_log = [v for v in st.session_state.violation_log if v["Time"] >= datetime.now() - timedelta(days=1)]

else:
    st.info("No violations recorded yet.")

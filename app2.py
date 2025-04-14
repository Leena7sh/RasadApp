import streamlit as st
from ultralytics import YOLO
import cv2
import os
from datetime import datetime, timedelta
import torch
import pandas as pd
import altair as alt
import threading
import time
from playsound import playsound

# 🎨 Streamlit Setup
st.set_page_config(page_title="Rasad - Violation Dashboard", layout="wide")
st.title("🛡️ Rasad - Safety Violation Dashboard")

# 🕒 Show live clock
st.markdown(f"### 🕒 Current Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

# 🧠 System Info
st.sidebar.subheader("🧠 System Info")
st.sidebar.write("Using GPU?", torch.cuda.is_available())
st.sidebar.write("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 🧬 Load YOLOv11 Model
model = YOLO("yolov11_baseline_model.torchscript", task="detect")

# 📁 Ensure folders exist
os.makedirs("violations", exist_ok=True)

# 📊 Session State Initialization
if "violation_log" not in st.session_state:
    st.session_state.violation_log = []
if "last_logged_time" not in st.session_state:
    st.session_state.last_logged_time = datetime.min

# 🎮 Sidebar Controls
run = st.sidebar.checkbox("▶️ Start Camera")
reset = st.sidebar.button("🔄 Reset Log")

if reset:
    st.session_state.violation_log.clear()
    st.session_state.last_logged_time = datetime.min

# 📹 Webcam Setup
cap = cv2.VideoCapture(0)
col1, col2 = st.columns(2)
FRAME_WINDOW = col1.image([])

# 🔊 Sound Alert Function
def play_alert():
    try:
        playsound("alert.mp3")
    except Exception as e:
        print("Sound error:", e)

# 🚨 Detection Settings
VIOLATION_CLASSES = ["maskoff", "no_glove", "no_hairnet"]
VIOLATION_WINDOW = timedelta(seconds=30)
LOG_INTERVAL = timedelta(seconds=5)

# ♻️ Frame-by-frame Detection
if run:
    success, frame = cap.read()
    if not success:
        st.warning("⚠️ Could not access webcam.")
    else:
        results = model.predict(frame, device="cpu")
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().tolist() if boxes is not None else []
        class_names = [model.names[int(cls)] for cls in class_ids] if class_ids else []
        xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else []

        now = datetime.now()
        violations = [(xyxy[i], class_names[i]) for i in range(len(class_names)) if class_names[i] in VIOLATION_CLASSES]

        if violations and now - st.session_state.last_logged_time > LOG_INTERVAL:
            st.session_state.last_logged_time = now
            st.error(f"🚨 Violation Detected: {', '.join(set([v[1] for v in violations]))}")

            for v in violations:
                st.session_state.violation_log.append({
                    "Time": now,
                    "Type": v[1]
                })

            threading.Thread(target=play_alert).start()

        # Display video frame
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame)

        # Display metrics
        recent_violations = [v for v in st.session_state.violation_log if v["Time"] >= now - VIOLATION_WINDOW]
        col2.metric("📷 Total Detections", len(class_names))
        col2.metric("❌ Violations (30s)", len(recent_violations))

    cap.release()
    time.sleep(0.2)
    st.experimental_rerun()

# 📊 Dashboard Section
if st.session_state.violation_log:
    st.markdown("## 📊 Violation Dashboard")
    df_log = pd.DataFrame(st.session_state.violation_log)
    df_log["Time"] = pd.to_datetime(df_log["Time"])

    # 📂 Downloadable Report
    today = datetime.now().strftime("%Y-%m-%d")
    csv_data = df_log.to_csv(index=False).encode('utf-8')
    st.download_button(label="📂 Download CSV Report", data=csv_data, file_name=f"violations_{today}.csv", mime='text/csv')

    # 📋 Summary Table (Type + First Time + Count)
    summary_table = df_log.groupby("Type").agg(
        First_Detected=("Time", "min"),
        Count=("Type", "count")
    ).reset_index()
    summary_table["First_Detected"] = summary_table["First_Detected"].dt.strftime("%b %d, %Y %H:%M")
    st.markdown("### 📋 Violation Summary Table")
    st.dataframe(summary_table)

    # 📈 Bar Chart
    st.markdown("### 🔺 Violation Trend (Bar Chart)")
    df_chart = df_log.copy()
    df_chart["Minute"] = df_chart["Time"].dt.floor("min")
    chart_data = df_chart.groupby(["Minute", "Type"]).size().reset_index(name="Count")

    chart = alt.Chart(chart_data).mark_bar().encode(
        x="Minute:T",
        y="Count:Q",
        color="Type:N",
        tooltip=["Minute", "Type", "Count"]
    ).properties(width=700, height=400)

    st.altair_chart(chart)

    # 🥧 Pie Chart
    st.markdown("### 🔢 Violation Type Breakdown (Pie Chart)")
    pie_data = df_log["Type"].value_counts().reset_index()
    pie_data.columns = ["Type", "Count"]

    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta="Count",
        color="Type",
        tooltip=["Type", "Count"]
    )
    st.altair_chart(pie_chart)

    # ♻️ Clean old logs
    st.session_state.violation_log = [v for v in st.session_state.violation_log if v["Time"] >= datetime.now() - timedelta(days=1)]

else:
    st.info("ℹ️ No violations recorded yet. Start the camera to begin detection.")
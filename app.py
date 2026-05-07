import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
import time
import io

# Optional paste support
try:
    from streamlit_paste_button import paste_image_button as pbi
    PASTE_AVAILABLE = True
except ImportError:
    PASTE_AVAILABLE = False

# --- Page Config ---
st.set_page_config(
    page_title="Helmet Surveillance",
    page_icon="🏍️",
    layout="centered", # Centered looks cleaner for "simple" apps
    initial_sidebar_state="collapsed"
)

# --- Minimal Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .sub-title {
        text-align: center;
        color: #808495;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "no_helmet.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()
if not model:
    st.error("Model file `no_helmet.pt` not found.")
    st.stop()
names = model.names

# ──────────────────────────────────────────────
# CORE LOGIC
# ──────────────────────────────────────────────
def inside(inner, outer):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return x1 > X1 and y1 > Y1 and x2 < X2 and y2 < Y2

def process_image(frame, conf_threshold):
    results = model(frame, verbose=False)
    motorcyclists, no_helmets, plates, helmets = [], [], [], []
    stats = {"violations": 0, "total_riders": 0}

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            if conf < conf_threshold: continue
            label = names[int(cls)]
            coords = tuple(map(int, box))

            if label == "motorcyclist":
                motorcyclists.append(coords)
                stats["total_riders"] += 1
            elif label == "no-helmet":
                no_helmets.append(coords)
            elif label == "plate":
                plates.append(coords)
            elif label == "helmet":
                helmets.append(coords)

    for rider in motorcyclists:
        has_no_helmet = any(inside(nh, rider) for nh in no_helmets)
        if has_no_helmet:
            stats["violations"] += 1
            x1, y1, x2, y2 = rider
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "NO HELMET", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            for p in plates:
                if inside(p, rider):
                    px1, py1, px2, py2 = p
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    break
    return frame, stats

# ──────────────────────────────────────────────
# UI LAYOUT
# ──────────────────────────────────────────────
st.markdown('<h1 class="main-title">🏍️ Helmet Surveillance</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Traffic Violation Detection System</p>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    conf_thresh = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05)
    st.divider()
    st.markdown("### Developed by")
    st.markdown("[Shaikh Akram](https://github.com/Akram-X207)")

# Main Tabs
tab_upload, tab_paste = st.tabs(["📤 Upload", "📋 Paste Image"])

# ── UPLOAD TAB
with tab_upload:
    source_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov"])
    
    if source_file:
        if source_file.type.startswith("image"):
            # Image Processing
            img = Image.open(source_file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            with st.spinner("Detecting..."):
                output, stats = process_image(frame.copy(), conf_thresh)
                
                st.image(output, channels="BGR", use_container_width=True)
                
                c1, c2 = st.columns(2)
                c1.metric("Total Riders", stats["total_riders"])
                c2.metric("Violations", stats["violations"])
                
                _, buf = cv2.imencode(".jpg", output)
                st.download_button("Download Result", buf.tobytes(), "report.jpg", "image/jpeg")
        
        else:
            # Video Processing (Simplified)
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(source_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            if st.button("Process Video"):
                progress = st.progress(0)
                st_frame = st.empty()
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                curr = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, _ = process_image(frame, conf_thresh)
                    if curr % 10 == 0:
                        st_frame.image(processed, channels="BGR", use_container_width=True)
                    curr += 1
                    progress.progress(min(curr/total_frames, 1.0))
                cap.release()
                st.success("Video processed!")

# ── PASTE TAB
with tab_paste:
    if PASTE_AVAILABLE:
        pasted = pbi("Paste Image from Clipboard")
        if pasted and pasted.image_data:
            frame = cv2.cvtColor(np.array(pasted.image_data.convert("RGB")), cv2.COLOR_RGB2BGR)
            output, stats = process_image(frame, conf_thresh)
            st.image(output, channels="BGR", use_container_width=True)
            st.metric("Violations Found", stats["violations"])
    else:
        st.info("Paste button is available after full deployment.")

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #555;'>© {time.strftime('%Y')} Shaikh Akram</p>", unsafe_allow_html=True)

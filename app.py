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
    page_title="Helmet Surveillance System",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    *, html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0a0a0f 0%, #131320 50%, #0a0a0f 100%);
    }

    /* Hero */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff4b4b, #ff8c42, #ff4b4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        animation: shimmer 4s ease-in-out infinite alternate;
    }
    @keyframes shimmer {
        from { filter: brightness(1); }
        to { filter: brightness(1.25); }
    }
    .hero-subtitle {
        text-align: center;
        color: #6b7394;
        font-size: 1rem;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,75,75,0.25), transparent);
        margin: 16px 0 24px 0;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.2em;
        background: linear-gradient(135deg, #ff4b4b, #ff6b35);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff3333, #ff5522);
        box-shadow: 0 6px 24px rgba(255,75,75,0.35);
        transform: translateY(-2px);
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.2em;
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stDownloadButton>button:hover {
        box-shadow: 0 6px 24px rgba(0,200,83,0.35);
        transform: translateY(-2px);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff4b4b, #ff8c42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #6b7394;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(255,255,255,0.04);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        color: #6b7394;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255,75,75,0.12) !important;
        border-color: rgba(255,75,75,0.35) !important;
        color: #ff4b4b !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff4b4b, #ff8c42);
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(180deg, transparent, rgba(10,10,15,0.95));
        color: #3d4563;
        text-align: center;
        padding: 14px;
        font-size: 0.72rem;
        letter-spacing: 1px;
        z-index: 999;
    }
    .footer a { color: #ff4b4b; text-decoration: none; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Info cards */
    .info-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .info-card h4 {
        color: #ff4b4b;
        margin-bottom: 8px;
    }
    .info-card p {
        color: #8892b0;
        font-size: 0.9rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load YOLO model with fallback paths."""
    paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_helmet.pt"),
        "no_helmet.pt",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return YOLO(p)
            except Exception as e:
                st.error(f"Model load error: {e}")
    st.error("Model file `no_helmet.pt` not found.")
    return None

model = load_model()
if model is None:
    st.stop()
names = model.names


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def inside(inner, outer):
    """Check if inner bounding box is inside outer."""
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return x1 > X1 and y1 > Y1 and x2 < X2 and y2 < Y2


def process_frame(frame, conf_threshold=0.5):
    """Run detection on a single frame and draw annotations."""
    results = model(frame, verbose=False)

    motorcyclists, no_helmets, plates, helmets = [], [], [], []
    stats = {"violations": 0, "total_riders": 0, "helmeted": 0}

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            if conf < conf_threshold:
                continue
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
        has_helmet = any(inside(h, rider) for h in helmets)

        if has_no_helmet:
            stats["violations"] += 1
            x1, y1, x2, y2 = rider
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label_text = "VIOLATION: NO HELMET"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 8, y1), (0, 0, 255), -1)
            cv2.putText(frame, label_text, (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for p in plates:
                if inside(p, rider):
                    px1, py1, px2, py2 = p
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(frame, "PLATE", (px1, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break
        elif has_helmet:
            stats["helmeted"] += 1
            x1, y1, x2, y2 = rider
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, "SAFE", (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    return frame, stats


def safe_load_image(source):
    """Robustly load an image from any source type."""
    try:
        frame = None

        if isinstance(source, Image.Image):
            img = source.convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        elif isinstance(source, (bytes, bytearray)):
            np_arr = np.frombuffer(bytes(source), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        elif isinstance(source, np.ndarray):
            frame = source

        elif hasattr(source, "read"):
            raw = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
            np_arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return None

        # Fix channel issues
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Resize very large images to prevent OOM
        h, w = frame.shape[:2]
        max_dim = 1920
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        return frame
    except Exception:
        return None


def display_results(output_frame, stats):
    """Render processed image and statistics."""
    st.image(output_frame, channels="BGR", use_container_width=True,
             caption="Detection Result")

    # Stats row
    c1, c2, c3 = st.columns(3)
    c1.metric("🏍️ Total Riders", stats["total_riders"])
    c2.metric("🚨 Violations", stats["violations"])
    c3.metric("✅ Helmeted", stats["helmeted"])

    if stats["violations"] > 0:
        st.warning(f"⚠️ {stats['violations']} helmet violation(s) detected!")
    elif stats["total_riders"] > 0:
        st.success("✅ All riders are wearing helmets.")
    else:
        st.info("ℹ️ No motorcyclists detected in this image.")

    # Download
    _, buf = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    st.download_button(
        label="📥 Download Processed Image",
        data=buf.tobytes(),
        file_name="violation_report.jpg",
        mime="image/jpeg",
    )


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/helmet.png", width=72)
    st.title("⚙️ Settings")
    st.markdown("---")

    conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
                            help="Higher = stricter detection, fewer false positives")

    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("""
1. **Upload** an image / video, or  
2. **Paste** an image from clipboard.  
3. Wait for AI analysis.  
4. Download results.
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 About")
    st.info("""
**Helmet Surveillance System**  
AI-powered helmet violation detection.

**Developer:**  
[Shaikh Akram](https://github.com/Akram-X207)
    """)


# ──────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🚦 AI Helmet Surveillance</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Real-time Traffic Violation Detection</p>',
            unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

tab_upload, tab_paste, tab_info = st.tabs(["📤 Upload File", "📋 Paste Image", "📊 System Info"])

# ── TAB 1: UPLOAD ────────────────────────────
with tab_upload:
    uploaded_file = st.file_uploader(
        "Drag & drop or browse — supports JPG, PNG, BMP, WEBP, MP4, AVI, MOV",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "mp4", "avi", "mov"],
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type or ""

        if file_type.startswith("image") or uploaded_file.name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")):
            with st.spinner("🔍 Analyzing image..."):
                frame = safe_load_image(uploaded_file)
                if frame is not None:
                    output, stats = process_frame(frame.copy(), conf_thresh)
                    col_orig, col_res = st.columns(2)
                    with col_orig:
                        st.markdown("**Original**")
                        st.image(frame, channels="BGR", use_container_width=True)
                    with col_res:
                        st.markdown("**Analysis**")
                        display_results(output, stats)
                else:
                    st.error("❌ Could not decode this image. Try a different file.")

        elif file_type.startswith("video") or uploaded_file.name.lower().endswith(
                (".mp4", ".avi", ".mov")):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("❌ Cannot open this video file.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if total_frames <= 0 or width <= 0 or height <= 0:
                    st.error("❌ Invalid video file.")
                else:
                    st.info("🎬 Processing video — this may take a moment...")
                    progress = st.progress(0)
                    preview = st.empty()

                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                    frame_idx = 0
                    total_violations = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        processed, s = process_frame(frame, conf_thresh)
                        total_violations += s["violations"]
                        out.write(processed)

                        if frame_idx % 5 == 0:
                            preview.image(processed, channels="BGR",
                                          use_container_width=True)
                        frame_idx += 1
                        progress.progress(min(frame_idx / total_frames, 1.0))

                    cap.release()
                    out.release()

                    st.success(f"✅ Done — {frame_idx} frames processed, "
                               f"{total_violations} violations found.")

                    with open(out_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f,
                            file_name="processed_violation_video.mp4",
                            mime="video/mp4",
                        )

                    try:
                        os.unlink(out_path)
                    except OSError:
                        pass

            try:
                os.unlink(tfile.name)
            except OSError:
                pass
        else:
            st.error("❌ Unsupported file type. Please upload an image or video.")

# ── TAB 2: PASTE ─────────────────────────────
with tab_paste:
    if PASTE_AVAILABLE:
        st.markdown("Click the button below, then **Ctrl+V** to paste an image from your clipboard.")
        paste_result = pbi(
            label="📋 Paste Image from Clipboard",
            text_color="#ffffff",
            background_color="#ff4b4b",
            hover_background_color="#ff3333",
        )
        if paste_result and paste_result.image_data is not None:
            with st.spinner("🔍 Analyzing pasted image..."):
                frame = safe_load_image(paste_result.image_data)
                if frame is not None:
                    output, stats = process_frame(frame.copy(), conf_thresh)
                    col_orig, col_res = st.columns(2)
                    with col_orig:
                        st.markdown("**Original**")
                        st.image(frame, channels="BGR", use_container_width=True)
                    with col_res:
                        st.markdown("**Analysis**")
                        display_results(output, stats)
                else:
                    st.error("❌ Could not process the pasted image.")
    else:
        st.warning("Paste support is unavailable. Install `streamlit-paste-button`.")
        st.code("pip install streamlit-paste-button", language="bash")

# ── TAB 3: INFO ──────────────────────────────
with tab_info:
    st.subheader("🛡️ System Information")

    st.markdown("""
<div class="info-card">
    <h4>🤖 AI Model</h4>
    <p>YOLOv11 — trained for helmet violation detection with 4 classes:
    <b>motorcyclist</b>, <b>helmet</b>, <b>no-helmet</b>, <b>plate</b>.</p>
</div>
<div class="info-card">
    <h4>⚡ Performance</h4>
    <p>Real-time inference on CPU & GPU. Model is cached for instant reloads.</p>
</div>
<div class="info-card">
    <h4>📦 Supported Formats</h4>
    <p><b>Images:</b> JPG, PNG, BMP, WEBP, TIFF &nbsp;|&nbsp; <b>Videos:</b> MP4, AVI, MOV</p>
</div>
<div class="info-card">
    <h4>🔒 Privacy</h4>
    <p>All processing happens server-side. No images are stored or transmitted externally.</p>
</div>
    """, unsafe_allow_html=True)

    # Show sample
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample", "input.jpeg")
    if os.path.exists(sample_path):
        st.markdown("---")
        st.subheader("🖼️ Sample Detection")
        if st.button("▶ Run on Sample Image"):
            with st.spinner("Processing sample..."):
                sample_frame = cv2.imread(sample_path)
                if sample_frame is not None:
                    output, stats = process_frame(sample_frame.copy(), conf_thresh)
                    display_results(output, stats)
                else:
                    st.error("Could not load sample image.")


# ── FOOTER ────────────────────────────────────
st.markdown(f"""
<div class="footer">
    © {time.strftime("%Y")} <a href="https://github.com/Akram-X207" target="_blank">Shaikh Akram</a>
    &nbsp;•&nbsp; Powered by Ultralytics YOLO
</div>
""", unsafe_allow_html=True)

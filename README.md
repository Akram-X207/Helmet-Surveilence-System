# 🚦 AI Helmet Surveillance System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-green.svg)](https://ultralytics.com/)

A premium computer vision application designed to detect helmet violations in real-time. Built with **YOLOv11** and **Streamlit**, this system provides a seamless interface for traffic surveillance and violation tracking.

**Developed by: [Shaikh Akram](https://github.com/Akram-X207)**

---

## ✨ Features

*   **Multi-Source Input**: Upload images, videos, or **paste directly from clipboard**.
*   **Intelligent Detection**:
    *   Detects Motorcyclists, Helmets, No-Helmets, and Number Plates.
    *   Smart Logic: Only flags riders *without* helmets.
    *   Spatial Mapping: Automatically links riders to their respective number plates.
*   **Premium UI/UX**:
    *   Modern Dark Theme with custom CSS.
    *   Real-time processing preview for videos.
    *   Interactive statistics dashboard.
*   **Violation Reports**: Download processed images/videos with bounding boxes and labels.

---

## 🚀 How It Works

1.  **Detection Phase**: The YOLOv11 model scans the frame for four primary classes.
2.  **Validation Logic**: 
    *   The system checks if a `no-helmet` detection resides within a `motorcyclist` boundary.
    *   It then searches for a `plate` within the same rider's area.
3.  **Reporting**: If a violation is confirmed, the system highlights the rider in red and the plate in green.

---

## 🛠️ Tech Stack

*   **Core**: Python 3.8+
*   **AI Engine**: YOLOv11 (Ultralytics)
*   **Image Processing**: OpenCV, NumPy, PIL
*   **Web Framework**: Streamlit
*   **Deployment**: Ready for Streamlit Cloud / Hugging Face Spaces

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Akram-X207/Helmet-Surveilence-System.git
cd Helmet-Surveilence-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
.
├── app.py              # Main Streamlit application
├── no_helmet.pt        # Pre-trained YOLOv11 model
├── requirements.txt    # Project dependencies
├── packages.txt        # System-level dependencies (for Linux/Cloud)
├── runtime.txt         # Python runtime version
└── README.md           # Project documentation
```

---

## 🛡️ Model Details

*   **Algorithm**: YOLOv11
*   **Classes**: `motorcyclist`, `helmet`, `no-helmet`, `plate`
*   **Optimization**: Cached model loading for high performance.

---

## 🤝 Contributing

This project is maintained by **Shaikh Akram**. Feel free to fork, open issues, or submit PRs to improve the detection accuracy or UI.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Link to Profile:** [github.com/Akram-X207](https://github.com/Akram-X207)

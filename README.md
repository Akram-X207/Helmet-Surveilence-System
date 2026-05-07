# 🚦 AI Helmet Surveillance System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-green.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance computer vision application designed to automate traffic safety monitoring. Built with **YOLOv11** and **Streamlit**, this system detects motorcyclists and identifies helmet violations in real-time, providing a seamless interface for traffic surveillance.

**Live Demo:** [helmet-surveilence-system.streamlit.app](https://helmet-surveilence-system.streamlit.app/)

---

## ✨ Features

*   **Real-time Detection**: Uses YOLOv11 for high-speed object detection in both images and videos.
*   **Intelligent Logic**: 
    *   Detects Motorcyclists, Helmets, No-Helmets, and Number Plates.
    *   **Spatial Mapping**: Automatically associates a violation (no-helmet) and a license plate with the correct rider.
*   **Multi-Source Input**: 
    *   Upload images/videos (JPG, PNG, MP4, MOV).
    *   **Direct Paste**: Support for pasting images from the clipboard.
*   **Downloadable Reports**: Save processed results with annotated bounding boxes for law enforcement records.
*   **Clean & Modern UI**: A user-friendly interface built with Streamlit's native components.

---

## 🛠️ Tech Stack

*   **Language**: Python 3.10
*   **AI Framework**: YOLOv11 (Ultralytics)
*   **Computer Vision**: OpenCV
*   **Web Framework**: Streamlit
*   **Deployment**: Streamlit Cloud / GitHub

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
├── app.py              # Main application logic
├── no_helmet.pt        # Pre-trained YOLOv11 model weights
├── requirements.txt    # Python dependencies
├── packages.txt        # System-level dependencies (for Streamlit Cloud)
├── runtime.txt         # Python runtime version
├── .streamlit/         # Streamlit configuration
└── README.md           # Project documentation
```

---

## 🤝 Contributing (Open Source)

This project is **Open Source** and I welcome contributions! Whether it's fixing a bug, adding a new feature, or improving documentation:

1.  **Fork** the repository.
2.  **Create** a new branch (`git checkout -b feature-branch`).
3.  **Commit** your changes (`git commit -m "Add some feature"`).
4.  **Push** to the branch (`git push origin feature-branch`).
5.  **Open** a Pull Request.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 Developed By

**Shaikh Akram**  
*   GitHub: [@Akram-X207](https://github.com/Akram-X207)
*   Portfolio: [Akram-X207.github.io](https://Akram-X207.github.io/) (if applicable)

---
*Powered by Ultralytics YOLOv11*

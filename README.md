# Real-Time Multimodal Sign Language Recognition System

## 📌 Overview

A real-time AI-powered sign language recognition system that converts hand gestures into text and speech using MediaPipe hand landmark extraction and a feedforward deep neural network.

The system enables multimodal communication support for speech- and hearing-impaired individuals through real-time gesture recognition, text generation, and offline text-to-speech synthesis.

---

## 🚀 Features

* Real-time webcam-based gesture recognition
* MediaPipe hand landmark extraction
* Feedforward neural network classification
* Text output generation
* Offline text-to-speech synthesis
* Multimodal pipeline (Video → Text + Speech)
* TensorFlow Lite (TFLite) optimized deployment model
* Lightweight real-time inference pipeline

---

## 🧠 Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* TensorFlow Lite (TFLite)
* NumPy
* pyttsx3

---

## ⚡ TensorFlow Lite Optimization

The trained feedforward neural network model was optimized using TensorFlow Lite (TFLite) for lightweight edge deployment and efficient low-latency inference.

### Optimization Results

* Original Model Size: 0.23 MB
* TFLite Model Size: 0.02 MB
* Model Size Reduction: ~91%

### Benefits

* Faster inference
* Reduced deployment overhead
* Edge-device compatibility
* Lightweight real-time prediction

---

## 📂 Project Structure

```text
sign-language-project/
│
├── data/                     # Raw gesture image dataset
├── data_processed/           # Processed landmark datasets
│
├── models/
│   ├── sign_model.h5         # Original trained model
│   └── sign_model.tflite     # Optimized TFLite model
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   ├── data_collection.py
│   └── test_setup.py
│
├── convert_to_tflite.py      # TFLite conversion script
├── test_tflite.py            # TFLite inference testing
│
├── demo.gif
├── requirements.txt
└── README.md
```

---

## ▶️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/harshitsharma544/sign-language-recognition.git
```

### 2. Create Virtual Environment

```bash
py -3.10 -m venv venv
```

### 3. Activate Virtual Environment

```bash
.\venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Real-Time Prediction

```bash
python src/predict.py
```

---

## 🎬 Demo

![Demo](demo.gif)

---

## 🎯 Use Case

This project helps speech- and hearing-impaired individuals communicate more effectively through AI-powered real-time gesture interpretation.

---

## 📌 Future Improvements

* Add more gesture classes and vocabulary
* Improve model robustness using larger datasets
* Integrate CNN-based spatial feature extraction
* Add NLP-based sentence generation pipelines
* Deploy as a mobile/web application
* Integrate full TensorFlow Lite real-time inference pipeline

---

## 📊 Model Pipeline

```text
Webcam Input
      ↓
MediaPipe Hand Landmark Extraction
      ↓
63-Dimensional Feature Vector
      ↓
Feedforward Neural Network
      ↓
Gesture Prediction
      ↓
Text + Speech Output
```

---

## 📄 Notes

* Large processed datasets are excluded due to repository size limitations.
* The current real-time inference pipeline uses the original TensorFlow/Keras `.h5` model.
* TensorFlow Lite optimization was implemented for future lightweight edge-device deployment and low-latency inference.

---

⭐ If you like this project, consider giving it a star!

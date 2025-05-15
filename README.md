# 🎭 Deepfake Detection using CNN + BiLSTM + Attention

This project implements a custom deepfake detection pipeline that combines spatial and temporal modeling. It is designed to generalize across manipulation techniques (e.g., DeepFake GANs, FaceSwap) and remains robust to video compression.

---

## 📌 Key Features

- CNN backbone for fine-grained spatial feature extraction
- BiLSTM to model temporal dependencies across frames
- Attention mechanism to focus on suspicious frames
- Confidence scoring and augmentations for real-world robustness

---

## 🧠 Model Architecture

```plaintext
Input Video
   │
Frame Extraction & Face Detection
   │
Face Alignment & Cropping
   │
CNN Feature Extraction (ResNet18 / MobileNetV2)
   │
Sequence Modeling with BiLSTM
   │
Weighted Temporal Attention
   │
Sigmoid Classifier → [0 (Real), 1 (Fake)]

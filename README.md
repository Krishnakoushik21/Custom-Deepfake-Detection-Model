# 🎭 Deepfake Detection using CNN + BiLSTM + Attention

This project implements a custom deepfake detection pipeline that combines spatial and temporal modeling. It is designed to generalize across manipulation techniques (e.g., DeepFake GANs, FaceSwap) and remains robust to video compression.

---

## 📌 Key Features

- CNN backbone for fine-grained spatial feature extraction
- BiLSTM to model temporal dependencies across frames
- Attention mechanism to focus on suspicious frames
- Confidence scoring and augmentations for real-world robustness

---

## DataSet 
- Get it from Kaggle -->https://www.kaggle.com/datasets/xdxd003/ff-c23
- or U can also use -->https://www.kaggle.com/code/hamditarek/deepfake-detection-challenge-kaggle/input For Training the Model 

## 📁 Project Structure

.
├── DATA/
│ ├── FAKE/ # Sample deepfake videos
│ └── REAL/ # Sample authentic videos
├── model/
│ ├── cnn_extractor.py
│ ├── bilstm_attention.py
│ └── model_wrapper.py
├── preprocessing/
│ ├── face_detector.py
│ └── align_crop.py
├── utils/
│ ├── metrics.py
│ └── evaluate.py
├── train.py # Training script
├── Requirements.txt # Dependencies
├── README.md # Project documentation

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




## 🚀 Getting Started

### 1. Clone the repo
git clone https://github.com/Krishnakoushik21/Custom-Deepfake-Detection-Model.git
cd Custom-Deepfake-Detection-Model
2. Install dependencies

pip install -r Requirements.txt

3. Run Training


python train.py


4. Run Evaluation

python utils/evaluate.py

# model/model_wrapper.py
from model.cnn_extractor import CNNFeatureExtractor
from model.bilstm_attention import BiLSTMAttentionModel

class DeepfakeDetector:
    def __init__(self, device):
        self.feature_extractor = CNNFeatureExtractor().to(device)
        self.temporal_model = BiLSTMAttentionModel().to(device)
        self.device = device

    def predict_video(self, frames_tensor):
        features = []
        for frame in frames_tensor:
            with torch.no_grad():
                feat = self.feature_extractor(frame.unsqueeze(0).to(self.device))
                features.append(feat.squeeze(0))
        sequence = torch.stack(features).unsqueeze(0)
        return self.temporal_model(sequence)

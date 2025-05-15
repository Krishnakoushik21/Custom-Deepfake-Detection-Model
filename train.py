# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocessing.face_detector import detect_faces_from_video
from preprocessing.align_crop import align_and_crop
from model.model_wrapper import DeepfakeDetector
from utils.metrics import compute_metrics
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset for videos
class VideoDataset(Dataset):
    def __init__(self, root_dir, label, transform=None, max_frames=15):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.label = label
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = self.files[idx]
        frames = detect_faces_from_video(video_path)
        frames = frames[:self.max_frames]  # Truncate to N frames

        tensor_frames = []
        for frame in frames:
            aligned = align_and_crop(frame)
            if self.transform:
                aligned = self.transform(aligned)
            tensor_frames.append(aligned)

        # If not enough frames, pad with zeros
        while len(tensor_frames) < self.max_frames:
            tensor_frames.append(torch.zeros_like(tensor_frames[0]))

        sequence = torch.stack(tensor_frames)
        return sequence, self.label

def load_data(data_dir, batch_size=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    real_dataset = VideoDataset(os.path.join(data_dir, "real"), 0, transform)
    fake_dataset = VideoDataset(os.path.join(data_dir, "fake"), 1, transform)

    dataset = torch.utils.data.ConcatDataset([real_dataset, fake_dataset])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train():
    model = DeepfakeDetector(device)
    optimizer = torch.optim.Adam(model.temporal_model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    dataloader = load_data("data", batch_size=2)
    model.temporal_model.train()

    for epoch in range(1, 6):  # 5 epochs
        all_labels = []
        all_preds = []
        running_loss = 0

        for sequences, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            predictions = []

            for sequence in sequences:
                pred = model.predict_video(sequence).squeeze()
                predictions.append(pred)

            predictions = torch.stack(predictions)
            labels = labels.float().to(device)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_binary = (predictions > 0.5).int().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds_binary)

        metrics = compute_metrics(all_labels, all_preds)
        print(f"\nEpoch {epoch} Loss: {running_loss:.4f}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.temporal_model.state_dict(), "checkpoints/best_model.pth")
    print("✅ Model saved to checkpoints/best_model.pth")

if __name__ == "__main__":
    train()

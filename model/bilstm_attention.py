# model/bilstm_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        weights = F.softmax(self.attn(lstm_output), dim=1)
        return (lstm_output * weights).sum(dim=1)

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        return torch.sigmoid(self.classifier(attn_out))

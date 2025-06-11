import torch.nn as nn
import torch.nn.functional as F

class SleepScoringModel(nn.Module):
    def __init__(self, dropout_rate=0.5, weight_decay=1e-4):
        super(SleepScoringModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 1280, 128)
        self.fc2 = nn.Linear(128, 3)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    """
    A modified Convolutional Neural Network for EMNIST with final pooling removed
    to improve Grad-CAM localisation. Also adjusts fully connected layer accordingly.
    """
    def __init__(self):
        super().__init__()

        # --- Convolutional Layers ---
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Updated for larger feature map (7x7 instead of 3x3)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 62)  # 62 classes for EMNIST ByClass

    def forward(self, x):
        # Conv block 1: (1, 28, 28) → (32, 14, 14)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # Conv block 2: (32, 14, 14) → (64, 7, 7)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # Conv block 3: (64, 7, 7) → (128, 7, 7) — NO pooling here
        x = self.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
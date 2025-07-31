import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    """
    An improved Convolutional Neural Network for image classification.

    This architecture incorporates Batch Normalization for stable and faster 
    training, and Dropout for regularization to prevent overfitting. Padding is 
    used in convolutional layers to maintain feature map dimensions through
    the network.
    """
    def __init__(self):
        super().__init__()

        # --- Convolutional Layers ---
        # Each convolutional block consists of Conv2D -> BatchNorm -> ReLU -> MaxPool

        # Block 1: Input (1, 28, 28) -> Output (32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        # Block 2: Input (32, 14, 14) -> Output (64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        
        # Block 3: Input (64, 7, 7) -> Output (128, 3, 3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        # --- Shared Layers ---
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # --- Fully Connected (Classifier) Layers ---
        self.fc1 = nn.Linear(in_features=128 * 3 * 3, out_features=256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=62) # 62 classes for EMNIST ByClass dataset

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, 62).
        """
        # Sequentially pass through the convolutional blocks
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten the feature map to a vector for the classifier
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

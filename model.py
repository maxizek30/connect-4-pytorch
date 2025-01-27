import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    A CNN-based network for Connect4 Q-learning.
    Input shape: (batch, 1, 6, 7)
    Output shape: (batch, num_actions=7)
    """

    def __init__(self, num_actions=7):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # Flatten dimension depends on how your convolution layers reduce.
        # For a 6x7 board (no padding/stride), shape after conv2 is (batch, 64, 2, 3).
        # That's 64 * 2 * 3 = 384 total features per sample.
        # We'll compute it dynamically in forward by using x.view.
        self.fc1 = nn.Linear(64 * 2 * 3, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        # x shape: (batch_size, 1, 6, 7)
        x = F.relu(self.conv1(x))  # -> (batch_size, 32, 4, 5)
        x = F.relu(self.conv2(x))  # -> (batch_size, 64, 2, 3)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # final layer => Q-values, no activation
        return x

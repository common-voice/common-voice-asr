import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNEncoder(nn.Module):
    def __init__(self, in_channels = 1, num_classes=27):
        super().__init__()
        # two conv blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Linear(64 * 20, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2) # (T, N, C)

        x = self.classifier(x)
        logits = torch.log_softmax(x, dim=2)
        return logits

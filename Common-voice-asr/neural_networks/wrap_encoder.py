import torch.nn as nn
import torch.nn.functional as F

class WrapEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, apply_head = True):
        super().__init__()
        self.encoder = encoder
        self.apply_head = apply_head
        if apply_head:
            self.head = nn.Linear(256, num_classes)
        else:
            self.head = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x



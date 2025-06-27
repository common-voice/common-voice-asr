import torch.nn as nn
import torch.nn.functional as F

class WrapEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, apply_head = True):
        super().__init__()
        self.encoder = encoder
        self.apply_head = apply_head
        self.num_classes = num_classes

    def forward(self, x):
        x = self.encoder(x)
        if self.apply_head:
            feature_dim = x.shape[-1]
            self.head = nn.Linear(feature_dim, self.num_classes)
            x = self.head(x)
        else:
            self.head = nn.Identity()
        return x



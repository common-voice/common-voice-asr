import torch.nn as nn

class WrapEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)



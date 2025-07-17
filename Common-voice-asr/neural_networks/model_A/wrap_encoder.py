import torch.nn as nn


class WrapEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, apply_head=True):
        super().__init__()
        self.encoder = encoder
        self.apply_head = apply_head
        self.num_classes = num_classes
        if self.apply_head:
            self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.encoder.output_size, num_classes))
            # self.classifier = nn.Linear(self.encoder.output_size, num_classes)
            # nn.init.xavier_uniform_(self.classifier.weight)
            # nn.init.zeros_(self.classifier.bias)
        else:
            self.classifier = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

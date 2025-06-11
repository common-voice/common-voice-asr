import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_size=128, projection_size=256, num_classes = 27):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, projection_size)
        self.lstm = nn.LSTM(input_size=projection_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        output, (hn, cn) = self.lstm(x)
        logits = self.output_proj(output)
        return logits


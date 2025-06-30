import torch.nn as nn


class CTC_RNNEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_size=128, projection_size=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, projection_size)
        self.lstm = nn.LSTM(projection_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.output_size = hidden_size * 2

    def forward(self, x):
        x = self.input_proj(x)
        output, (hn, cn) = self.lstm(x)
        return output


class CEL_RNNEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_size=128, projection_size=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, projection_size)
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(hidden_size * 2, projection_size)

    def forward(self, x):
        x = self.input_proj(x)
        output, (hn, cn) = self.lstm(x)

        x = output.mean(dim=1)
        x = self.output_proj(x)
        return x


import torch.nn as nn
from neural_networks.model_A.cnn_encoder import CTC_CNNEncoder


class CTC_RNNEncoder(nn.Module):
    def __init__(self, input_dim=80, cnn_hidden=32, hidden_size=128, projection_size=256):
        super().__init__()
        self.cnn = CTC_CNNEncoder(hidden_dim=cnn_hidden, input_freq_bins=input_dim, layer=True)
        input_dim = self.cnn.output_channels * self.cnn.downsampled_freq
        self.input_proj = nn.LazyLinear(projection_size)
        # self.input_proj = nn.Linear(input_dim, projection_size)
        
        self.lstm = nn.LSTM(projection_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.output_size = hidden_size * 2

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = x.unsqueeze(1)
        # x = x.permute(0, 1, 3, 2)
        x = self.cnn(x)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        output, (hn, cn) = self.lstm(x)
        return output


class CEL_RNNEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_size=128, projection_size=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, projection_size)
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(hidden_size * 2, projection_size)
        self.output_size = hidden_size * 2

    def forward(self, x):
        x = self.input_proj(x)
        output, (hn, cn) = self.lstm(x)

        x = output.mean(dim=1)
        x = self.output_proj(x)
        return x

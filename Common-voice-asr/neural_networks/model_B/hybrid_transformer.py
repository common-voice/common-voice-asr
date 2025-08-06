import torch.nn as nn
from neural_networks.model_A.cnn_encoder import CTC_CNNEncoder
from neural_networks.model_B.positional_encoding import PositionalEncoding


class HybridTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, cnn_hidden=32, d_model=512, nhead=8, dim_feedforward=2048, nlayers=6,
                 lstm_hidden=256, lstm_layers=1, dropout=0.5, conv_layer=False):
        super().__init__()
        self.conv_layer = conv_layer
        if conv_layer:
            self.cnn = CTC_CNNEncoder(hidden_dim=cnn_hidden, input_freq_bins=input_dim, layer=True)
            input_dim = self.cnn.output_channels * self.cnn.downsampled_freq
        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, num_layers=lstm_layers, bidirectional=True)

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_hidden * 2, vocab_size))

    def forward(self, x):
        # print("Input to encoder: ", x.shape)
        if self.conv_layer:
            x = self.cnn(x)
        # print("Before inputting to linear: ", x.shape)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # print("Before inputting to lstm", x.shape)
        x, _ = self.lstm(x)  # accept (Batch, Time, Features)
        x = self.classifier(x)
        # print("Final model output: ", x.shape)
        return x

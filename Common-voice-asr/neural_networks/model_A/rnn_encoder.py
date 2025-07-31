import torch.nn as nn
import torch
from neural_networks.model_A.cnn_encoder import CTC_CNNEncoder

class CTC_RNNEncoder(nn.Module):
    def __init__(self, num_classes, input_dim=80, cnn_hidden=32, hidden_size=128, projection_size=256):
        super().__init__()
        # 1. The CNN frontend, which downsamples by 4x
        self.cnn = CTC_CNNEncoder(hidden_dim=cnn_hidden, input_freq_bins=input_dim, layer=True)

        # 2. A projection layer to match LSTM input size
        # The input dimension comes from the flattened output of the CNN
        cnn_output_dim = self.cnn.output_channels * self.cnn.downsampled_freq
        self.input_proj = nn.Linear(cnn_output_dim, projection_size)
        
        # 3. The LSTM layer. It's configured to accept (Batch, Time, Features)
        self.lstm = nn.LSTM(projection_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        
        # 4. The final classifier layer. This makes the model complete.
        # The output is from the bidirectional LSTM (hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Input x from collate_fn is expected as (Batch, Time, Freq), e.g., [1, 374, 80]
        # Permute and unsqueeze for the Conv2D, which expects (B, Channels, Freq, Time)
        x = x.permute(0, 2, 1) # -> (B, Freq, Time), e.g., [1, 80, 374]
        x = x.unsqueeze(1)     # -> (B, 1, Freq, Time)

        # Pass through the CNN frontend
        x = self.cnn(x)
        # CNN output is (B, T_downsampled, Features), e.g., [1, 93, 640]

        # Project to the LSTM's expected input dimension
        x = self.input_proj(x)
        # Shape is now (B, T_downsampled, projection_size), e.g., [1, 93, 256]
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("NaNs/Infs before LSTM input")

        # Pass through the LSTM. The shape is already correct for batch_first=True.
        x, _ = self.lstm(x)
        # LSTM output is (B, T_downsampled, hidden_size * 2), e.g., [1, 93, 256]

        # Pass through the final classifier
        x = self.classifier(x)
        # Output is (B, T_downsampled, num_classes), e.g., [1, 93, 31]

        return x
    
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
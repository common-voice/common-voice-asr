import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.wrap_encoder import WrapEncoder
from neural_networks.cnn_encoder import CTC_CNNEncoder
from neural_networks.rnn_encoder import CTC_RNNEncoder
from neural_networks.datasets import char_to_id

# In tests/test_ctc_pipeline.py, create a dummy batch of shape [2,1,80,50] with a short transcript, 
# run one forward+loss call, and assert loss is a finite scalar.
# pytest Common-voice-asr/tests/test_ctc_pipeline.py


def test_ctc_forward_pass_cnn():
    batch_size = 2
    channels = 1
    freq_bins = 80
    time_steps = 50
    inputs = torch.rand(batch_size, channels, freq_bins, time_steps)

    transcripts = ['CAT', 'DOG']
    char_id_dict = char_to_id()
    targets = [torch.tensor([char_id_dict[c] for c in t]) for t in transcripts]
    target_lengths = torch.tensor([len(t) for t in targets])
    targets = torch.cat(targets)
    
    encoder = CTC_CNNEncoder()
    model = WrapEncoder(encoder, 36, False)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)
    input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    assert torch.isfinite(loss).item(), "CTC loss is not finite"
    assert loss.ndim == 0, "CTC loss should be a scalar"
    print("All tests for CNN Encoder passed.")


def test_ctc_forward_pass_rnn():
    batch_size = 2
    channels = 1
    freq_bins = 80
    time_steps = 50
    inputs = torch.rand(batch_size, channels, freq_bins, time_steps)
    inputs = inputs.squeeze(1).permute(0, 2, 1)

    transcripts = ['CAT', 'DOG']
    char_id_dict = char_to_id()
    targets = [torch.tensor([char_id_dict[c] for c in t]) for t in transcripts]
    target_lengths = torch.tensor([len(t) for t in targets])
    targets = torch.cat(targets)
    
    encoder = CTC_RNNEncoder()
    model = WrapEncoder(encoder, 36)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)
    input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    assert torch.isfinite(loss).item(), "CTC loss is not finite"
    assert loss.ndim == 0, "CTC loss should be a scalar"
    print("All tests for RNN Encoder passed.")


if __name__ == "__main__":
    test_ctc_forward_pass_cnn()
    test_ctc_forward_pass_rnn()

    

    

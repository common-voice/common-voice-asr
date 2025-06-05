#to run: python -m tests.test_rnn_encoder (via Common-voice-asr)
import os
import numpy as np
import torch
from neural_networks.rnn_encoder import RNNEncoder
from neural_networks.wrap_encoder import WrapEncoder

PROCESSED_DIR = "/Users/setongerrity/Desktop/Mozilla/common-voice-asr/Common-voice-asr/data/processed/mini_cv"
NUM_TEST_FILES = 5

def test_single_forward_pass():
    encoder = RNNEncoder()
    encoder.eval()

    npy_files = [file for file in os.listdir(PROCESSED_DIR) if file.endswith(".npy")]
    assert len(npy_files) > 0, "No files found for testing"

    path = os.path.join(PROCESSED_DIR, npy_files[0])
    spect = np.load(path)

    spect_tensor = torch.tensor(spect, dtype=torch.float32).transpose(0,1).unsqueeze(0)

    with torch.no_grad():
        output = encoder(spect_tensor)
    
    assert output.shape == (1, 256), f"Unexpected shape: {output.shape}"
    


def test_batch_forward_pass():
    encoder = RNNEncoder()
    encoder.eval()

    npy_files = [file for file in os.listdir(PROCESSED_DIR) if file.endswith(".npy")][:NUM_TEST_FILES]
    tensors = []
    max_width = 0

    for filename in npy_files:
        path = os.path.join(PROCESSED_DIR, filename)
        spect = np.load(path)
        max_width = max(max_width, spect.shape[1])
        tensors.append(spect)

    padded_spects = []
    for spect in tensors:
        pad_width = max_width - spect.shape[1]
        padded = np.pad(spect, ((0,0), (0, pad_width)), mode='constant')

        padded = padded.T
        padded_spects.append(torch.tensor(padded, dtype=torch.float32))

    batch_tensor = torch.stack(padded_spects)

    with torch.no_grad():
        output = encoder(batch_tensor)
    
    assert output.shape[0] == len(npy_files), "Batch size does not match # files"
    assert output.shape[1] == 256, "Output feature dimension does not equal 256"

def test_variable_sequence_lengths():
    encoder = RNNEncoder()
    encoder.eval()
    lengths = [80, 120, 160]
    for l in lengths:
        x = torch.randn(1, l, 80)
        with torch.no_grad():
            output = encoder(x)
        assert output.shape == (1, 256)

def test_wrapped_encoder_single_forward_pass():
    encoder = RNNEncoder()
    wrap_encoder = WrapEncoder(encoder)
    wrap_encoder.eval()

    npy_files = [file for file in os.listdir(PROCESSED_DIR) if file.endswith(".npy")]
    spect = np.load(os.path.join(PROCESSED_DIR, npy_files[0]))

    tensor = torch.tensor(spect.T, dtype=torch.float32).unsqueeze(0)

    if (tensor[-1] < 300).any():
        pad_width = 300 - tensor.shape[1]
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_width))
    
    with torch.no_grad():
        output = wrap_encoder(tensor)

    assert output.shape == (1, 10)
    

if __name__ == "__main__":
    test_single_forward_pass()
    test_batch_forward_pass()
    test_variable_sequence_lengths()
    test_wrapped_encoder_single_forward_pass()
    print("All tests passed.")
#to run: python -m tests.test_rnn_encoder (via Common-voice-asr)
import os
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pathlib import Path
from neural_networks.rnn_encoder import CEL_RNNEncoder
from neural_networks.wrap_encoder import WrapEncoder

load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR"))
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mini_cv"
NUM_TEST_FILES = 5


def test_single_forward_pass():
    encoder = CEL_RNNEncoder()
    encoder.eval()

    npy_files = [file for file in PROCESSED_DIR.glob("*.npy")]
    assert len(npy_files) > 0, "No files found for testing"

    path = os.path.join(PROCESSED_DIR, npy_files[0])
    spect = np.load(path)

    spect_tensor = torch.tensor(spect, dtype=torch.float32)

    spect_tensor = spect_tensor.transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        output = encoder(spect_tensor)
        
    assert output.shape == (1, 256), f"Unexpected shape: {output.shape}"


def test_batch_forward_pass():
    encoder = CEL_RNNEncoder()
    encoder.eval()

    npy_files = [file for file in PROCESSED_DIR.glob("*.npy")][:NUM_TEST_FILES]
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
        padded = np.pad(spect, ((0, 0), (0, pad_width)), mode='constant')

        padded = padded.T
        padded_spects.append(torch.tensor(padded, dtype=torch.float32))

    batch_tensor = torch.stack(padded_spects)

    with torch.no_grad():
        output = encoder(batch_tensor)
    
    assert output.shape[0] == len(npy_files), "Batch size does not match # files"
    assert output.shape[1] == 256, "Output feature dimension does not equal 256"


def test_variable_sequence_lengths():
    encoder = CEL_RNNEncoder()
    encoder.eval()
    lengths = [80, 120, 160]
    for length in lengths:
        x = torch.randn(1, length, 80)
        with torch.no_grad():
            output = encoder(x)
        assert output.shape == (1, 256)


def test_wrapped_encoder_single_forward_pass():
    encoder = CEL_RNNEncoder()
    wrap_encoder = WrapEncoder(encoder, 10)
    wrap_encoder.eval()

    npy_files = [file for file in PROCESSED_DIR.glob("*.npy")]
    spect = np.load(os.path.join(PROCESSED_DIR, npy_files[0]))

    tensor = torch.tensor(spect.T, dtype=torch.float32).unsqueeze(0)

    if (tensor[-1] < 300).any():
        pad_len = 300 - tensor.shape[1]
        tensor = F.pad(tensor, (0, 0, 0, pad_len)) 
    
    with torch.no_grad():
        output = wrap_encoder(tensor)

    assert output.shape == (1, 10)
    

if __name__ == "__main__":
    test_single_forward_pass()
    test_batch_forward_pass()
    test_variable_sequence_lengths()
    test_wrapped_encoder_single_forward_pass()
    print("All tests passed.")
# to run: python -m tests.test_cnn_encoder (via Common-voice-asr)
import os
import numpy as np
import torch
from dotenv import load_dotenv
from pathlib import Path
from neural_networks.cnn_encoder import CEL_CNNEncoder
from neural_networks.wrap_encoder import WrapEncoder

load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR"))
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mini_cv"
NUM_TEST_FILES = 5


def load_spectogram(path):
    spect = np.load(path)
    assert spect.shape[0] == 80, f"Expected 80 mel bands, got {spect.shape[0]}"
    spect_tensor = torch.tensor(spect, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return spect_tensor


def test_single_forward_pass():
    encoder = CEL_CNNEncoder()
    encoder.eval()

    npy_files = [file for file in PROCESSED_DIR.glob("*.npy")][:NUM_TEST_FILES]
    assert len(npy_files) > 0, "No files found for testing"

    for filename in npy_files:
        path = os.path.join(PROCESSED_DIR, filename)
        input_tensor = load_spectogram(path)
        with torch.no_grad():
            output = encoder(input_tensor)

        assert output.shape == (1, 256), f"Unexpected output shape: {output.shape}"


def test_batch_forward_pass():
    encoder = CEL_CNNEncoder()
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
        padded_spects.append(torch.tensor(padded, dtype=torch.float32).unsqueeze(0))

    batch_tensor = torch.stack(padded_spects)

    with torch.no_grad():
        output = encoder(batch_tensor)

    assert output.shape[0] == len(npy_files), "Batch size does not match # files"
    assert output.shape[1] == 256, "Output feature dimension does not equal 256"


def test_wrapped_encoder_single_forward_pass():
    encoder = CEL_CNNEncoder()
    wrap_encoder = WrapEncoder(encoder, 10)
    wrap_encoder.eval()

    npy_files = [file for file in PROCESSED_DIR.glob("*.npy")]
    spect = np.load(os.path.join(PROCESSED_DIR, npy_files[0]))

    tensor = torch.tensor(spect, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if (tensor[-1] < 300).any():
        pad_width = 300 - tensor.shape[1]
        tensor = torch.nn.functional.pad(tensor, (0, pad_width))

    with torch.no_grad():
        output = wrap_encoder(tensor)

    assert output.shape == (1, 10)


if __name__ == "__main__":
    test_single_forward_pass()
    test_batch_forward_pass()
    test_wrapped_encoder_single_forward_pass()
    print("All tests passed.")

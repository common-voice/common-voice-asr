import os
import librosa
import numpy as np
import argparse
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80

def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_mini', action='store_true', help='Load full mini dataset')
    return parser.parse_args()

def preprocess(raw_audio_dir, output_dir):
    for filename in os.listdir(raw_audio_dir):
        if filename.endswith(".mp3"):
            input_path = os.path.join(raw_audio_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".mp3", ".npy"))

            try:
                y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0, top_db=80)
                np.save(output_path, log_mel_spec)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main(full_mini: bool = False):
    if full_mini:
        raw_audio_dir = os.path.join(BASE_DIR, "data/raw/full_mini_cv")
        output_dir = os.path.join(BASE_DIR, "data/processed/full_mini_cv")
    else:
        raw_audio_dir = os.path.join(BASE_DIR, "data/raw/mini_cv")
        output_dir = os.path.join(BASE_DIR, "data/processed/mini_cv")
    preprocess(raw_audio_dir, output_dir)

if __name__ == "__main__":
    args = parse_command_args()
    main(full_mini=args.full_mini)


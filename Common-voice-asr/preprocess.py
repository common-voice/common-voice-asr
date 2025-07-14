import os
import librosa
import numpy as np
import argparse
import random
from dotenv import load_dotenv
from pathlib import Path
from itertools import product

# cd Common-voice-asr python -m preprocess --corpus
load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))

sample_rate = [16000, 22050]
n_fft = [512, 1024, 2048]
hop_length = [256, 512]
n_mels = [80, 128]
param_combos = list(product(sample_rate, n_fft, hop_length, n_mels))
train_sample = 1275
dev_sample = 225


def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_mini', action='store_true', help='Load full mini dataset')
    parser.add_argument('--corpus', action='store_true', help='Preprocess corpus dataset, split into train & dev')
    parser.add_argument('--sample', action='store_true', help='Generate samples of mel spectograms based on varying parameters')
    return parser.parse_args()


def preprocess(raw_audio_dir, output_dir, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=80, subset=None):
    os.makedirs(output_dir, exist_ok=True)
    if subset is not None:
        file_list = subset
    else:
        file_list = os.listdir(raw_audio_dir)
    for filename in file_list:
        if filename.endswith(".mp3"):
            input_path = os.path.join(raw_audio_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".mp3", ".npy"))
            
            if os.path.exists(output_path):
                continue

            try:
                y, sr = librosa.load(input_path, sr=sample_rate)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0, top_db=80)
                np.save(output_path, log_mel_spec)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def get_sample(raw_audio_dir, sample_type):
    files = [f for f in os.listdir(raw_audio_dir) if f.endswith(".mp3")]
    if sample_type == 'dev':
        sample_size = dev_sample
    else:
        sample_size = train_sample
    return random.sample(files, min(len(files), sample_size))
                

def main(full_mini: bool = False, corpus: bool = False, sample: bool = False):
    if corpus:
        for split in ['dev', 'train']:
            raw_audio_dir = os.path.join(BASE_DIR, f"corpus_data/raw/{split}_cv")
            output_dir = os.path.join(BASE_DIR, f"corpus_data/processed/{split}_cv")
            preprocess(raw_audio_dir, output_dir)
    elif sample:
        for split in ['dev', 'train']:
            raw_audio_dir = os.path.join(BASE_DIR, f"corpus_data/raw/{split}_cv")
            sample_files = get_sample(raw_audio_dir, split)
            for sr, n_fft, hop, n_mels in param_combos:
                output_dir = os.path.join(BASE_DIR,
                                          f"corpus_data/processed/{split}_cv/sample/sr{sr}_nfft{n_fft}_hop{hop}_nmels{n_mels}")
                preprocess(raw_audio_dir, output_dir, sr, n_fft, hop, n_mels, subset=sample_files)       
    else:
        if full_mini:
            raw_audio_dir = os.path.join(BASE_DIR, "data/raw/full_mini_cv")
            output_dir = os.path.join(BASE_DIR, "data/processed/full_mini_cv")
        else:
            raw_audio_dir = os.path.join(BASE_DIR, "data/raw/mini_cv")
            output_dir = os.path.join(BASE_DIR, "data/processed/mini_cv")
        preprocess(raw_audio_dir, output_dir)


# python -m common-voice-asr.Common-voice-asr.preprocess --corpus 
if __name__ == "__main__":
    args = parse_command_args()
    main(full_mini=args.full_mini, corpus=args.corpus, sample=args.sample)

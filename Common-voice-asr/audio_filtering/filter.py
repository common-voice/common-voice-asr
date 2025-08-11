import librosa
import numpy as np
import torch
import os
import shutil
import csv
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))
DEV_DIR = os.path.join(BASE_DIR, "corpus_data/raw/dev_cv")
TRAIN_DIR = os.path.join(BASE_DIR, "corpus_data/raw/train_cv")
filter_base_dir = os.path.join(BASE_DIR, "audio_filtering")
CLEAN_DIR = os.path.join(filter_base_dir, "clean")
NOISY_DIR = os.path.join(filter_base_dir, "noisy")
MANIFEST_PATH = os.path.join(filter_base_dir, "audio_metrics.csv")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


def voiced_percentage(path, threshold=0.5):
    # silerovad expects 16 kHz mono
    sr = 16000
    wav = read_audio(str(path), sampling_rate=sr)

    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)
    total_samples = len(wav)
    voiced_samples = sum([segment['end'] - segment['start'] for segment in speech_timestamps])
    voice_ratio = voiced_samples / total_samples
    return voice_ratio


def get_audio_metrics(path, frame_ms=20, hop_ms=10):
    y, sr = librosa.load(str(path))
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    # root mean square of signal calculation
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]

    # peak amplitudes per frame
    peaks = []
    for i in range(0, len(y)-frame_len+1, hop_len):
        frame = y[i:i+frame_len]
        peaks.append(np.max(np.abs(frame)))
    peaks = np.array(peaks)

    # signal-to-noise ratio
    noise_rms_estimate = np.median(rms[:max(1, int(0.5*sr/hop_len))])
    voice_mask = rms > (noise_rms_estimate * 2)
    if voice_mask.any():
        signal_rms = rms[voice_mask].mean()
    else:
        signal_rms = 0.0
    snr_db = 20 * np.log10(((signal_rms + 1e-12) / (noise_rms_estimate + 1e-12)))

    return float(np.mean(rms)), float(np.max(peaks)), float(snr_db)


def process_audio(audio_dir, rms_min=0.02, amp_max=1.0, snr_min=10, voice_ratio_min=0.5):
    with open(MANIFEST_PATH, 'w', newline='', encoding='utf-8') as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(['filename', 'avg_rms', 'peak_amp', 'signal_noise_ratio', 'voice_ratio', 'clean'])
        for file in os.listdir(audio_dir):
            if file.lower().endswith(('.mp3')):
                path = os.path.join(audio_dir, file)
                avg_rms, peak_amp, snr_db = get_audio_metrics(path)
                voice_ratio = voiced_percentage(path)
                if avg_rms >= rms_min and peak_amp < amp_max and snr_db > snr_min and voice_ratio >= voice_ratio_min:
                    clean = True
                    print("Saved clean audio: ", path)
                    shutil.copy(path, CLEAN_DIR)
                else:
                    clean = False
                    print("Saved noisy audio: ", path)
                    shutil.copy(path, NOISY_DIR)
                writer.writerow([file, avg_rms, peak_amp, snr_db, voice_ratio, clean])


def main():
    process_audio(DEV_DIR)
    process_audio(TRAIN_DIR)


if __name__ == "__main__":
    main()

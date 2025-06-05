import os
import librosa
import numpy as np

RAW_AUDIO_DIR = "/Users/setongerrity/Desktop/Mozilla/common-voice-asr/Common-voice-asr/data/raw/mini_cv"
OUTPUT_DIR = "/Users/setongerrity/Desktop/Mozilla/common-voice-asr/Common-voice-asr/data/processed/mini_cv"

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80

for filename in os.listdir(RAW_AUDIO_DIR):
    if filename.endswith(".mp3"):
        input_path = os.path.join(RAW_AUDIO_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".mp3", ".npy"))

        try:
            y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0, top_db=80)
            np.save(output_path, log_mel_spec)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


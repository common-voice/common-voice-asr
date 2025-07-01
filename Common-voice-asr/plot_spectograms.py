import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

mini = False
if mini:
    PROCESSED_DIR = "/Users/setongerrity/Desktop/Mozilla/common-voice-asr/Common-voice-asr/data/processed/mini_cv"
else:
    PROCESSED_DIR = "/Users/setongerrity/Desktop/Mozilla/common-voice-asr/Common-voice-asr/data/processed/full_mini_cv"
NUM_PLOT = 5

npy_files = [file for file in os.listdir(PROCESSED_DIR) if file.endswith('.npy')][:NUM_PLOT]

for npy_file in npy_files:
    spect_path = os.path.join(PROCESSED_DIR, npy_file)
    S_dB = np.load(spect_path)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=f"Mel-frequency spectrogram - {npy_file}")
    plt.tight_layout()
    plt.show()

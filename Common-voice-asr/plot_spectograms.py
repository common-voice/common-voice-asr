import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))
mini = False
full_mini = False
if mini:
    PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed/mini_cv")
elif full_mini:
    PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed/full_mini_cv")
else:
    PROCESSED_DIR = os.path.join(BASE_DIR, "corpus_data/processed/best_train_cv")
NUM_PLOT = 5

npy_files = [file for file in os.listdir(PROCESSED_DIR) if file.endswith('.npy')][:NUM_PLOT]


def main():
    for npy_file in npy_files:
        spect_path = os.path.join(PROCESSED_DIR, npy_file)
        S_dB = np.load(spect_path)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title=f"Mel-frequency spectrogram - {npy_file}")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{npy_file}.png")


if __name__ == "__main__":
    main()
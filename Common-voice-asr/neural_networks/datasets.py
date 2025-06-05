from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from neural_networks.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MiniCVDataset(Dataset):
    def __init__(self, manifest_path, spect_dir, transform = None):
        self.manifest = pd.read_csv(manifest_path)
        self.spect_dir = spect_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        spect_filename = row['filename'].replace('.mp3', '.npy')
        spect_path = os.path.join(self.spect_dir, spect_filename)
        spect = np.load(spect_path) 

        spect_tensor = torch.tensor(spect, dtype=torch.float32).unsqueeze(0) 

        if self.transform:
            spect_tensor = self.transform(spect_tensor)

        label = int(row['label'])
        return spect_tensor, label


import torch.nn.functional as F

# Collate_fn implementation for RNN
def rnn_collate_fn(batch):
    return 0


from torch.nn.utils.rnn import pad_sequence

#custom implementation for variable-length spects
def collate_fn(batch):
    spects, transcripts = zip(*batch)

    max_T = max(s.shape[-1] for s in spects)

    padded_spects = []
    for s in spects:
        pad_len = max_T - s.shape[-1]

        padded_s = F.pad(s, pad=(0, pad_len), mode='constant', value=0)
        padded_spects.append(padded_s)
    
    batch_tensor = torch.stack(padded_spects)
    label_tensor = torch.tensor(transcripts, dtype=torch.long)
    return batch_tensor, label_tensor

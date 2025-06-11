from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import torch

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

        spec = torch.tensor(spect, dtype=torch.float32).unsqueeze(0) 
        if self.transform:
            spec = self.transform(spec)

        transcript = row['transcript'].upper()
        char_id_dict = char_to_id()
        transcript_ids = [char_id_dict[char] for char in transcript]
        transcript_ids = torch.tensor(transcript_ids, dtype=torch.long)

        input_lengths = torch.tensor([spec.shape[-1]], dtype=torch.int32)
        target_lengths = torch.tensor([len(transcript_ids)], dtype=torch.int32)

        return spec, transcript_ids, input_lengths, target_lengths
    
import string

def char_to_id():
    dict = {'<blank>' : 0, ' ' : 1}
    for id, char in enumerate(string.ascii_uppercase, start=2):
        dict[char] = id
    return dict

import torch.nn.functional as F

# Collate_fn implementation for RNN
def rnn_collate_fn(batch):
    spects, transcripts, input_lengths, target_lengths = zip(*batch)

    max_T = max(s.shape[2] for s in spects)
    padded_spects = []

    for spect in spects:
        pad_len = max_T - spect.shape[2]
        if pad_len > 0:
            spect = F.pad(spect, (0, pad_len))
        padded_spects.append(spect)
    
    batch_tensor = torch.stack(padded_spects)
    batch_tensor = batch_tensor.squeeze(1).permute(0, 2, 1)

    concat_transcripts = torch.cat(transcripts)
    
    input_lengths = torch.tensor([s.shape(2) for s in spects], dtype=torch.int32)
    target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.int32)

    return batch_tensor, concat_transcripts, input_lengths, target_lengths

#custom implementation for variable-length spects
def collate_fn(batch):
    spects, transcripts, input_lengths, target_lengths = zip(*batch)

    max_T = max(s.shape[2] for s in spects)
    padded_spects = []

    for spect in spects:
        pad_len = max_T - spect.shape[2]
        if pad_len > 0:
            spect = torch.nn.functional.pad(spect, (0, pad_len))
        padded_spects.append(spect)
    
    batch_tensor = torch.stack(padded_spects)

    concat_transcripts = torch.cat(transcripts)
    
    input_lengths = torch.tensor([s.shape(2) for s in spects], dtype=torch.int32)
    target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.int32)

    return batch_tensor, concat_transcripts, input_lengths, target_lengths

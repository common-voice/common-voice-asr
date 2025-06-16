import torch
import os
import string
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class CTC_MiniCVDataset(Dataset):
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
        transcript = normalize(transcript)
        char_id_dict = char_to_id()
        transcript_ids = [char_id_dict[char] for char in transcript]
        transcript_ids = torch.tensor(transcript_ids, dtype=torch.long)

        input_lengths = torch.tensor([spec.shape[2]], dtype=torch.long)
        target_lengths = torch.tensor([len(transcript_ids)], dtype=torch.long)

        return spec, transcript_ids, input_lengths, target_lengths

tokens = ['<blank>', ' '] + list(string.ascii_uppercase + '.' + '!' + '?' + '-' + ',' + '"' + "'" + ':')
char2idx = {c: i for i, c in enumerate(tokens)}
def tokenize(text):
    return [char2idx[c] for c in text.upper() if c in char2idx]
def char_to_id():
    return {char: idx for idx, char in enumerate(tokens)}
def normalize(text):
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    return text

class CEL_MiniCVDataset(Dataset):
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

# Collate_fn implementation for RNN w CTCLoss
def ctc_rnn_collate_fn(batch):
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
    
    input_lengths = torch.tensor([s.shape[2] for s in spects], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)

    return batch_tensor, concat_transcripts, input_lengths, target_lengths

#custom implementation for variable-length spects CNN Model w CTCLoss
def ctc_collate_fn(batch):
    filtered_batch = [
        (spect, transcript, input_lengths, target_lengths)
        for spect, transcript, input_lengths, target_lengths in batch
        if transcript.shape[0] <= spect.shape[-1]
    ]

    if len(filtered_batch) == 0:
        raise ValueError("All samples in batch were too long for their input lengths")
    spects, transcripts, input_lengths, target_lengths = zip(*filtered_batch)

    max_T = max(s.shape[2] for s in spects)
    padded_spects = []

    for spect in spects:
        mean = spect.mean(dim=2, keepdim=True)
        std = spect.std(dim=2, keepdim=True)
        spect = (spect - mean) / (std + 1e-5)
        pad_len = max_T - spect.shape[2]
        if pad_len > 0:
            spect = torch.nn.functional.pad(spect, (0, pad_len))
        padded_spects.append(spect)
    
    batch_tensor = torch.stack(padded_spects)

    concat_transcripts = torch.cat(transcripts)
    
    input_lengths = torch.tensor([s.shape[2] for s in spects], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)

    return batch_tensor, concat_transcripts, input_lengths, target_lengths

def cel_collate_fn(batch):
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

def cel_rnn_collate_fn(batch):
    spects, transcripts = zip(*batch)

    max_T = max(s.shape[-1] for s in spects)
    padded_spects = []
    for s in spects:
        pad_len = max_T - s.shape[-1]

        padded_s = F.pad(s, pad=(0, pad_len), mode='constant', value=0)
        padded_spects.append(padded_s)
    batch_tensor = torch.stack(padded_spects)
    if batch_tensor.dim() == 4:
        batch_tensor = batch_tensor.squeeze(1)
    batch_tensor = batch_tensor.permute(0, 2, 1)
    label_tensor = torch.tensor(transcripts, dtype=torch.long)
    return batch_tensor, label_tensor

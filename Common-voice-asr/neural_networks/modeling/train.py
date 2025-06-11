# to run: python -m neural_networks.modeling.train (via Common-voice-asr)
# add --check-data
# add --model-type cnn --epochs 3

from loguru import logger
import torch
import typer
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from dotenv import load_dotenv
from rich.progress import Progress
from torch.utils.data import random_split, DataLoader
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from neural_networks.wrap_encoder import WrapEncoder
from neural_networks.datasets import MiniCVDataset, collate_fn, rnn_collate_fn
from neural_networks.config import MODELS_DIR, PROCESSED_DATA_DIR

from neural_networks.cnn_encoder import CNNEncoder
from neural_networks.rnn_encoder import RNNEncoder

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))

import argparse

def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-data', action='store_true', help='Check if data loads correctly')
    parser.add_argument('--model_type', choices=['cnn', 'rnn'], required=True, help='Specify which model to use')
    parser.add_argument('--epochs', type=int, required=True, help= "Number of epochs to train")
    return parser.parse_args()

app = typer.Typer()

def train(model, train_loader, optimizer, criterion, device, epoch, log_interval):
    model.train()
    losses = []

    with Progress() as progress:
        pbar = progress.add_task(f"[green]Epoch {epoch}...", total=len(train_loader))

        for batch_idx, batch in enumerate(train_loader):
            spects, targets = batch[:2]
            spects = spects.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(spects)
        
            if outputs.dim() == 3:
                outputs = outputs.view(-1, outputs.shape[-1])
                targets = targets.view(-1)

            loss = criterion(outputs, targets) # (log_probs, targets, input_lengths, target_lengths) for CTCLoss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(spects)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                
            progress.advance(pbar)
    return sum(losses)/len(losses)

def validate(model,val_loader, criterion, device):
    model.eval()
    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for spects, targets in val_loader:
            spects = spects.to(device)
            targets = targets.to(device)
            outputs = model(spects)

            if outputs.dim() == 3:
                outputs = outputs.view(-1, outputs.shape[-1])
                targets = targets.view(-1)

            loss = criterion(outputs,targets) 
            losses.append(loss.item())

            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = sum(losses) / len(losses)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy



@app.command()
def main(check_data: bool = False, model_type: str = "cnn", epochs: int = 3):
    manifest_path = BASE_DIR / "data" / "manifest.csv"
    spect_dir = BASE_DIR / "data" / "processed" / "mini_cv"

    log_dir = BASE_DIR / "neural_networks" / "runs" / f"week3_{model_type}"

    df = pd.read_csv(manifest_path)
    if 'label' not in df.columns:
        print("Adding dummy label column to manifest.csv")
        df['label'] = [i % 10 for i in range(len(df))]
        df.to_csv(manifest_path, index = False)

    dataset = MiniCVDataset(manifest_path, spect_dir)
    total_len = len(dataset)
    train_len = int(0.6 * total_len) # 60% for training, 20% for validation, 20% for testing later
    val_len = int(0.2 * total_len)
    test_len = total_len - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    if model_type == "cnn":
        model = CNNEncoder()
        collate = collate_fn
    elif model_type == "rnn":
        model = RNNEncoder()
        collate = rnn_collate_fn
    else:
        raise ValueError("Unknown model type: {model_type}")
    
    train_loader = DataLoader(train_set, batch_size=4, collate_fn=collate, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, collate_fn=collate)

    if check_data:
        for batch in train_loader:
            spects, transcripts = batch[:2]

            print("Spectrogram shape: ", spects.shape)
            print("Transcripts: ", transcripts)
            break
        return
    
    model = WrapEncoder(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    writer = SummaryWriter()

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, log_interval=20)
        val_loss, val_acc = validate(model,val_loader, criterion, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        
        print(f"\n Epoch {epoch} completed")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val accuracy: {val_acc:.4f}")
    writer.flush()
    writer.close()

if __name__ == "__main__":
    args = parse_command_args()
    main(check_data=args.check_data, model_type=args.model_type, epochs=args.epochs)
    app()

# to run: python -m neural_networks.modeling.train (via Common-voice-asr)
# add --check-data
# add --model_type cnn --epochs 3
# python -m neural_networks.modeling.train --full_mini --model_type rnn --epochs 5
# python -m neural_networks.modeling.train --full_mini --model_type cnn --epochs 5
# python -m neural_networks.modeling.train --full_mini --model_type rnn --epochs 5 --lr 1e-3 --logdir runs/week4_ctc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import os
import torchaudio
import wandb
import tempfile
import shutil
import argparse
import string
import time
import random
import torch.amp
from dotenv import load_dotenv
from rich.progress import Progress
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from neural_networks.wrap_encoder import WrapEncoder
from neural_networks.datasets import CEL_MiniCVDataset, cel_collate_fn, cel_rnn_collate_fn
from neural_networks.datasets import CTC_MiniCVDataset, ctc_collate_fn, ctc_rnn_collate_fn
from neural_networks.cnn_encoder import CTC_CNNEncoder, CEL_CNNEncoder
from neural_networks.rnn_encoder import CTC_RNNEncoder, CEL_RNNEncoder
from neural_networks.greedy_ctc_decoder import GreedyCTCDecoder
from neural_networks.beam_search_decoder import beam_search_decoder

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))

tokens = ['<blank>', '|', ' '] + list(string.ascii_uppercase + '.' + '!' + '?' + '-' + ',' + '"' + "'" + ':')


def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-data', action='store_true', help='Check if data loads correctly')
    parser.add_argument('--full_mini', action='store_true', help="Load larger data split from CV Train")
    parser.add_argument('--corpus', action='store_true', help="Load corpus train & dev large datasets")
    parser.add_argument('--greedy', action='store_true', help="Use Greedy CTC Decoder")
    parser.add_argument('--model_type', choices=['cnn', 'rnn'], required=True, help='Specify which model to use')
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--logdir', type=str, required=True, help="Folder to write trains to")
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for model')
    parser.add_argument('--test-sweep', action='store_true', help="Testing sweep with small dataset")
    parser.add_argument('--lm-weight', type=float, default=3.23, help="Learning model weight for beam search decoder")
    parser.add_argument('--word-score', type=float, default=-0.26, help="Word score for beam search decoder")
    parser.add_argument('--sample_size', type=int, default=0, help="Sample size for dataset")
    return parser.parse_args()


def cel_train(model, train_loader, optimizer, criterion, device, epoch, log_interval):
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

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(spects)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            progress.advance(pbar)
    return sum(losses)/len(losses)


def cel_validate(model, val_loader, criterion, device):
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

            loss = criterion(outputs, targets)
            losses.append(loss.item())

            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    avg_loss = sum(losses) / len(losses)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def get_references(target_lengths, targets):
    references = []
    start = 0
    for t_len in target_lengths:
        end = start + t_len.item()
        indices = targets[start:end]
        ref = "".join([tokens[i] for i in indices]).replace("|", " ").strip().split()
        references.append(ref)
        start = end
    return references


def get_train_wer(references, hypotheses, count, total_wer, batch_idx):
    for ref, hyp in zip(references, hypotheses):
        if batch_idx == 0:
            print(f"Reference: {ref}")
            print(f"Hypotheses: {hyp}")
        total_wer += min(torchaudio.functional.edit_distance(ref, hyp) / max(len(ref), 1), 1)
        count += 1
    return total_wer, count


def get_val_err(references, hypotheses, count, total_wer, total_cer, batch_idx):
    for ref, hyp in zip(references, hypotheses):
        if batch_idx == 0:
            print(f"Reference: {ref}")
            print(f"Hypotheses: {hyp}")
        total_wer += min(torchaudio.functional.edit_distance(ref, hyp) / max(len(ref), 1), 1)
        total_cer += min(torchaudio.functional.edit_distance(list(ref), list(hyp)) / max(len(ref), 1), 1)
        count += 1
    return total_wer, total_cer, count
        
def ctc_train(model, train_loader, optimizer, criterion, device, epoch, log_interval, decoder, corpus, sample_size):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    losses = []
    count = 0
    total_wer = 0.0
    
    total_batches = len(train_loader)
    log_percent = 5
    log_interval_batches = max(1, int((log_percent / 100) * total_batches))
    
    scaler = torch.amp.GradScaler('cuda')

    with Progress() as progress:
        pbar = progress.add_task(f"[green]Epoch {epoch}...", total=total_batches)

        for batch_idx, (spects, targets, input_lengths, target_lengths) in enumerate(train_loader):
            spects = spects.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(spects)
                outputs = outputs.to(torch.float32)
                outputs = torch.clamp(outputs, min=-50, max=50)

                log_probs = F.log_softmax(outputs, dim=2)
                log_probs = log_probs.transpose(0, 1)
                max_output_len = log_probs.size(0)
                input_lengths = torch.clamp(input_lengths, max=max_output_len)

                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            losses.append(loss.item())
            torch.cuda.synchronize()
            if corpus and sample_size == 0:
                if batch_idx % log_interval_batches == 0:
                    hypotheses = decoder(log_probs.transpose(0, 1))
                    references = get_references(target_lengths, targets)
                    total_wer, count = get_train_wer(references, hypotheses, count, total_wer, batch_idx)

            else:
                hypotheses = decoder(log_probs.transpose(0, 1))
                references = get_references(target_lengths, targets)
                total_wer, count = get_train_wer(references, hypotheses, count, total_wer, batch_idx)

            if batch_idx % log_interval_batches == 0:
                percent = 100. * batch_idx / total_batches
                print(f"Train Epoch: {epoch} [{batch_idx * len(spects)}/{len(train_loader.dataset)} "
                      f"({percent:.0f}%)]\tLoss: {loss.item():.6f}")
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            progress.advance(pbar)

    avg_loss = sum(losses)/len(losses)
    avg_wer = total_wer / count if count > 0 else 1.0
    return avg_loss, avg_wer


def ctc_validate(model, val_loader, criterion, device, decoder, epoch, corpus, sample_size):
    model.eval()
    losses = []
    count = 0
    total_wer = 0.0
    total_cer = 0.0
    
    scaler = torch.amp.GradScaler('cuda')
    
    total_batches = len(val_loader)
    log_percent = 5
    log_interval_batches = max(1, int((log_percent / 100) * total_batches))
    
    with Progress() as progress:
        pbar = progress.add_task(f"[green]Epoch {epoch}...", total=total_batches)
        with torch.no_grad():
            for batch_idx, (spects, targets, input_lengths, target_lengths) in enumerate(val_loader):
                with torch.amp.autocast('cuda'):
                    spects = spects.to(device)
                    targets = targets.to(device)
                    input_lengths = input_lengths.to(device)
                    target_lengths = target_lengths.to(device)

                    outputs = model(spects)
                    log_probs = F.log_softmax(outputs, dim=2)
                    log_probs = log_probs.permute(1, 0, 2)
                    max_output_len = log_probs.size(0)
                    input_lengths = torch.clamp(input_lengths, max=max_output_len)

                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                losses.append(loss.item())
                torch.cuda.synchronize()
                if corpus and sample_size == 0:
                    if batch_idx % 8 == 0:
                        hypotheses = decoder(log_probs.transpose(0, 1))
                        references = get_references(target_lengths, targets)
                        total_wer, total_cer, count = get_val_err(references, hypotheses, count, total_wer, total_cer, batch_idx)
                else:
                    hypotheses = decoder(log_probs.transpose(0, 1))
                    references = get_references(target_lengths, targets)
                    total_wer, total_cer, count = get_val_err(references, hypotheses, count, total_wer, total_cer, batch_idx)
                if batch_idx % log_interval_batches == 0:
                    percent = 100. * batch_idx / total_batches
                    print(f"Validation Epoch: {epoch} [{batch_idx * len(spects)}/{len(val_loader.dataset)} ({percent:.0f}%)]")
                progress.advance(pbar)

    avg_loss = sum(losses) / len(losses)
    avg_wer = total_wer / count if count > 0 else 1.0
    avg_cer = total_cer / count

    return avg_loss, avg_wer, avg_cer


def setup_encoder_and_data(mini, manifest_path, spect_dir, model_type, hidden_dim):
    if mini:
        df = pd.read_csv(manifest_path)
        if 'label' not in df.columns:
            print("Adding dummy label column to manifest.csv")
            df['label'] = [i % 10 for i in range(len(df))]
            df.to_csv(manifest_path, index=False)
        dataset = CEL_MiniCVDataset(manifest_path, spect_dir)
        num_classes = 10
        apply = True
        if model_type == "cnn":
            model = CEL_CNNEncoder()
            collate = cel_collate_fn
        elif model_type == "rnn":
            model = CEL_RNNEncoder()
            collate = cel_rnn_collate_fn
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        df = pd.read_csv(manifest_path)
        dataset = CTC_MiniCVDataset(manifest_path, spect_dir)
        num_classes = len(tokens)
        if model_type == "cnn":
            model = CTC_CNNEncoder(hidden_dim=hidden_dim)
            collate = ctc_collate_fn
            apply = False
        elif model_type == "rnn":
            model = CTC_RNNEncoder(hidden_size=hidden_dim)
            collate = ctc_rnn_collate_fn
            apply = True
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return df, dataset, num_classes, apply, model, collate


def test_sweep(df, spect_dir, batch_size, collate, model, optimizer, criterion, device, epochs, log_interval):
    test_df = df.head(5)
    test_spect_dir = tempfile.mkdtemp()
    for file in test_df["filename"]:
        spect_file = file.replace(".mp3", ".npy")
        full_path = os.path.join(spect_dir, spect_file)
        dest_path = os.path.join(test_spect_dir, spect_file)
        shutil.copy(full_path, dest_path)
    test_manifest = os.path.join(test_spect_dir, "test_manifest.csv")
    test_df.to_csv(test_manifest, index=False)
    test_set = CTC_MiniCVDataset(test_manifest, test_spect_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate)
    test_loss, test_wer = ctc_train(model, test_loader, optimizer, criterion, device, epochs, log_interval)
    assert wandb.run, "W&B is not running"
    wandb.log({'epoch': epochs, 'test/loss': test_loss, 'test/wer': test_wer})


def run_ctc(model, epochs, train_loader, val_loader, optimizer, criterion, device, decoder, log_interval, writer, corpus, 
            sample_size):
    min_val_wer = 1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_wer = ctc_train(model, train_loader, optimizer, criterion, device, epoch, log_interval, decoder, 
                                          corpus, sample_size)
        val_loss, val_wer, val_cer = ctc_validate(model, val_loader, criterion, device, decoder, epoch, corpus, sample_size)
        writer.add_scalar('train/ctc_loss', train_loss, epoch)
        writer.add_scalar('train/wer', train_wer, epoch)
        writer.add_scalar('val/ctc_loss', val_loss, epoch)
        writer.add_scalar('val/wer', val_wer, epoch)
        writer.add_scalar('val/cer', val_cer, epoch)
        if wandb.run:
            wandb.log({
                'epoch': epoch,
                'train/ctc_loss': train_loss,
                'train/wer': train_wer,
                'val/ctc_loss': val_loss,
                'val/wer': val_wer,
                'val/cer': val_cer,
                })

        min_val_wer = min(min_val_wer, val_wer)
        print(f"\n Epoch {epoch} completed")
        print(f"Train CTC loss: {train_loss:.4f}")
        print(f"Train WER: {train_wer:.4f}")
        print(f"Val CTC loss: {val_loss:.4f}")
        print(f"Val WER: {val_wer:.4f}")
        print(f"Val CER: {val_cer:.4f}")

    return min_val_wer


def run_cel(epochs, model, train_loader, val_loader, optimizer, criterion, device, log_interval, writer):
    for epoch in range(1, epochs + 1):
        train_loss = cel_train(model, train_loader, optimizer, criterion, device, epoch, log_interval)
        val_loss, val_acc = cel_validate(model, val_loader, criterion, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"\n Epoch {epoch} completed")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val accuracy: {val_acc:.4f}")


def main(check_data: bool = False, full_mini: bool = False, corpus: bool = False, greedy: bool = False, 
         model_type: str = "cnn", epochs: int = 3, lr: float = 1e-3, logdir: str = 'runs/week4_ctc', 
         batch_size: int = 4, hidden_dim: int = 64, test_sweep: bool = False, lm_weight: float = 3.23, 
         word_score: float = -0.26, sample_size: int = 0):

    log_dir = os.path.join("neural_networks", logdir)
    log_path = os.path.join(BASE_DIR, log_dir)
    
    mini = False
    if full_mini:
        manifest_path = os.path.join(BASE_DIR, "data/cleaned_manifest.csv")
        spect_dir = os.path.join(BASE_DIR, "data/processed/full_mini_cv")
        df, dataset, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                 manifest_path, spect_dir, model_type, hidden_dim)
    elif corpus:
        train_manifest_path = os.path.join(BASE_DIR, "corpus_data/cleaned_train.csv")
        train_spect_dir = os.path.join(BASE_DIR, "corpus_data/processed/train_cv")
        dev_manifest_path = os.path.join(BASE_DIR, "corpus_data/cleaned_dev.csv")
        dev_spect_dir = os.path.join(BASE_DIR, "corpus_data/processed/dev_cv")
        train_df, train_set, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                         train_manifest_path, train_spect_dir, 
                                                                                         model_type, hidden_dim)
        val_df, val_set, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                     dev_manifest_path, dev_spect_dir, 
                                                                                     model_type, hidden_dim)
    else:
        mini = True
        manifest_path = BASE_DIR / "data" / "manifest.csv"
        spect_dir = BASE_DIR / "data" / "processed" / "mini_cv"
        df, dataset, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                 manifest_path, spect_dir, model_type, hidden_dim)
    if mini or full_mini:
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
    
    if not sample_size == 0:
        if corpus:
            total_train = len(train_set)
            total_val = len(val_set)
            data_len = total_train + total_val
        else: 
            data_len = len(dataset)
        if sample_size > data_len:
            print(f"Sample size cannot be greater than {data_len}")
            return
        train_ratio = total_train / data_len
        val_ratio = total_val / data_len
        train_sample = round(sample_size * train_ratio)
        val_sample = sample_size - train_sample
        train_indices = random.sample(range(total_train), k=train_sample)
        val_indices = random.sample(range(total_val), k=val_sample) if total_val > 0 else []
        train_set = [train_set[i] for i in train_indices]
        val_set = [val_set[i] for i in val_indices]
        

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate, num_workers=4, pin_memory=True, prefetch_factor=2)

    log_interval = 20
    if check_data:
        for batch in train_loader:
            spects, transcripts = batch[:2]

            print("Spectrogram shape: ", spects.shape)
            print("Transcripts: ", transcripts)
            break
        return

    model = WrapEncoder(model, num_classes, apply)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter(log_dir=log_path)

    if full_mini or corpus:
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        greedy = False
        if greedy:
            decoder = GreedyCTCDecoder(tokens)
        else:
            decoder = beam_search_decoder(tokens, lm_weight, word_score)
        if test_sweep:
            test_sweep(df, spect_dir, batch_size, collate, model, optimizer, criterion, device, epochs, log_interval)
            return
        val_wer = run_ctc(model, epochs, train_loader, val_loader, optimizer, criterion, device, decoder, log_interval, writer,
                          corpus, sample_size)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        run_cel(epochs, model, train_loader, val_loader, optimizer, criterion, device, log_interval, writer)

    save_best = False
    if save_best:
        os.makedirs(logdir, exist_ok=True)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'config': {
                    'model_type': model_type, 'hidden_dim': hidden_dim, 'lr': lr,
                    'batch_size': batch_size, 'epochs': epochs,
                }
            },
            os.path.join(log_path, "best_rnn.pth")
        )

    if wandb.run:
        wandb.summary['final_val_wer'] = val_wer

    writer.flush()
    writer.close()


if __name__ == "__main__":
    args = parse_command_args()
    main(args.check_data, args.full_mini, args.corpus, args.greedy, args.model_type, args.epochs, args.lr,
         args.logdir, args.batch_size, args.hidden_dim, args.test_sweep,
         args.lm_weight, args.word_score, args.sample_size)

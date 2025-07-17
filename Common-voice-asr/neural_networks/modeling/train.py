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

from neural_networks.model_A.wrap_encoder import WrapEncoder
from neural_networks.datasets import CEL_MiniCVDataset, cel_collate_fn, cel_rnn_collate_fn
from neural_networks.datasets import CTC_MiniCVDataset, ctc_collate_fn, ctc_rnn_collate_fn, transformer_collate_fn
from neural_networks.model_A.cnn_encoder import CTC_CNNEncoder, CEL_CNNEncoder
from neural_networks.model_A.rnn_encoder import CTC_RNNEncoder, CEL_RNNEncoder
from neural_networks.model_B.hybrid_transformer import HybridTransformer
from neural_networks.greedy_ctc_decoder import GreedyCTCDecoder
from neural_networks.beam_search_decoder import beam_search_decoder

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))

tokens = ['<blank>', '|'] + list(string.ascii_uppercase) + [' ', "'", '-']
N_MELS = 80


def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-data', action='store_true', help='Check if data loads correctly')
    parser.add_argument('--full_mini', action='store_true', help="Load larger data split from CV Train")
    parser.add_argument('--corpus', action='store_true', help="Load corpus train & dev large datasets")
    parser.add_argument('--greedy', action='store_true', help="Use Greedy CTC Decoder")
    parser.add_argument('--model_type', choices=['cnn', 'rnn', 'transformer'], required=True, help='Specify which model to use')
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--logdir', type=str, required=True, help="Folder to write trains to")
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for model')
    parser.add_argument('--test-sweep', action='store_true', help="Testing sweep with small dataset")
    parser.add_argument('--lm-weight', type=float, default=3.23, help="Learning model weight for beam search decoder")
    parser.add_argument('--word-score', type=float, default=-0.26, help="Word score for beam search decoder")
    parser.add_argument('--sample_size', type=int, default=0, help="Specifying sample size from the dataset")
    parser.add_argument('--d_model', type=int, default=512, help="Number of expected features in the encoder/decoder inputs")
    parser.add_argument('--nhead', type=int, default=8, help="Number of heads in the multiheadattention models")
    parser.add_argument('--dim_feedforward', type=int, default=2048, help="Dimension of the feedforward network model")
    parser.add_argument('--nlayers', type=int, default=6, help="Number of layers in the decoder")
    parser.add_argument('--lstm_hidden', type=int, default=256, help="Number of hidden dimensions for LSTM layer of transformer model")
    parser.add_argument('--lstm_layers', type=int, default=1, help="Number of lstm layers in model")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout value for transformer model")
    parser.add_argument('--sample_spect_folder', type=str, default=None, help='Train & validate with a mel-spectogram sweep')
    parser.add_argument('--debug_sample', action='store_true', help="Deugging with single training loop on one sample")
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


def entropy_loss(log_probs):
    probs = log_probs.exp()
    entropy = -torch.mean(torch.sum(probs * log_probs, dim=-1))
    return entropy.mean()

        
def ctc_train(model, train_loader, optimizer, criterion, device, epoch, log_interval, decoder, corpus, sample_size):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    losses = []
    count = 0
    total_wer = 0.0
    seen_samples = 0
    
    total_batches = len(train_loader)
    log_percent = 5
    log_interval_batches = max(1, int((log_percent / 100) * total_batches))
    
    scaler = torch.amp.GradScaler('cuda')

    with Progress() as progress:
        total_samples = len(train_loader.dataset)
        pbar = progress.add_task(f"[green]Epoch {epoch}...", total=total_samples)
        for batch_idx, (spects, targets, input_lengths, target_lengths) in enumerate(train_loader):
            batch_size = spects.size(0)
            seen_samples += batch_size
            spects = spects.to(device)
            targets = targets.to(device)
            if torch.isnan(targets).any():
                    print("NaN targets detected!")
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            print("Input length:", input_lengths.tolist())
            print("Target length:", target_lengths.tolist())
            # print("Target tokens:", targets.tolist())
            for i_len, t_len in zip(input_lengths.tolist(), target_lengths.tolist()):
                assert i_len >= t_len, f"Target too long! Input len: {i_len}, Target len: {t_len}"

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(spects)
                outputs = outputs.clone()
                print("Logits mean/std:", outputs.mean().item(), outputs.std().item())
                if torch.isnan(outputs).any():
                    print("NaN outputs detected!")
                # outputs = torch.clamp(outputs, min=-50, max=50)
                # print("Raw outputs mean/std:", outputs.mean().item(), outputs.std().item())
                    
                log_probs = F.log_softmax(outputs, dim=2)
                if log_probs.shape[0] == input_lengths.shape[0]:
                    log_probs = log_probs.transpose(0, 1)
                # print("DEBUG: raw log_probs: ", log_probs)
                print("log_probs variance (time dim):", log_probs.var(dim=0).mean().item())
                probs = log_probs.exp()
                blank_probs = probs[:, :, 0]  # assuming index 0 is blank
                print("Mean blank prob:", blank_probs.mean().item())
                decoded_indices = log_probs.argmax(dim=-1)  # [T, B]
                print(decoded_indices.T[0])
                """
                # print("Predicted token indices:", log_probs[:, 0].tolist())
                print("Target tensor max:", targets.max().item())
                print("Num classes:", log_probs.shape[2])
                print("Sample log prob distribution:", F.softmax(outputs, dim=2)[0,0,:])
                
                print("Log prob sample max:", log_probs[0, 0].max().item())
                print("Log prob sample min:", log_probs[0, 0].min().item())
                """
                
                T, B, V = log_probs.shape
                """
                print("Spectrogram mean/std (batch):", spects.mean().item(), spects.std().item())
                print(f"[DEBUG] input_lengths shape: {input_lengths.shape}, values: {input_lengths}")
                print(f"[DEBUG] target_lengths shape: {target_lengths.shape}, values: {target_lengths}")
                print(f"[DEBUG] targets shape: {targets.shape}")
                print(f"[DEBUG] outputs shape: {outputs.shape}")      # (B, T, V)
                print(f"[DEBUG] log_probs shape: {log_probs.shape}")  # (T, B, V)
                print(f"[DEBUG] batch size (from log_probs): {B}")
                print(f"[DEBUG] input_lengths size: {input_lengths.size(0)}")
                # assert input_lengths.size(0) == B, "[ERROR] input_lengths must match batch size!"
                """

                max_output_len = log_probs.size(0)
                input_lengths = torch.clamp(input_lengths, max=max_output_len)
                # print("[DEBUG] input_lengths:", input_lengths)
                # print("[DEBUG] target_lengths:", target_lengths)

                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                alpha = min(2.0, 0.2 * epoch)
                # loss = loss + (alpha * entropy_loss(log_probs))
                
            losses.append(loss.item())
            torch.cuda.synchronize()
            if corpus:
                if batch_idx % log_interval_batches == 0:
                    lp_bt = log_probs.permute(1, 0, 2).contiguous()  
                    hypotheses = decoder(lp_bt)
                    references = get_references(target_lengths, targets)
                    print("DEBUG")
                    print(type(hypotheses), hypotheses)
                    print(type(references), references)
                    total_wer, count = get_train_wer(references, hypotheses, count, total_wer, batch_idx)

            else:
                log_probs_bt = log_probs.transpose(0, 1).contiguous()
                hypotheses = decoder(log_probs_bt)
                references = get_references(target_lengths, targets)
                print("DEBUG")
                print(type(hypotheses), hypotheses)
                print(type(references), references)
                total_wer, count = get_train_wer(references, hypotheses, count, total_wer, batch_idx)

            if batch_idx % log_interval_batches == 0:
                print(f"Train Epoch: {epoch} Loss: {loss.item():.6f}")
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            progress.advance(pbar, batch_size)
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"{name} grad is None")
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            # print(f"[DEBUG] Grad norm: {total_norm ** 0.5}")
            # print(f"[DEBUG] Step loss: {loss.item():.4f}")

    avg_loss = sum(losses)/len(losses)
    avg_wer = total_wer / count if count > 0 else 1.0
    return avg_loss, avg_wer


def ctc_validate(model, val_loader, criterion, device, decoder, epoch, corpus, sample_size):
    model.eval()
    losses = []
    count = 0
    total_wer = 0.0
    total_cer = 0.0
    seen_samples = 0
    
    scaler = torch.amp.GradScaler('cuda')
    
    total_batches = len(val_loader)
    log_percent = 5
    log_interval_batches = max(1, int((log_percent / 100) * total_batches))
    
    with Progress() as progress:
        total_samples = len(val_loader.dataset)
        pbar = progress.add_task(f"[green]Epoch {epoch}...", total=total_samples)
        with torch.no_grad():
            for batch_idx, (spects, targets, input_lengths, target_lengths) in enumerate(val_loader):
                batch_size = spects.size(0)
                seen_samples += batch_size
                with torch.amp.autocast('cuda'):
                    spects = spects.to(device)
                    targets = targets.to(device)
                    input_lengths = input_lengths.to(device)
                    target_lengths = target_lengths.to(device)

                    outputs = model(spects)
                    outputs = outputs.clone()
                    log_probs = F.log_softmax(outputs, dim=2)
                    if log_probs.shape[0] == input_lengths.shape[0]:
                        log_probs = log_probs.transpose(0, 1)
                    max_output_len = log_probs.size(0)
                    input_lengths = torch.clamp(input_lengths, max=max_output_len)
                    
                    decoded_indices = log_probs.argmax(dim=-1)
                    # print("Decoded indices:", decoded_indices)
                    T, B, V = log_probs.shape
                    """
                    print(f"[DEBUG] batch size (from log_probs): {B}")
                    print(f"[DEBUG] input_lengths size: {input_lengths.size(0)}")
                    assert input_lengths.size(0) == B, "[ERROR] input_lengths must match batch size!"
                    """
                    
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                    alpha = min(1.0, 0.2 * epoch)
                    # loss = loss + (alpha * entropy_loss(log_probs))
                    
                losses.append(loss.item())
                torch.cuda.synchronize()
                if corpus and sample_size == 0:
                    if batch_idx % 8 == 0:
                        lp_bt = log_probs.permute(1, 0, 2).contiguous()  
                        hypotheses = decoder(lp_bt)
                        references = get_references(target_lengths, targets)
                        total_wer, total_cer, count = get_val_err(references, hypotheses, count, total_wer, total_cer, batch_idx)
                else:
                    lp_bt = log_probs.permute(1, 0, 2).contiguous()  
                    hypotheses = decoder(lp_bt)
                    references = get_references(target_lengths, targets)
                    print("Reference:", references[0])
                    print("Hypothesis:", hypotheses[0])
                    total_wer, total_cer, count = get_val_err(references, hypotheses, count, total_wer, total_cer, batch_idx)
                if batch_idx % log_interval_batches == 0:
                    print(f"Validation Epoch: {epoch}")
                progress.advance(pbar, batch_size)

    avg_loss = sum(losses) / len(losses)
    avg_wer = total_wer / count if count > 0 else 1.0
    avg_cer = total_cer / count

    return avg_loss, avg_wer, avg_cer


def setup_encoder_and_data(mini, manifest_path, spect_dir, model_type, hidden_dim, d_model, nhead, dim_ff, nlayers, lstm_hidden,
                           lstm_layers, dropout, debug_sample):
    apply = False
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
    elif debug_sample:
        df = pd.read_csv(manifest_path)
        dataset = CTC_MiniCVDataset(manifest_path, spect_dir)
        dataset = torch.utils.data.Subset(dataset, [0])
        num_classes = len(tokens)
        if model_type == "cnn":
            model = CTC_CNNEncoder(hidden_dim=hidden_dim)
            collate = ctc_collate_fn
            apply = False
        elif model_type == "rnn":
            model = CTC_RNNEncoder(hidden_size=hidden_dim)
            collate = ctc_rnn_collate_fn
            apply = True
        elif model_type == "transformer":
            model = HybridTransformer(input_dim=N_MELS, vocab_size=len(tokens), d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                      nlayers=nlayers, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=dropout)
            collate = transformer_collate_fn
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
        elif model_type == "transformer":
            model = HybridTransformer(input_dim=N_MELS, vocab_size=len(tokens), d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                      nlayers=nlayers, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=dropout)
            collate = transformer_collate_fn
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
            sample_size, sample_spect_folder = None):
    min_val_wer = 1.0
    best_epoch = 0
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
        if min_val_wer > val_wer:
            min_val_wer = val_wer
            best_epoch = epoch
        print(f"\n Epoch {epoch} completed")
        print(f"Train CTC loss: {train_loss:.4f}")
        print(f"Train WER: {train_wer:.4f}")
        print(f"Val CTC loss: {val_loss:.4f}")
        print(f"Val WER: {val_wer:.4f}")
        print(f"Val CER: {val_cer:.4f}")
    if wandb.run:
        if sample_spect_folder is not None:
            wandb.run.summary['sample_spect'] = sample_spect_folder
        wandb.run.summary['val/wer_min'] = min_val_wer
        wandb.run.summary["val/wer_best_epoch"] = best_epoch


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
        

def print_grad_hook(grad):
    print("Gradients on classifier weight:", grad.norm())



def main(check_data: bool = False, full_mini: bool = False, corpus: bool = False, greedy: bool = False, 
         model_type: str = "cnn", epochs: int = 3, lr: float = 1e-3, logdir: str = 'runs/week4_ctc', 
         batch_size: int = 4, hidden_dim: int = 64, test_sweep: bool = False, lm_weight: float = 3.23, 
         word_score: float = -0.26, sample_size: int = 0, 
         d_model: int = 512, n_head: int = 8, dim_feedforward: int = 2048, nlayers: int = 6, lstm_hidden: int = 256, 
         lstm_layers: int = 1, dropout: float = 0.5, sample_spect_folder: str = None, debug_sample: bool = False):

    log_dir = os.path.join("neural_networks", logdir)
    log_path = os.path.join(BASE_DIR, log_dir)
    
    mini = False
    if full_mini:
        manifest_path = os.path.join(BASE_DIR, "data/cleaned_manifest.csv")
        spect_dir = os.path.join(BASE_DIR, "data/processed/full_mini_cv")
        df, dataset, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                 manifest_path, spect_dir, model_type, hidden_dim,
                                                                                 d_model, n_head, dim_feedforward, nlayers,
                                                                                 lstm_hidden, lstm_layers, dropout, debug_sample)
    elif corpus:
        train_manifest_path = os.path.join(BASE_DIR, "corpus_data/cleaned_train.csv")
        train_spect_dir = os.path.join(BASE_DIR, "corpus_data/processed/train_cv")
        dev_manifest_path = os.path.join(BASE_DIR, "corpus_data/cleaned_dev.csv")
        dev_spect_dir = os.path.join(BASE_DIR, "corpus_data/processed/dev_cv")
        train_df, train_set, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                         train_manifest_path, train_spect_dir, 
                                                                                         model_type, hidden_dim, d_model, n_head,
                                                                                         dim_feedforward, nlayers, lstm_hidden,
                                                                                         lstm_layers, dropout, debug_sample)
        val_df, val_set, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                     dev_manifest_path, dev_spect_dir, 
                                                                                     model_type, hidden_dim, d_model, n_head,
                                                                                     dim_feedforward, nlayers, lstm_hidden,
                                                                                     lstm_layers, dropout, debug_sample)
        print(train_df["transcript"].iloc[0])
    elif sample_spect_folder is not None:
        sample_spect = True
        train_manifest_path = os.path.join(BASE_DIR, "corpus_data/cleaned_train.csv")
        train_spect_dir = os.path.join(BASE_DIR, os.path.join("corpus_data/processed/train_cv/sample", sample_spect_folder))
        dev_manifest_path = os.path.join(BASE_DIR, "corpus_data/cleaned_dev.csv")
        dev_spect_dir = os.path.join(BASE_DIR, os.path.join("corpus_data/processed/dev_cv/sample", sample_spect_folder))
        train_df, train_set, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                         train_manifest_path, train_spect_dir, 
                                                                                         model_type, hidden_dim, d_model, n_head,
                                                                                         dim_feedforward, nlayers, lstm_hidden,
                                                                                         lstm_layers, dropout, debug_sample)
        val_df, val_set, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                     dev_manifest_path, dev_spect_dir, 
                                                                                     model_type, hidden_dim, d_model, n_head,
                                                                                     dim_feedforward, nlayers, lstm_hidden,
                                                                                     lstm_layers, dropout, debug_sample)
    elif debug_sample:
        train_manifest_path = os.path.join(BASE_DIR, "corpus_data/cleaned_train.csv")
        train_spect_dir = os.path.join(BASE_DIR, "corpus_data/processed/train_cv")
        df, dataset, num_classes, apply, model, collate = setup_encoder_and_data(mini, train_manifest_path, train_spect_dir, 
                                                                                 model_type, hidden_dim, d_model, n_head,
                                                                                 dim_feedforward, nlayers, lstm_hidden,
                                                                                 lstm_layers, dropout, debug_sample)
        print("DEBUG MODE ENABLED - One sample only")
        spect, transcript, *_ = dataset[0]
        print("Spectrogram shape:", spect.shape)
        print("Transcript indices:", transcript)
        print("Transcript string:", ''.join([tokens[i] for i in transcript.tolist()]))
        train_set = dataset
        val_set = dataset
    else:
        mini = True
        manifest_path = BASE_DIR / "data" / "manifest.csv"
        spect_dir = BASE_DIR / "data" / "processed" / "mini_cv"
        df, dataset, num_classes, apply, model, collate = setup_encoder_and_data(mini,
                                                                                 manifest_path, spect_dir, model_type, hidden_dim,
                                                                                 d_model, n_head, dim_feedforward, nlayers, 
                                                                                 lstm_hidden, lstm_layers, dropout, debug_sample)
    if debug_sample:
        print("Skipping sample size logic for debugging")
    else:
        if mini or full_mini:
            total_len = len(dataset)
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
            train_sample = round(sample_size * 0.85)
            val_sample = sample_size - train_sample
            if val_sample > total_val:
                print(f"The 15% sample size split for validation, currently {val_sample}, cannot be greater than {total_val}")
                return
            train_indices = random.sample(range(total_train), k=train_sample)
            val_indices = random.sample(range(total_val), k=val_sample) if total_val > 0 else []
            train_set = [train_set[i] for i in train_indices]
            val_set = [val_set[i] for i in val_indices]
        

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=2)
    
    # More robust way to handle different batch formats
    for batch in train_loader:
        spec, label = batch[0], batch[1]  # First two elements are always spec and label
        print(f"Spec shape: {spec.shape}, Label shape: {label.shape}")
        if len(batch) == 4:  # CTC case has additional elements
            input_lengths, target_lengths = batch[2], batch[3]
            print(f"Input lengths: {input_lengths.shape}, Target lengths: {target_lengths.shape}")
        break
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate, num_workers=4, pin_memory=True, prefetch_factor=2)

    log_interval = 20
    if check_data:
        for batch in train_loader:
            spects, transcripts = batch[:2]

            print("Spectrogram shape: ", spects.shape)
            print("Transcripts: ", transcripts)
            break
        return
    if not model_type == "transformer":
        model = WrapEncoder(model, num_classes, apply)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.classifier.weight.register_hook(print_grad_hook)

    writer = SummaryWriter(log_dir=log_path)

    if full_mini or corpus or debug_sample:
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
        run_ctc(model, epochs, train_loader, val_loader, optimizer, criterion, device, decoder, log_interval, writer,
                corpus, sample_size, sample_spect_folder)
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

    writer.flush()
    writer.close()


if __name__ == "__main__":
    args = parse_command_args()
    main(args.check_data, args.full_mini, args.corpus, args.greedy, args.model_type, args.epochs, args.lr,
         args.logdir, args.batch_size, args.hidden_dim, args.test_sweep, args.lm_weight, args.word_score, 
         args.sample_size, args.d_model, args.nhead, args.dim_feedforward, args.nlayers, 
         args.lstm_hidden, args.lstm_layers, args.dropout, args.sample_spect_folder, args.debug_sample)

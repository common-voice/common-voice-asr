# train.py (Corrected and Refactored)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import torchaudio
import wandb
import argparse
import string
import random
import torch.amp
from dotenv import load_dotenv
from rich.progress import Progress
from torch.utils.data import random_split, DataLoader, Subset
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from neural_networks.model_A.wrap_encoder import WrapEncoder
from neural_networks.datasets import CEL_MiniCVDataset, cel_collate_fn, cel_rnn_collate_fn
from neural_networks.datasets import CTC_MiniCVDataset, ctc_collate_fn, ctc_rnn_collate_fn
from neural_networks.datasets import transformer_collate_fn, transformer_conv_collate
from neural_networks.model_A.cnn_encoder import CTC_CNNEncoder, CEL_CNNEncoder
from neural_networks.model_A.rnn_encoder import CTC_RNNEncoder, CEL_RNNEncoder
from neural_networks.model_B.hybrid_transformer import HybridTransformer
from neural_networks.greedy_ctc_decoder import GreedyCTCDecoder
from neural_networks.beam_search_decoder import beam_search_decoder

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))

tokens = ['<blank>', '|'] + list(string.ascii_uppercase) + [' ', "'", '-']
N_MELS = 80
# for memory errors
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-data', action='store_true', help='Check if data loads correctly')
    parser.add_argument('--full_mini', action='store_true', help="Load larger data split from CV Train")
    parser.add_argument('--corpus', action='store_true', help="Load corpus train & dev large datasets")
    parser.add_argument('--greedy', action='store_true', help="Use Greedy CTC Decoder")
    parser.add_argument('--model_type', choices=['cnn', 'rnn', 'transformer'], required=True,
                        help='Specify which model to use')
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--logdir', type=str, required=True, help="Folder to write trains to")
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for model')
    parser.add_argument('--test-sweep', action='store_true', help="Testing sweep with small dataset")
    parser.add_argument('--lm_weight', type=float, default=3.23, help="Learning model weight for beam search decoder")
    parser.add_argument('--word_score', type=float, default=-0.26, help="Word score for beam search decoder")
    parser.add_argument('--sample_size', type=int, default=0,
                        help="Specifying sample size from the dataset, specifically with corpus")
    parser.add_argument('--d_model', type=int, default=512, help="Number of expected features in the encoder/decoder inputs")
    parser.add_argument('--nhead', type=int, default=8, help="Number of heads in the multiheadattention models")
    parser.add_argument('--dim_feedforward', type=int, default=2048, help="Dimension of the feedforward network model")
    parser.add_argument('--nlayers', type=int, default=6, help="Number of layers in the decoder")
    parser.add_argument('--lstm_hidden', type=int, default=256,
                        help="Number of hidden dimensions for LSTM layer of transformer model")
    parser.add_argument('--lstm_layers', type=int, default=1, help="Number of lstm layers in model")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout value for transformer model")
    parser.add_argument('--sample_spect_folder', type=str, default=None, help='Train & validate with a mel-spectogram sweep')
    parser.add_argument('--debug_sample', action='store_true', default=False,
                        help="Deugging with single training loop on one sample")
    parser.add_argument('--conv_layer', action='store_true', default=False, help="Add convolutional layer to Transformer")
    parser.add_argument('--beam_width', type=int, default=32, help="Beam size for beam search decoder")
    parser.add_argument('--best', action='store_true',
                        help="Using the best mel spectograms generated - EDIT: ran into errors using these files, DON'T USE")
    parser.add_argument('--demo', action='store_true', help='Run demo in Jupyter Notebook')
    return parser.parse_args()


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


def ctc_train(model, train_loader, optimizer, criterion, device, epoch, decoder, sample_size, corpus):
    model.train()
    losses = []
    count = 0
    total_wer = 0.0

    decode_interval = max(1, int(len(train_loader) * 0.25))

    # Use torch.cuda.amp.GradScaler for mixed-precision training
    scaler = torch.amp.GradScaler('cuda')  # Changed - torch.cuda.amp deprecated warning
    torch.autograd.set_detect_anomaly(True)  # for running out of GPU memory issues - recommended to help with that

    with Progress() as progress:
        pbar = progress.add_task(f"[green]Training Epoch {epoch}...", total=len(train_loader))
        for batch_idx, (spects, targets, input_lengths, target_lengths, empty_batch) in enumerate(train_loader):
            if empty_batch:
                continue
            spects, targets = spects.to(device), targets.to(device)
            input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

            optimizer.zero_grad()
            # Mitigating RuntimeError: Function 'CudnnRnnBackward0' returned nan values in 0th output
            with torch.amp.autocast('cuda', enabled=False):
                outputs = model(spects)  # Expected output shape: (Batch, Time, Classes)

                # --- FIX 2: ADD A GENTLE BIAS TO THE BLANK TOKEN ---
                # This encourages the model to use the blank token (index 0)
                blank_bias = 1.5
                outputs[:, :, 0] += blank_bias
                # --- END FIX 2 ---

                outputs = outputs.float()  # avoiding float16 instability in the LSTM layer
                log_probs = F.log_softmax(outputs, dim=2)

                # Permute for CTCLoss: (Time, Batch, Classes)
                log_probs_for_loss = log_probs.permute(1, 0, 2)

                T_max = outputs.size(1)
                input_lengths = input_lengths.clamp(max=T_max)

                loss = criterion(log_probs_for_loss, targets, input_lengths, target_lengths)

            # Backpropagation
            scaler.scale(loss).backward()

            # --- FIX 1: RE-INTRODUCE GRADIENT CLIPPING ---
            # This is crucial for stabilizing RNN training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # adjusting from 5.0 to 1.0 for error
            # --- END FIX 1 ---

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            # Decode for WER calculation
            # print("Input to decoder: ", outputs.shape)
            # if sample_size == 0 and corpus:
            if corpus:
                if batch_idx % decode_interval == 0:
                    print("input to decoder: ", outputs.shape)
                    hypotheses = decoder(outputs)
                    references = get_references(target_lengths, targets)
                    total_wer, count = get_train_wer(references, hypotheses, count, total_wer, batch_idx)
            else:
                hypotheses = decoder(outputs)  # Decoder expects (Batch, Time, Classes)
                references = get_references(target_lengths, targets)
                total_wer, count = get_train_wer(references, hypotheses, count, total_wer, batch_idx)

            progress.update(pbar, advance=1, description=f"[green]Training Epoch {epoch}... Loss: {loss.item():.4f}")

    avg_loss = sum(losses) / len(losses)
    avg_wer = total_wer / count if count > 0 else 1.0
    return avg_loss, avg_wer


def ctc_validate(model, val_loader, criterion, device, decoder, epoch):
    model.eval()
    losses = []
    count = 0
    total_wer, total_cer = 0.0, 0.0

    with Progress() as progress:
        pbar = progress.add_task(f"[cyan]Validating Epoch {epoch}...", total=len(val_loader))
        with torch.no_grad():
            for batch_idx, (spects, targets, input_lengths, target_lengths, empty_batch) in enumerate(val_loader):
                if empty_batch:
                    continue
                spects, targets = spects.to(device), targets.to(device)
                input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

                outputs = model(spects)
                outputs = outputs.float()
                log_probs = F.log_softmax(outputs, dim=2)
                log_probs_for_loss = log_probs.permute(1, 0, 2)

                loss = criterion(log_probs_for_loss, targets, input_lengths, target_lengths)
                losses.append(loss.item())

                hypotheses = decoder(outputs)
                references = get_references(target_lengths, targets)
                total_wer, total_cer, count = get_val_err(references, hypotheses, count, total_wer, total_cer, batch_idx)

                progress.update(pbar, advance=1, description=f"[cyan]Validating Epoch {epoch}... Loss: {loss.item():.4f}")

    avg_loss = sum(losses) / len(losses)
    avg_wer = total_wer / count if count > 0 else 1.0
    avg_cer = total_cer / count if count > 0 else 1.0
    return avg_loss, avg_wer, avg_cer


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


def tokens_to_str(tokens):
    return " ".join(tokens) if isinstance(tokens, list) else str(tokens)


def get_dataset(use_cel, use_best, data_type=None):
    corpus_path = os.path.join(BASE_DIR, "corpus_data")
    if use_cel:
        manifest_path = BASE_DIR / "data" / "manifest.csv"
        spect_dir = BASE_DIR / "data" / "processed" / "mini_cv"
        dataset = CEL_MiniCVDataset(manifest_path, spect_dir)
    else:
        if data_type == 'full_mini':
            manifest_path = os.path.join(BASE_DIR, "data/cleaned_manifest.csv")
            spect_dir = os.path.join(BASE_DIR, "data/processed/full_mini_cv")
        elif data_type == 'corpus':
            if use_best:
                print("Using best mel spectograms")
                train_spect_dir = os.path.join(corpus_path, "processed/best_train_cv")
                dev_spect_dir = os.path.join(corpus_path, "processed/best_dev_cv")
            else:
                train_spect_dir = os.path.join(corpus_path, "processed/train_cv")
                dev_spect_dir = os.path.join(corpus_path, "processed/dev_cv")
            train_manifest_path = os.path.join(corpus_path, "cleaned_train.csv")
            dev_manifest_path = os.path.join(corpus_path, "cleaned_dev.csv")
            train_set = CTC_MiniCVDataset(train_manifest_path, train_spect_dir)
            val_set = CTC_MiniCVDataset(dev_manifest_path, dev_spect_dir)
            return train_set, val_set
        elif data_type == "demo":
            manifest_path = os.path.join(corpus_path, 'demo_manifest.csv')
            spect_dir = os.path.join(corpus_path, "processed/demos")
        dataset = CTC_MiniCVDataset(manifest_path, spect_dir)
    return dataset


def split_set(set_len, corpus, dataset, train_set=None, val_set=None):
    total_len = set_len
    train_len = int(0.85 * total_len)
    val_len = total_len - train_len

    if corpus:
        train_indices = random.sample(range(len(train_set)), train_len)
        val_indices = random.sample(range(len(val_set)), val_len)
        if train_len > len(train_set) or val_len > len(val_set):
            raise ValueError(f"Sample sizes too large: train ({train_len}/{len(train_set)}), val ({val_len}/{len(val_set)})")
        train_set = Subset(train_set, train_indices)
        val_set = Subset(val_set, val_indices)
    else:
        train_set, val_set = random_split(dataset, [train_len, val_len])
    return train_set, val_set


def model_creation(model_type, num_classes, use_cel, hidden_dim, dropout, d_model, nhead, dim_ff, nlayers, lstm_hidden,
                   lstm_layers, conv_layer):
    if model_type == 'rnn':
        if use_cel:
            model = CEL_RNNEncoder()
            collate_fn = cel_rnn_collate_fn
        else:
            model = CTC_RNNEncoder(num_classes=num_classes, hidden_size=hidden_dim)
            collate_fn = ctc_rnn_collate_fn
    elif model_type == 'cnn':
        if use_cel:
            base_model = CEL_CNNEncoder()
            collate_fn = cel_collate_fn
        else:
            base_model = CTC_CNNEncoder(hidden_dim=hidden_dim)
            collate_fn = ctc_collate_fn
        # CNN model requires wrapping
        model = WrapEncoder(base_model, num_classes, apply=False, dropout=dropout)
    elif model_type == 'transformer':
        # Assuming transformer is self-contained like the new RNN model
        model = HybridTransformer(input_dim=N_MELS, vocab_size=num_classes, d_model=d_model, nhead=nhead,
                                  dim_feedforward=dim_ff, nlayers=nlayers, lstm_hidden=lstm_hidden,
                                  lstm_layers=lstm_layers, dropout=dropout, conv_layer=conv_layer)
        # Allowing for conformer model adjustment
        if args.conv_layer:
            collate_fn = transformer_conv_collate
        else:
            collate_fn = transformer_collate_fn
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model, collate_fn


def check_data(train_loader):
    for batch in train_loader:
        spects, transcripts = batch[:2]

        print("Spectrogram shape: ", spects.shape)
        print("Transcripts: ", transcripts)
        break


def train_loop(epochs, use_cel, model, train_loader, val_loader, optimizer, criterion, device, decoder, sample_size,
               corpus, writer):
    for epoch in range(1, epochs + 1):
        if use_cel:
            train_loss = cel_train(model, train_loader, optimizer, criterion, device)
            val_loss = cel_validate(model, train_loader, optimizer, criterion, device)
            print(f"\n--- Epoch {epoch} Summary ---")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss  : {val_loss:.4f}")
            print("--------------------------\n")
        else:
            train_loss, train_wer = ctc_train(model, train_loader, optimizer, criterion, device, epoch, decoder,
                                              sample_size, corpus)
            val_loss, val_wer, val_cer = ctc_validate(model, val_loader, criterion, device, decoder, epoch)

            print(f"\n--- Epoch {epoch} Summary ---")
            print(f"Train Loss: {train_loss:.4f}, Train WER: {train_wer:.4f}")
            print(f"Val Loss  : {val_loss:.4f}, Val WER  : {val_wer:.4f}, Val CER: {val_cer:.4f}")
            print("--------------------------\n")

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('WER/train', train_wer, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('WER/val', val_wer, epoch)
            writer.add_scalar('CER/val', val_cer, epoch)

            if wandb.run:
                wandb.log({
                    'epoch': epoch,
                    'train/ctc_loss': train_loss,
                    'train/wer': train_wer,
                    'val/ctc_loss': val_loss,
                    'val/wer': val_wer,
                    'val/cer': val_cer,
                    })

    writer.close()


def main(args):
    log_dir = os.path.join("neural_networks", args.logdir)
    log_path = os.path.join(BASE_DIR, log_dir)
    writer = SummaryWriter(log_dir=log_path)

    # --- Simplified Data Loading Logic ---
    use_cel = False

    if args.corpus or args.debug_sample:
        train_set, val_set = get_dataset(use_cel, args.best, 'corpus')
    else:
        if args.full_mini:
            dataset = get_dataset(use_cel, args.best, 'full_mini')
        elif args.demo:
            dataset = get_dataset(use_cel, args.best, 'demo')
        else:
            use_cel = True
            dataset = get_dataset(use_cel, args.best, 'mini')
        # For non-corpus cases that need splitting
        train_set, val_set = split_set(len(dataset), args.corpus, dataset)

    if args.debug_sample:
        print("--- DEBUG MODE ENABLED: USING ONE SAMPLE ---")
        train_set = torch.utils.data.Subset(train_set, [0])
        val_set = torch.utils.data.Subset(val_set, [0])

    # Change: Sample size splitting
    if args.corpus and not args.sample_size == 0:
        train_set, val_set = split_set(args.sample_size, args.corpus, dataset, train_set, val_set)

    # --- Simplified Model Creation Logic ---
    num_classes = len(tokens)
    model, collate_fn = model_creation(args.model_type, num_classes, use_cel, args.hidden_dim, args.dropout, args.d_model,
                                       args.nhead, args.dim_feedforward, args.nlayers, args.lstm_hidden, args.lstm_layers,
                                       args.conv_layer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    if args.check_data:
        check_data(train_loader)
        return

    # --- Training Loop ---
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.greedy:
        decoder = GreedyCTCDecoder(tokens)
    else:
        # Assuming beam search decoder is the alternative
        decoder = beam_search_decoder(tokens, lm_weight=args.lm_weight, word_score=args.word_score, beam_size=args.beam_width)
    
    train_loop(args.epochs, use_cel, model, train_loader, val_loader, optimizer, criterion, device, decoder, args.sample_size,
               args.corpus, writer)


if __name__ == "__main__":
    args = parse_command_args()
    main(args)

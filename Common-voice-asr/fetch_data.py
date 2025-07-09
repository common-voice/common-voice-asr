import os
import shutil
import pandas as pd
import librosa
import csv
import argparse
from dotenv import load_dotenv
from pathlib import Path
from preprocess import preprocess

# python -m fetch_data.py --corpus
# from base: python -m common-voice-asr.Common-voice-asr.fetch_data --corpus 

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))


def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_mini', action='store_true', help='Fetch full mini dataset')
    parser.add_argument('--corpus', action='store_true', help='Fetch corpus dataset, split into dev & train')
    return parser.parse_args()


DATA_DIR_21 = "/Users/setongerrity/Desktop/Mozilla/cv-corpus-21.0-delta-2025-03-14/en"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/raw/mini_cv")
TRANSCRIPTS_FILE_21 = os.path.join(DATA_DIR_21, "validated.tsv")
CLIPS_DIR_21 = os.path.join(DATA_DIR_21, "clips")
MANIFEST_PATH = os.path.join(BASE_DIR, "data/manifest.csv")

DATA_DIR_20 = "/Users/setongerrity/Desktop/Mozilla/cv-corpus-20.0-delta-2024-12-06/en"
TRANSCRIPTS_FILE_20 = os.path.join(DATA_DIR_20, "validated.tsv")
CLIPS_DIR_20 = os.path.join(DATA_DIR_20, "clips")
OUTPUT_FULL_DIR = os.path.join(BASE_DIR, "data/raw/full_mini_cv")
MANIFEST_FULL_PATH = os.path.join(BASE_DIR, "data/manifest_full.csv")

DATA_DIR_CORPUS = 'cv-corpus-22.0-2025-06-20/en'
TRANSCRIPTS_FILE_TRAIN = os.path.join(DATA_DIR_CORPUS, 'train.tsv')
TRANSCRIPTS_FILE_DEV = os.path.join(DATA_DIR_CORPUS, 'dev.tsv')
DURATIONS_FILE = os.path.join(DATA_DIR_CORPUS, 'clip_durations.tsv')
CLIPS_DIR_CORPUS =  os.path.join(DATA_DIR_CORPUS, "clips")
OUTPUT_DIR_TRAIN = os.path.join(BASE_DIR, 'common-voice-asr/Common-voice-asr/corpus_data/raw/train_cv')
OUTPUT_DIR_DEV = os.path.join(BASE_DIR, 'common-voice-asr/Common-voice-asr/corpus_data/raw/dev_cv')
MANIFEST_TRAIN_PATH = os.path.join(BASE_DIR, 'common-voice-asr/Common-voice-asr/corpus_data/manifest_train.csv')
MANIFEST_DEV_PATH = os.path.join(BASE_DIR, 'common-voice-asr/Common-voice-asr/corpus_data/manifest_dev.csv')
SPECT_TRAIN_PATH = os.path.join(BASE_DIR, 'common-voice-asr/Common-voice-asr/corpus_data/processed/train_cv')
SPECT_DEV_PATH = os.path.join(BASE_DIR, 'common-voice-asr/Common-voice-asr/corpus_data/processed/dev_cv')


def fetch_full_mini():
    transcripts_frame_21 = pd.read_csv(TRANSCRIPTS_FILE_21, sep="\t")
    transcripts_frame_20 = pd.read_csv(TRANSCRIPTS_FILE_20, sep="\t")
    with open(MANIFEST_FULL_PATH, 'w', newline='', encoding='utf-8') as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(['filename', 'transcript', 'duration'])

        for idx, row in transcripts_frame_21.iterrows():
            filename = row['path']
            sentence = row['sentence']

            og_audio_path = os.path.join(CLIPS_DIR_21, filename)
            dest_audio_path = os.path.join(OUTPUT_FULL_DIR, filename)

            if os.path.exists(og_audio_path):
                shutil.copy(og_audio_path, dest_audio_path)
                try:
                    duration = librosa.get_duration(path=dest_audio_path)
                except Exception as e:
                    print(f"Could not load audio file {filename}: {e}")
                    continue
                writer.writerow([filename, sentence, round(duration, 3)])
            else:
                print(f"Warning: {filename} not found")
        for idx, row in transcripts_frame_20.iterrows():
            filename = row['path']
            sentence = row['sentence']

            og_audio_path = os.path.join(CLIPS_DIR_20, filename)
            dest_audio_path = os.path.join(OUTPUT_FULL_DIR, filename)

            if os.path.exists(og_audio_path):
                shutil.copy(og_audio_path, dest_audio_path)
                try:
                    duration = librosa.get_duration(path=dest_audio_path)
                except Exception as e:
                    print(f"Could not load audio file {filename}: {e}")
                    continue
                writer.writerow([filename, sentence, round(duration, 3)])
            else:
                print(f"Warning: {filename} not found")


# Mini manifest.csv & audio file generation
def fetch_mini():
    num_samples = 100
    transcripts_frame_21 = pd.read_csv(TRANSCRIPTS_FILE_21, sep="\t")
    sampled_transcripts_frame = transcripts_frame_21.sample(n=num_samples, random_state=42)
    with open(MANIFEST_PATH, 'w', newline='', encoding='utf-8') as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(['filename', 'transcript', 'duration'])

        for idx, row in sampled_transcripts_frame.iterrows():
            filename = row['path']
            sentence = row['sentence']

            og_audio_path = os.path.join(CLIPS_DIR_21, filename)
            dest_audio_path = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(og_audio_path):
                shutil.copy(og_audio_path, dest_audio_path)
                try:
                    duration = librosa.get_duration(path=dest_audio_path)
                except Exception as e:
                    print(f"Could not load audio file {filename}: {e}")
                    continue
                writer.writerow([filename, sentence, round(duration, 3)])
            else:
                print(f"Warning: {filename} not found")


def fetch_corpus(manifest_path, transcripts_file, output_dir):
    transcripts_frame = pd.read_csv(transcripts_file, sep="\t", usecols=["path", "sentence"], low_memory=False)
    durations_frame = pd.read_csv(DURATIONS_FILE, sep="\t", usecols=["clip", "duration[ms]"], low_memory=False)
    durations_frame.rename(columns={'clip': 'path'}, inplace=True)
    
    merged_frame = pd.merge(transcripts_frame, durations_frame, on='path', how='inner')
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(manifest_path, 'w', newline='', encoding='utf-8') as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(['filename', 'transcript', 'duration'])

        for idx, row in merged_frame.iterrows():
            filename = row['path']
            sentence = row['sentence']
            duration_sec = round(row['duration[ms]'] / 1000.0, 3)

            og_audio_path = os.path.join(CLIPS_DIR_CORPUS, filename)
            dest_audio_path = os.path.join(output_dir, filename)

            if os.path.exists(og_audio_path):
                shutil.copy(og_audio_path, dest_audio_path)
                writer.writerow([filename, sentence, duration_sec])
            else:
                print(f"Warning: {filename} not found")

def regen_csv(tsv_path, durations_path, output_csv):
    transcripts = pd.read_csv(tsv_path, sep='\t', usecols=["path", "sentence"], quoting=3)
    durations = pd.read_csv(durations_path, sep='\t', usecols=["clip", "duration[ms]"], quoting=3)
    durations = durations.rename(columns={"clip": "path"})
    merged = pd.merge(transcripts, durations, on='path', how='inner')
    merged = merged.rename(columns={"path": "filename", "sentence": "transcript", "duration[ms]": "duration_ms"})
    
    merged["duration"] = (merged["duration_ms"] / 1000.0).round(3)
    merged = merged[["filename", "transcript", "duration"]]
    merged.to_csv(output_csv, index=False, sep=',', quoting=1)

def fetch_absent_files(transcripts_file, output_dir):
    transcripts_frame = pd.read_csv(transcripts_file, sep="\t", usecols=["path"], low_memory=False)
    os.makedirs(output_dir, exist_ok=True)
    for filename in transcripts_frame["path"]:
        og_audio_path = os.path.join(CLIPS_DIR_CORPUS, filename)
        dest_audio_path = os.path.join(output_dir, filename)
        if os.path.exists(og_audio_path) and not os.path.exists(dest_audio_path):
            shutil.copy(og_audio_path, dest_audio_path)
        elif not os.path.exists(og_audio_path):
            print(f"Warning: {filename} not found in original audio dir.")
            
def find_missing_npy(manifest_csv, spect_dir):
    df = pd.read_csv(manifest_csv)
    missing = []
    for filename in df['filename']:
        npy_path = os.path.join(spect_dir, filename.replace(".mp3", ".npy"))
        if not os.path.exists(npy_path):
            missing.append(npy_path)
    print(f"{len(missing)} spectograms missing")
    return missing


def if_exist_process(missing_files, raw_dir, spect_dir):
    for file in missing_files:
        if file.endswith(".npy"):
            mp3_filename = file.replace(".npy", ".mp3")
            input_path = os.path.join(raw_dir, mp3_filename)
            output_path = os.path.join(spect_dir, file)

            if not os.path.exists(input_path):
                print(f"[MISSING AUDIO] {input_path}")
                continue
            if os.path.exists(output_path):
                continue
            try:
                y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                                                          n_mels=N_MELS, power=2.0)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0, top_db=80)
                np.save(output_path, log_mel_spec)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {mp3_filename}: {e}")
                    

def main(full_mini: bool = False, corpus: bool = False):
    """
    if full_mini:
        fetch_full_mini()
    elif corpus:
        fetch_corpus(MANIFEST_TRAIN_PATH, TRANSCRIPTS_FILE_TRAIN, OUTPUT_DIR_TRAIN)
        fetch_corpus(MANIFEST_DEV_PATH, TRANSCRIPTS_FILE_DEV, OUTPUT_DIR_DEV)
    else:
        fetch_mini()
    """
    # regen_csv(TRANSCRIPTS_FILE_DEV, DURATIONS_FILE, MANIFEST_DEV_PATH)
    fetch_absent_files(TRANSCRIPTS_FILE_TRAIN, OUTPUT_DIR_TRAIN)
    fetch_absent_files(TRANSCRIPTS_FILE_DEV, OUTPUT_DIR_DEV)
    train_missing = find_missing_npy(MANIFEST_TRAIN_PATH, SPECT_TRAIN_PATH)
    dev_missing = find_missing_npy(MANIFEST_DEV_PATH, SPECT_DEV_PATH)
    if_exist_process(train_missing, OUTPUT_DIR_TRAIN, SPECT_TRAIN_PATH)
    if_exist_process(dev_missing, OUTPUT_DIR_DEV, SPECT_DEV_PATH)


if __name__ == "__main__":
    args = parse_command_args()
    main(full_mini=args.full_mini, corpus=args.corpus)

import os
import shutil
import pandas as pd
import librosa
import csv
import argparse
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))


def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_mini', action='store_true', help='Fetch full mini dataset')
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

transcripts_frame_21 = pd.read_csv(TRANSCRIPTS_FILE_21, sep="\t")
transcripts_frame_20 = pd.read_csv(TRANSCRIPTS_FILE_20, sep="\t")


def fetch_full_mini():
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


def main(full_mini: bool = False):
    if full_mini:
        fetch_full_mini()
    else:
        fetch_mini()


if __name__ == "__main__":
    args = parse_command_args()
    main(full_mini=args.full_mini)

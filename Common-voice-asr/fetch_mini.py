import os
import shutil
import pandas as pd
import librosa
import csv

DATA_DIR = "/Users/setongerrity/Desktop/Mozilla/cv-corpus-21.0-delta-2025-03-14/en"
OUTPUT_DIR = "/Users/setongerrity/Desktop/Mozilla/common-voice-asr/Common-voice-asr/data/raw/mini_cv"
TRANSCRIPTS_FILE = os.path.join(DATA_DIR, "validated.tsv")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
MANIFEST_PATH = "/Users/setongerrity/Desktop/Mozilla/common-voice-asr/Common-voice-asr/data/manifest.csv"

transcripts_frame = pd.read_csv(TRANSCRIPTS_FILE, sep="\t")
num_samples = 100
sampled_transcripts_frame = transcripts_frame.sample(n=num_samples, random_state=42)

with open(MANIFEST_PATH, 'w', newline='', encoding='utf-8') as manifest_file:
    writer = csv.writer(manifest_file)
    writer.writerow(['filename', 'transcript', 'duration'])


    for idx, row in sampled_transcripts_frame.iterrows():
        filename = row['path']
        sentence = row['sentence']

        og_audio_path = os.path.join(CLIPS_DIR, filename)
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


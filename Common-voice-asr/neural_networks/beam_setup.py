import pandas as pd
import os
import re
import csv
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

# cd into Common-voice-asr then python -m neural_networks.beam_setup

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))

manifest_path = BASE_DIR / "data" / "manifest_full.csv"
output_path = BASE_DIR / "data" / "corpus.txt"
lexicon_path = BASE_DIR / "data" / "lexicon.txt"
cleaned_path = BASE_DIR / "data" / "cleaned_text.txt"
cleaned_csv = BASE_DIR / "data" / "cleaned_manifest.csv"
                
MANIFEST_TRAIN = os.path.join(BASE_DIR, 'corpus_data/manifest_train.csv')
MANIFEST_DEV = os.path.join(BASE_DIR, 'corpus_data/manifest_dev.csv')
ENT_LEXICON_PATH = BASE_DIR / "corpus_data" / "lexicon.txt"
ENT_CORPUS_PATH = BASE_DIR / "corpus_data" / "corpus.txt"
CLEANED_TRAIN = BASE_DIR / "corpus_data" / "cleaned_train.csv"
CLEANED_DEV = BASE_DIR / "corpus_data" / "cleaned_dev.csv"
SPECT_TRAIN_PATH = os.path.join(BASE_DIR, 'corpus_data/processed/train_cv')
SPECT_DEV_PATH = os.path.join(BASE_DIR, 'corpus_data/processed/dev_cv')


def is_en(text):
    return all(ord(c) < 128 for c in text)


def normalize_text(text):
    text = text.upper()
    text = re.sub(r"[-]", " ", text)
    text = re.sub(r"[_]", " ", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if is_en(text) else ""


def clean_manifest(manifest_path, cleaned_path, spect_dir):
    manifest = pd.read_csv(manifest_path, usecols=['filename', 'duration', 'transcript'])
    
    # transcript
    manifest['transcript'] = manifest['transcript'].fillna("")
    manifest['transcript'] = manifest['transcript'].apply(normalize_text)
    manifest = manifest[manifest["transcript"].str.strip().astype(bool)]
    
    # removing non-existing spects
    valid_rows = []
    for idx, row in manifest.iterrows():
        spect_path = os.path.join(spect_dir, row["filename"].replace(".mp3", ".npy"))
        if os.path.exists(spect_path):
            valid_rows.append(row)
        else:
            print(f"[REMOVED] No spectrogram: {row['filename']}")
    
    cleaned_df = pd.DataFrame(valid_rows)
    cleaned_df.to_csv(cleaned_path, index=False)
    return cleaned_df['transcript'].tolist()


def write_corpus(transcripts, corpus_path):
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in transcripts:
            if line.strip():
                f.write(line.strip() + "\n")
    

def create_lexicon(corpus_path, lexicon_path):
    with open(corpus_path, encoding="utf-8") as f:
        lines = f.readlines()
    word_counts = Counter()
    for line in lines:
        # was very testy about the specific quotation marks & apostrophes so important to normalize
        line = line.replace("’", "'")
        line = line.replace("‘", "'")
        line = line.replace("“", '"')
        line = line.replace("”", '"')
        words = line.strip().upper().split()
        word_counts.update(words)

    with open(lexicon_path, "w", encoding="utf-8") as f:
        for word in sorted(word_counts.keys()):
            word_clean = word.strip("'\" ")
            if not word_clean or not word_clean.isascii() or re.match(r"^\W+$", word_clean):
                continue
            spelling = " ".join(list(word_clean))
            f.write(f"{word_clean} {spelling}\n")


if __name__ == "__main__":
    cleaned_train_lines = clean_manifest(MANIFEST_TRAIN, CLEANED_TRAIN, SPECT_TRAIN_PATH)
    cleaned_dev_lines = clean_manifest(MANIFEST_DEV, CLEANED_DEV, SPECT_DEV_PATH)
    write_corpus(cleaned_train_lines + cleaned_dev_lines, ENT_CORPUS_PATH)
    create_lexicon(ENT_CORPUS_PATH, ENT_LEXICON_PATH)

import pandas as pd
import os
import re
import csv
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))

manifest_path = BASE_DIR / "data" / "manifest_full.csv"
manifest = pd.read_csv(manifest_path)
transcripts = manifest['transcript'].dropna().str.upper()
corpus_path = BASE_DIR / "data" / "corpus.txt"
lexicon_path = BASE_DIR / "data" / "lexicon.txt"
cleaned_path = BASE_DIR / "data" / "cleaned_text.txt"
cleaned_csv = BASE_DIR / "data" / "cleaned_manifest.csv"

def extract_transcripts():
    with open(corpus_path, "w") as f:
        for line in transcripts:
            f.write(line.strip() + "\n")

def create_lexicon():
    with open(corpus_path) as f:
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

    with open(lexicon_path, "w") as f:
        for word in word_counts:
            spelling = " ".join(list(word))
            f.write(f"{word} {spelling}\n")

def normalize_text(text):
    text = text.upper()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_transcripts():
    manifest["cleaned_transcript"] = manifest["transcript"].apply(normalize_text)
    manifest["cleaned_transcript"].to_csv(cleaned_path, index=False, header=False)

def clean_lexicon():
    with open(cleaned_path) as f:
        lines = f.readlines()
        word_counts = Counter()
        for line in lines:
            words = line.strip().upper().split()
            word_counts.update(words)
    with open(lexicon_path, "w") as f:
        for word in word_counts:
            spelling = " ".join([c.upper() if c != "'" else "'" for c in word])
            f.write(f"{word} {spelling}\n")

def txt_to_csv():
    with open(cleaned_path, "r") as f:
        cleaned_lines = [line.strip() for line in f.readlines()]
    with open(manifest_path, "r") as original, open(cleaned_csv, "w", newline="") as output:
        reader = csv.reader(original)
        writer = csv.writer(output)
        header = next(reader)
        writer.writerow(header)
        for idx, row in enumerate(reader):
            if idx < len(cleaned_lines):
                row[1] = cleaned_lines[idx]
            writer.writerow(row)

if __name__ == "__main__":
    txt_to_csv()
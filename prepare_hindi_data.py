import os
os.environ["HF_HOME"] = "/media/storage/hf_cache"

from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from pathlib import Path
from tqdm import tqdm
import random

# Test phonemizer works first
test = phonemize("नमस्ते", language='hi', backend='espeak', with_stress=True)
print(f"Phonemizer test: {test}")

metadata_path = "/media/storage/vani_dataset/metadata.txt"
wav_root = "/media/storage/vani_dataset/wav"

lines = open(metadata_path, encoding="utf-8").readlines()
print(f"Total samples: {len(lines)}")

records = []
failed = 0
texts_batch = []
wavpaths_batch = []

# Collect all valid entries first
for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split("|", 1)
    if len(parts) != 2:
        continue
    fname, text = parts
    wav_path = f"{wav_root}/{fname}"
    if Path(wav_path).exists() and text:
        texts_batch.append(text)
        wavpaths_batch.append(wav_path)

print(f"Valid entries: {len(texts_batch)}")
print("Phonemizing in batches (this takes ~5-10 mins)...")

# Batch phonemize — much faster than one-by-one
phonemes_list = phonemize(
    texts_batch,
    language='hi',
    backend='espeak',
    with_stress=True,
    njobs=4,  # use 4 CPU cores
    separator=None,
)

for wav_path, phonemes in zip(wavpaths_batch, phonemes_list):
    if phonemes and phonemes.strip():
        records.append(f"{wav_path}|{phonemes.strip()}|0")
    else:
        failed += 1

print(f"Phonemized: {len(records)}  Failed: {failed}")
print("\nSample outputs:")
for r in records[:3]:
    print(r)

# Split 95/5
random.seed(42)
random.shuffle(records)
split = int(len(records) * 0.95)
train = records[:split]
val   = records[split:]

with open("/media/storage/vani-training/Data/train_list.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train))
with open("/media/storage/vani-training/Data/val_list.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val))

print(f"\nTrain: {len(train)} → Data/train_list.txt")
print(f"Val:   {len(val)} → Data/val_list.txt")

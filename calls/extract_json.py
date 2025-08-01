# nanogpt/data/prepare_enriched_transcripts.py
"""
Merge 911 call transcripts (.txt) with JSON side-info into a single input.txt
Usage (from nanogpt/):
    python data/prepare_enriched_transcripts.py
"""

import os, json

BASE_DIR      = "/mnt/scratch/users/sgkshah/999"
TEXT_DIR      = os.path.join(BASE_DIR, "output")
JSON_FILE     = os.path.join(BASE_DIR, "sample.json")
OUT_FILE      = "input.txt"
BLOCK_CHARS   = 5_000  # rough ≈ 128 BPE tokens, tune if needed

# ------------------------------------------------------------------
# 1. Load JSON side-info
with open(JSON_FILE) as f:
    metas = json.load(f)                    # list[dict]

# ------------------------------------------------------------------
# 2. Collect TXT files
txt_files = sorted([os.path.join(TEXT_DIR, f)
                    for f in os.listdir(TEXT_DIR)
                    if f.endswith('.txt')])

if not txt_files:
    raise FileNotFoundError(f"No .txt files found in {TEXT_DIR}")

# ------------------------------------------------------------------
# 3. Merge everything into one text
# Removed os.makedirs since OUT_FILE is in the current directory

with open(OUT_FILE, "w", encoding="utf-8") as fout:
    for i, txt_path in enumerate(txt_files):
        # Read conversation
        with open(txt_path, encoding="utf-8") as f:
            convo = f.read().strip()

        # Add header only if within the number of metas
        if i < len(metas):
            meta = metas[i]
            header = f"Name: {meta['name']}, Address: {meta['address']}\n"
        else:
            header = ""

        # Concatenate
        sample = header + convo

        # Optional: truncate if too long (hard cut to stay inside block_size)
        if len(sample) > BLOCK_CHARS:
            sample = sample[:BLOCK_CHARS]

        fout.write(sample + "\n\n")  # two newlines = document separator

print(f"Wrote {len(txt_files)} enriched samples → {OUT_FILE}")
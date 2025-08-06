#!/usr/bin/env python3
"""
Merge 911 call transcripts (.txt) with JSON side-info into a single input.txt for training.
Removed Hugging Face dataset integration as requested.
Usage (from nanogpt/):
    python data/prepare_enriched_transcripts.py
"""

import os
import json
import traceback
import glob

BASE_DIR      = "999"
TEXT_DIR      = os.path.join(BASE_DIR, "output")
JSON_FILE     = os.path.join(BASE_DIR, "sample.json")
OUT_FILE      = "input.txt"

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEXT_DIR: {TEXT_DIR}")
print(f"JSON_FILE: {JSON_FILE}")

# ------------------------------------------------------------------
# 1. Load JSON side-info
try:
    with open(JSON_FILE) as f:
        metas = json.load(f)  # list[dict]
    print(f"Loaded {len(metas)} metadata entries from JSON")
except Exception as e:
    print(f"Error loading JSON file: {e}")
    metas = []

# ------------------------------------------------------------------
# 2. Collect TXT files recursively
try:
    txt_files = sorted(glob.glob(os.path.join(TEXT_DIR, '**/*.txt'), recursive=True))
    
    print(f"Found {len(txt_files)} .txt files recursively in {TEXT_DIR} and subdirectories")
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {TEXT_DIR} or subdirectories")
        
except Exception as e:
    print(f"Error accessing TEXT_DIR: {e}")
    txt_files = []

# ------------------------------------------------------------------
# 3. Merge everything into one text
print("Writing to output file...")
local_count = 0
skipped_local = 0

try:
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        print("Processing local TXT files...")
        for i, txt_path in enumerate(txt_files):
            try:
                with open(txt_path, encoding="utf-8", errors="ignore") as f:
                    convo = f.read().strip()
                
                if not convo:
                    print(f"Warning: Empty content in {txt_path}")
                    skipped_local += 1
                    continue

                # Add header only if within the number of metas
                if i < len(metas):
                    meta = metas[i]
                    header = f"Name: {meta.get('name', 'Unknown')}, Address: {meta.get('address', 'Unknown')}\n"
                else:
                    header = ""

                sample = header + convo

                fout.write(sample + "\n\n")  # two newlines = document separator
                local_count += 1
                
                if i % 10 == 0:  # Progress indicator
                    print(f"Processed {i+1}/{len(txt_files)} local files")
                    
            except Exception as e:
                print(f"Error processing {txt_path}: {e}")
                skipped_local += 1
                continue

    print(f"SUCCESS: Wrote {local_count} local samples → {OUT_FILE} (skipped: {skipped_local})")
    
except Exception as e:
    print(f"Error writing to output file: {e}")
    traceback.print_exc()

# ------------------------------------------------------------------
# 4. Verify output
try:
    file_size = os.path.getsize(OUT_FILE)
    print(f"Output file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    
    # Show first few lines for verification
    with open(OUT_FILE, 'r', encoding='utf-8') as f:
        first_lines = f.read(1000)
        print("\nFirst 1000 characters of output:")
        print("-" * 50)
        print(first_lines)
        print("-" * 50)
        
except Exception as e:
    print(f"Error checking output file: {e}")
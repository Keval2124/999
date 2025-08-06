#!/usr/bin/env python3
"""
clean_and_merge.py
Cleans local data, removes duplicates from HF dataset,
and merges unique dialogs into clean_input.txt
"""

import os
import re
import difflib

def clean_message(text):
    """Identical to download script's cleaner"""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    deduped = []
    for word in words:
        if not deduped or word.lower() != deduped[-1].lower():
            deduped.append(word)
    return ' '.join(deduped)

def read_dialogs(path):
    """Read dialogs from file with blank line separation"""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    return [
        [ln.strip() for ln in block.splitlines() if ln.strip()]
        for block in content.split("\n\n")
    ] if content else []

def read_and_clean_local(path):
    """Parse local input.txt with timestamp handling"""
    if not os.path.exists(path):
        print(f"Local file {path} not found")
        return []
    
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    
    dialogs, current = [], []
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                dialogs.append(current)
                current = []
            continue
        
        # Remove timestamps
        line = re.sub(r'\[\d+:\d+:\d+\]', '', line).strip()
        
        # Parse speaker and content
        match = re.match(r'^\s*(caller|operator)\s*:\s*(.+)$', line, re.I)
        if match:
            speaker, content = match.groups()
            speaker = speaker.upper()
            content = clean_message(content)
            current.append(f"{speaker}: {content}")
    
    if current:
        dialogs.append(current)
    return dialogs

def get_content_block(dialog):
    """Extract dialog content without speaker labels"""
    return '\n'.join(ln.split(':', 1)[1].strip() for ln in dialog if ':' in ln)

# Load data
local_dialogs = read_and_clean_local("input.txt")
hf_dialogs = read_dialogs("additional_911_dialogs.txt")

print(f"Local dialogs: {len(local_dialogs)}")
print(f"HF dialogs: {len(hf_dialogs)}")

# Deduplicate local against HF
unique_local = []
SIMILARITY_THRESHOLD = 0.8

for local in local_dialogs:
    local_content = get_content_block(local)
    if not any(
        difflib.SequenceMatcher(None, local_content, get_content_block(hf)).ratio() > SIMILARITY_THRESHOLD
        for hf in hf_dialogs
    ):
        unique_local.append(local)

print(f"Unique local dialogs after deduplication: {len(unique_local)}")

# Combine and save
with open("clean_input.txt", "w", encoding="utf-8") as f:
    for dialog in hf_dialogs + unique_local:
        f.write("\n".join(dialog) + "\n\n")

print("Saved combined dialogs to clean_input.txt")
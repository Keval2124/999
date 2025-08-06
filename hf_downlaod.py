#!/usr/bin/env python3
"""
download_hf_dataset.py
Downloads and pre-processes Hugging Face dataset.
Saves cleaned dialogs to 'hf_cleaned_dialogs.txt'
"""

from datasets import load_dataset
import re

def clean_message(text):
    """Clean text message by removing extra spaces and duplicates"""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    deduped = []
    for word in words:
        if not deduped or word.lower() != deduped[-1].lower():
            deduped.append(word)
    return ' '.join(deduped)

# Load dataset
print("Downloading Hugging Face dataset...")
hf_dataset = load_dataset("spikecodes/911-call-transcripts")

# Process and clean dialogs
with open("hf_cleaned_dialogs.txt", "w", encoding="utf-8") as f:
    for example in hf_dataset['train']:
        dialog_lines = []
        for msg in example.get('messages', []):
            role = msg.get('role', '').upper()
            content = clean_message(msg.get('content', ''))
            
            if not content:
                continue
                
            if 'USER' in role or 'CALLER' in role:
                dialog_lines.append(f"CALLER: {content}")
            elif 'ASSISTANT' in role or 'OPERATOR' in role:
                dialog_lines.append(f"OPERATOR: {content}")
        
        if dialog_lines:
            f.write("\n".join(dialog_lines) + "\n\n")

print("Saved cleaned HF dialogs to additional_911_dialogs.txt")
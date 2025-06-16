#!/usr/bin/env python3
"""
Transcription + Speaker Diarization + Overlap Detection for 911 calls.
Uses Whisper and pyannote.audio.
"""

import os
import json
import argparse
import whisper
import torch
import torchaudio
from pydub import AudioSegment
from pyannote.audio import Pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_INPUT_DIR = "./911_recordings"
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_CLEAN_DIR = "./clean_audio"
DEFAULT_DIALOGUE_PATH = "./calls_dialogue.json"
DEFAULT_MAX_FILES = 20

# Load the model and utilities
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
model = model.to(device)  # Move model to GPU/CPU
get_speech_timestamps, save_audio, read_audio, VADIterator, _ = utils

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio.export(wav_path, format="wav")
    return wav_path

def apply_vad(wav_path):
    wav, sr = torchaudio.load(wav_path)
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # Average channels to mono
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000
    wav = wav.to(device)

    speech_timestamps = get_speech_timestamps(wav.squeeze(0), model, sampling_rate=sr)

    if not speech_timestamps:
        print(f"No speech detected in {wav_path}, returning original")
        return wav_path

    speech_audio = torch.cat([wav[:, ts['start']:ts['end']] for ts in speech_timestamps], dim=1)
    cleaned_path = wav_path.replace(".wav", "_clean.wav")
    torchaudio.save(cleaned_path, speech_audio.cpu(), sr)
    return cleaned_path

def run_diarization(pipeline, wav_path):
    diarization_result = pipeline(wav_path)
    return diarization_result

def load_whisper_model():
    print("Loading Whisper model...")
    model = whisper.load_model("tiny")
    return model

def transcribe_with_whisper(model, wav_path):
    print(f"Transcribing {wav_path}...")
    result = model.transcribe(wav_path)
    return result["segments"]

def merge_diarization_transcript(diarization, segments):
    merged_dialogue = []

    # Convert pyannote diarization to list of dicts with start, end, speaker
    diarization_segments = []
    for turn in diarization.itersegments(label=True):
        diarization_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": turn.label
        })

    # For each Whisper segment, find overlapping diarization speakers
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        text = seg["text"].strip()
        if not text:
            continue

        # Find diarization speakers overlapping this segment
        overlapping_speakers = set()
        for ds in diarization_segments:
            overlap_start = max(seg_start, ds["start"])
            overlap_end = min(seg_end, ds["end"])
            if overlap_start < overlap_end:  # overlap exists
                overlapping_speakers.add(ds["speaker"])

        # Handle no diarization speaker found (assign "UNKNOWN")
        if not overlapping_speakers:
            overlapping_speakers.add("UNKNOWN")

        # Create one dialogue entry per speaker for overlaps
        for spk in overlapping_speakers:
            merged_dialogue.append({
                "speaker": spk,
                "text": text,
                "start": seg_start,
                "end": seg_end,
                "overlap": len(overlapping_speakers) > 1
            })

    return merged_dialogue

def save_transcription(output_path, merged_dialogue):
    with open(output_path, 'w') as f:
        for turn in merged_dialogue:
            overlap_tag = " (overlap)" if turn.get("overlap", False) else ""
            f.write(f"{turn['speaker']}{overlap_tag}: {turn['text']}\n\n")

def transcribe_audio_files(input_dir, output_dir, clean_dir, dialogue_path, max_files=DEFAULT_MAX_FILES):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    model = load_whisper_model()

    # Load pyannote diarization pipeline (using pretrained speaker diarization pipeline)
    print("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=os.environ.get("HF_TOKEN"))

    audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.lower().endswith(('.mp3', '.wav'))][:max_files]

    all_dialogues = {}

    for file_path in audio_files:
        filename = os.path.basename(file_path)

        # Convert MP3 to WAV if needed
        if filename.lower().endswith('.mp3'):
            wav_path = os.path.join(clean_dir, filename.replace('.mp3', '.wav'))
            convert_mp3_to_wav(file_path, wav_path)
        else:
            wav_path = file_path

        # Apply VAD trimming
        cleaned_wav_path = apply_vad(wav_path)

        # Run diarization
        diarization = run_diarization(pipeline, cleaned_wav_path)

        # Run Whisper transcription
        segments = transcribe_with_whisper(model, cleaned_wav_path)

        # Merge diarization and transcription segments
        merged_dialogue = merge_diarization_transcript(diarization, segments)

        # Save per-file transcription
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        save_transcription(output_path, merged_dialogue)
        print(f"Saved transcription with diarization to {output_path}")

        # Store all dialogue for JSON output
        all_dialogues[filename.replace('.mp3', '.wav')] = merged_dialogue

    # Save combined JSON
    with open(dialogue_path, 'w') as f:
        json.dump(all_dialogues, f, indent=2)

    print(f"Saved all dialogues to {dialogue_path}")
    return all_dialogues

def main():
    parser = argparse.ArgumentParser(description='911 call transcription with diarization and VAD')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--clean_dir', type=str, default=DEFAULT_CLEAN_DIR)
    parser.add_argument('--dialogue_path', type=str, default=DEFAULT_DIALOGUE_PATH)
    parser.add_argument('--max_files', type=int, default=DEFAULT_MAX_FILES)
    args = parser.parse_args()

    transcribe_audio_files(args.input_dir, args.output_dir, args.clean_dir, args.dialogue_path, args.max_files)

if __name__ == "__main__":
    main()

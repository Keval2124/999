#!/usr/bin/env python3
"""
Full transcription pipeline with automatic post-finetuning
every 50 new transcripts.
"""

import torch
import torch.multiprocessing as mp
import os
import whisper
from pyannote.audio import Pipeline
from datetime import timedelta
import json
import re
from collections import defaultdict
import nlpaug.augmenter.word as naw
from transformers import pipeline as hf_pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
import gc
from multiprocessing import Manager
import psutil
import time
import sys
import subprocess          # <--- NEW (only for post-finetune trigger)

# ------------------------------------------------------------------
# NLTK bootstrap (unchanged)
# ------------------------------------------------------------------
custom_nltk_dir = os.path.abspath('nltk_data')
nltk.data.path.append(custom_nltk_dir)
tagger_path = os.path.join(custom_nltk_dir, 'taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
if not os.path.exists(tagger_path):
    print("[+] Downloading NLTK tagger to custom dir...")
    nltk.download('averaged_perceptron_tagger', quiet=True, download_dir=custom_nltk_dir)
else:
    print("[+] NLTK tagger found in custom dir, skipping download.")

# ------------------------------------------------------------------
# Configuration (unchanged)
# ------------------------------------------------------------------
class Config:
    HF_TOKEN = ""
    WHISPER_MODEL = "large-v3"
    FINETUNED_BERT_MODEL_PATH = "fine_tuned_bert_role_classifier"
    AUDIO_FOLDER = "wav_folder"
    OUTPUT_FOLDER = "output"
    BERT_LABEL_MAP = {0: "operator", 1: "caller", 2: "unknown"}

# ------------------------------------------------------------------
# CUDA failsafe helpers (unchanged)
# ------------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_load_whisper_model(model_name):
    model = whisper.load_model(model_name, device="cpu")
    return model, torch.device("cpu")

def safe_load_pyannote_pipeline(token):
    device = get_device()
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        pipe.to(device)
        return pipe, device
    except Exception as e:
        print(f"[!] Error loading PyAnnote pipeline: {e}, falling back to CPU")
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        pipe.to(torch.device("cpu"))
        return pipe, torch.device("cpu")

def safe_load_bert_model(path):
    device = get_device()
    torch_dtype = torch.float16 if device.type == 'cpu' else 'auto'
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(path)
        model = DistilBertForSequenceClassification.from_pretrained(path, num_labels=3, torch_dtype=torch_dtype)
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        print(f"[!] Error loading BERT model: {e}, falling back to CPU without fp16")
        tokenizer = DistilBertTokenizer.from_pretrained(path)
        model = DistilBertForSequenceClassification.from_pretrained(path, num_labels=3, torch_dtype=torch.float32)
        model.to(torch.device("cpu"))
        return tokenizer, model, torch.device("cpu")

def safe_load_hf_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if device == -1 else None
    try:
        return hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
            device=device,
            torch_dtype=torch_dtype
        ), device
    except Exception as e:
        print(f"[!] Error loading HF pipeline: {e}, falling back to CPU without fp16")
        return hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
            device=-1
        ), -1

# ------------------------------------------------------------------
# Helper functions (unchanged)
# ------------------------------------------------------------------
def classify_utterance_bert(text, tokenizer, model, device):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        max_prob, pred = torch.max(probs, dim=1)
        return Config.BERT_LABEL_MAP[pred.item()] if max_prob.item() > 0.7 else None
    except Exception as e:
        print(f"[!] Error in BERT classification: {e}")
        return None

def segment_emotion(text, emo_pipe):
    try:
        result = emo_pipe(text)
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "label" in result[0]:
            return result[0]["label"].lower()
        else:
            print(f"[!] Unexpected result from emotion pipeline for text '{text}': {result}")
            return "neutral"
    except Exception as e:
        print(f"[!] Error in emotion detection for text '{text}': {e}")
        return "neutral"

def find_best_matching_speaker(start, end, diarization_result):
    best_overlap, best_speaker = 0, "Unknown"
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        overlap = max(0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, speaker
    return best_speaker

def sentence_split(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

# ------------------------------------------------------------------
# Worker function (unchanged)
# ------------------------------------------------------------------
def process_audio_file(audio_indices, shared_data, overwrite=False, gpu_id=None):
    """
    Process a list of audio files into .txt transcripts.
    Automatically chooses GPU if available / sufficient VRAM, else CPU.
    """
    # ---------- GPU / CPU helper ----------
    def _pick_gpu():
        if not torch.cuda.is_available():
            return None
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free >= 2 * 1024**3:        # ≥ 2 GB free
                return i
        return None

    # ---------- Reproducible RNG ----------
    def _set_seed(seed=42):
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ---------- Pin worker to device ----------
    if gpu_id is None:
        gpu_id = _pick_gpu()
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        _set_seed()
        print(f"[+] Worker pinned to GPU {gpu_id}")
    else:
        _set_seed()
        print("[+] Worker running on CPU")

    # ---------- Worker body (unchanged) ----------
    process = psutil.Process()
    torch.set_num_threads(mp.cpu_count())
    print(f"[+] Set torch threads to {mp.cpu_count()} for worker {mp.current_process().name}")
    print(f"[+] Initial memory usage: {process.memory_info().rss / 1024**3:.2f} GB (rank {mp.current_process().name})")
    start_time = time.time()
    successfully_processed = []

    # Load models once per worker
    try:
        whisper_model, _ = safe_load_whisper_model(Config.WHISPER_MODEL)
        pipeline, _ = safe_load_pyannote_pipeline(Config.HF_TOKEN)
        emo_pipe, _ = safe_load_hf_pipeline()
        bert_tokenizer, bert_model, bert_device = safe_load_bert_model(Config.FINETUNED_BERT_MODEL_PATH)
    except Exception as e:
        print(f"[!] Error initializing models for worker {mp.current_process().name}: {e}")
        return []

    # Access shared cues
    CALLER_CUES = shared_data["CALLER_CUES"]
    OPERATOR_CUES = shared_data["OPERATOR_CUES"]

    # Emotion → role rules
    CALLER_EMOS = {"fear", "anger", "sadness", "surprise"}
    OPERATOR_EMOS = {"neutral", "calm"}

    for audio_index in audio_indices:
        out_path = os.path.join(Config.OUTPUT_FOLDER, f"{audio_index}.txt")
        audio_file = os.path.join(Config.AUDIO_FOLDER, f"{audio_index}.wav")

        if os.path.exists(out_path) and not overwrite:
            print(f"[!] Transcript exists, skipping: {out_path} (rank {mp.current_process().name})")
            continue
        if not os.path.exists(audio_file):
            print(f"[!] Audio file missing, skipping: {audio_file} (rank {mp.current_process().name})")
            continue

        print(f"[=>> Processing: {audio_file} (rank {mp.current_process().name}) ]")
        file_start_time = time.time()

        try:
            transcription = whisper_model.transcribe(
                audio_file, language="en", word_timestamps=True,
                condition_on_previous_text=False, verbose=False
            )
            diarization = pipeline(audio_file)
        except Exception as e:
            print(f"[!] Error in transcription or diarization for {audio_file}: {e} (rank {mp.current_process().name})")
            continue

        segments = transcription["segments"]
        speaker_segments = defaultdict(list)
        first_utt_time = {}
        for seg in segments:
            start_t, end_t = seg["start"], seg["end"]
            spk = find_best_matching_speaker(start_t, end_t, diarization)
            speaker_segments[spk].append(seg["text"])
            first_utt_time.setdefault(spk, start_t)

        speaker_scores = []
        for spk, segs in speaker_segments.items():
            question_words = {"what", "where", "when", "why", "who", "how", "do", "does", "can", "is", "are", "911"}
            q_count = sum(any(word.lower() in question_words for word in s.split()) for s in segs)
            speaker_scores.append((spk, first_utt_time[spk], q_count))
        speaker_scores.sort(key=lambda x: (x[1], -x[2]))
        role_map = {}
        if speaker_scores:
            role_map[speaker_scores[0][0]] = "operator"
        if len(speaker_scores) > 1:
            role_map[speaker_scores[1][0]] = "caller"

        merged, buffer_txt, buffer_start, buffer_end = [], [], None, None
        prev_role, prev_spk = None, None

        for seg in segments:
            s, e, txt = seg["start"], seg["end"], seg["text"].strip()
            spk = find_best_matching_speaker(s, e, diarization)
            bert_role = classify_utterance_bert(txt, bert_tokenizer, bert_model, bert_device)
            emotion = segment_emotion(txt, emo_pipe)
            txt_lower = txt.lower()

            if bert_role in ["operator", "caller"]:
                role = bert_role
            elif any(cue in txt_lower for cue in CALLER_CUES):
                role = "caller"
            elif any(cue in txt_lower for cue in OPERATOR_CUES):
                role = "operator"
            elif emotion in CALLER_EMOS and any(p in txt_lower for p in {"i", "my", "he", "she"}):
                role = "caller"
            elif emotion in OPERATOR_EMOS:
                role = "operator"
            else:
                role = role_map.get(spk, "unknown")

            new_turn = (role != prev_role) or (buffer_end and s - buffer_end > 1.5) or (spk != prev_spk)
            if new_turn and buffer_txt:
                all_sentences = sentence_split(" ".join(buffer_txt))
                timestamp = str(timedelta(seconds=int(buffer_start)))
                for sent in all_sentences:
                    if sent.strip():
                        merged.append(f"[{timestamp}] {prev_role}: {sent.strip()}")
                buffer_txt, buffer_start = [], None

            if not buffer_txt:
                buffer_start = s
            buffer_txt.append(txt)
            buffer_end, prev_role, prev_spk = e, role, spk

        if buffer_txt:
            timestamp = str(timedelta(seconds=int(buffer_start)))
            for sent in sentence_split(" ".join(buffer_txt)):
                if sent.strip():
                    merged.append(f"[{timestamp}] {prev_role}: {sent.strip()}")

        # Save transcript immediately
        try:
            os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
            with open(out_path, "w") as f:
                f.write("\n".join(merged))
            print(f"[✓] Saved transcript to: {out_path} (rank {mp.current_process().name})")
            successfully_processed.append(audio_index)
        except Exception as e:
            print(f"[!] Error saving transcript to {out_path}: {e} (rank {mp.current_process().name})")

        file_time = time.time() - file_start_time
        print(f"[+] Time for {audio_file}: {file_time:.2f} seconds")
        print(f"[+] Memory usage after processing {audio_file}: {process.memory_info().rss / 1024**3:.2f} GB (rank {mp.current_process().name})")

        # Partial cleanup
        del transcription, diarization, segments, speaker_segments, first_utt_time
        del speaker_scores, role_map, merged, buffer_txt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[+] Partial memory cleared after {audio_file}: {process.memory_info().rss / 1024**3:.2f} GB (rank {mp.current_process().name})")

    # Final cleanup
    del whisper_model, pipeline, emo_pipe, bert_tokenizer, bert_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[+] Final memory cleared for worker {mp.current_process().name}: {process.memory_info().rss / 1024**3:.2f} GB")

    total_time = time.time() - start_time
    print(f"[+] Total time for rank {mp.current_process().name}: {total_time:.2f} seconds")
    return successfully_processed

# ------------------------------------------------------------------
# Post-finetuning bookkeeping helpers (NEW)
# ------------------------------------------------------------------
_FINETUNE_STATE_FILE = "last_finetune.json"
_FINETUNE_CHUNK      = 50          # trigger every 50 new transcripts

def _load_finetune_state():
    if os.path.exists(_FINETUNE_STATE_FILE):
        with open(_FINETUNE_STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_idx": 0}

def _save_finetune_state(state):
    with open(_FINETUNE_STATE_FILE, "w") as f:
        json.dump(state, f)

def _run_post_finetune(new_indices):
    """
    Launch finetune_bert.py on the next _FINETUNE_CHUNK unseen indices.
    Blocks until training is complete.
    """
    if not new_indices:
        return

    state = _load_finetune_state()
    last_used = state["last_idx"]

    # indices that have *never* been seen by BERT so far
    unseen = [idx for idx in new_indices if idx > last_used]

    while len(unseen) >= _FINETUNE_CHUNK:
        chunk = unseen[:_FINETUNE_CHUNK]
        print(f"[POST-FINETUNE] Triggering training on {len(chunk)} new files: {chunk}")
        cmd = [sys.executable, "finetune_bert.py", json.dumps(chunk)]
        subprocess.run(cmd, check=True)          # blocks until done
        state["last_idx"] = max(chunk)
        _save_finetune_state(state)
        unseen = unseen[_FINETUNE_CHUNK:]

# ------------------------------------------------------------------
# transcribe_batch (unchanged except final two lines)
# ------------------------------------------------------------------
def transcribe_batch(audio_indices, overwrite=False, num_workers_hint=None):
    mp.set_start_method('spawn', force=True)
    torch.set_num_threads(mp.cpu_count())
    job_start_time = time.time()
    print(f"[+] Transcription batch started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if num_workers_hint is not None and num_workers_hint > 0:
        world_size = min(num_workers_hint, 20)
        print(f"[+] Using num_workers_hint: {num_workers_hint}, capped to {world_size} workers.")
    else:
        fallback_count = mp.cpu_count()
        world_size = min(fallback_count, 10)
        print(f"[+] No num_workers_hint provided or invalid. Using mp.cpu_count(): {fallback_count}, capped to {world_size} workers.")

    try:
        with open("data.json", "r") as f:
            cue_data = json.load(f)
    except FileNotFoundError:
        print("[!] Error: data.json not found")
        cue_data = {"CALLER_CUES": [], "OPERATOR_CUES": []}
    except Exception as e:
        print(f"[!] Error loading data.json: {e}")
        cue_data = {"CALLER_CUES": [], "OPERATOR_CUES": []}

    manager = Manager()
    shared_data = manager.dict()
    shared_data["CALLER_CUES"] = set(cue_data.get("CALLER_CUES", []))
    shared_data["OPERATOR_CUES"] = set(cue_data.get("OPERATOR_CUES", []))

    try:
        aug = naw.SynonymAug(aug_src='wordnet')
        for cue in list(shared_data["CALLER_CUES"]):
            try:
                augmented = aug.augment(cue.lower())
                if isinstance(augmented, list):
                    shared_data["CALLER_CUES"].add(augmented[0])
                else:
                    shared_data["CALLER_CUES"].add(augmented)
            except:
                pass
        for cue in list(shared_data["OPERATOR_CUES"]):
            try:
                augmented = aug.augment(cue.lower())
                if isinstance(augmented, list):
                    shared_data["OPERATOR_CUES"].add(augmented[0])
                else:
                    shared_data["OPERATOR_CUES"].add(augmented)
            except:
                pass
    except Exception as e:
        print(f"[!] Error augmenting cues: {e}")

    print(f"[+] CALLER_CUES count: {len(shared_data['CALLER_CUES'])}")
    print(f"[+] OPERATOR_CUES count: {len(shared_data['OPERATOR_CUES'])}")

    if not audio_indices:
        print("[!] No audio files to process in this batch. Exiting.")
        return []

    print(f"[+] Processing {len(audio_indices)} audio files in this batch.")

    print(f"[+] Creating process pool with {world_size} workers...")
    indices_per_process = [[] for _ in range(world_size)]
    for i, idx in enumerate(sorted(audio_indices)):
        indices_per_process[i % world_size].append(idx)
    print(f"[+] Assigned indices per process: {[(i, indices) for i, indices in enumerate(indices_per_process) if indices]}")

    pool = mp.Pool(processes=world_size)
    results = [pool.apply_async(process_audio_file, args=(indices, shared_data, overwrite)) for indices in indices_per_process if indices]
    successful_indices = []
    for r in results:
        try:
            successful_indices.extend(r.get())
        except Exception as e:
            print(f"[!] Error in worker: {e}")
    pool.close()
    pool.join()

    missing = [idx for idx in audio_indices if idx not in successful_indices]
    if missing:
        print(f"[!] Missing transcripts for indices in batch: {missing}")
    else:
        print("[✓] All transcripts in batch are present.")

    del shared_data, manager, cue_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------ POST-FINETUNE TRIGGER (NEW) ------------
    _run_post_finetune(successful_indices)
    # -----------------------------------------------------

    job_time = time.time() - job_start_time
    print(f"\nBatch transcription done. Time: {job_time:.2f} seconds.")
    return successful_indices

# ------------------------------------------------------------------
# Entry point (unchanged)
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_indices = json.loads(sys.argv[1])
    else:
        # Load all indices from folder if no arg provided
        audio_files = [f for f in os.listdir(Config.AUDIO_FOLDER) if f.endswith('.wav')]
        audio_indices = [int(f.split('.')[0]) for f in audio_files if f.split('.')[0].isdigit()]
        audio_indices.sort()
    transcribe_batch(audio_indices)
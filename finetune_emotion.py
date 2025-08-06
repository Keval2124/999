# finetune_emotion.py
import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,

    DataCollatorWithPadding,
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)
import json
import numpy as np
import torch.nn as nn
import gc
import nlpaug.augmenter.word as naw
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import os
import re
import torch.multiprocessing as mp
import sys
import nltk
import logging
from concurrent.futures import ThreadPoolExecutor
import random
import mlflow
import socket
from accelerate import Accelerator
import shutil
from collections import defaultdict, Counter

# --- ADD YOUR TOKEN HERE ---
HF_TOKEN = ""
# --- END ADDITION ---

# NEW -------------------------------------------------
mlflow.set_experiment("emotion_finetune_experiment")

# create **one** global run the **first** time the file is imported
if not hasattr(mlflow, "_GLOBAL_RUN"):
    mlflow._GLOBAL_RUN = mlflow.start_run()

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Append custom NLTK data path
custom_nltk_dir = os.path.abspath('nltk_data')
nltk.data.path.append(custom_nltk_dir)
tagger_path = os.path.join(custom_nltk_dir, 'taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
if not os.path.exists(tagger_path):
    logger.warning("[!] NLTK tagger missing. Download on vis node first.")
    sys.exit(1)

# Detect GPU and set environment accordingly
use_gpu = torch.cuda.is_available()
if use_gpu:
    os.environ["ACCELERATE_USE_CPU"] = "false"  # Enable GPU
    logger.info("GPU detected and enabled.")
else:
    os.environ["ACCELERATE_USE_CPU"] = "true"  # Fallback to CPU
    logger.info("No GPU detected; falling back to CPU.")

# -----------------------------------------------------------
# 1.  NEW UTILITY: download/cache base emotion model once
# -----------------------------------------------------------
def cache_base_emotion_model(model_id="j-hartmann/emotion-english-distilroberta-base",
                             local_dir="./local_emotion_model",
                             token=None):
    """Download (if necessary) and cache the base emotion model once."""
    if not os.path.isdir(local_dir):
        logger.info(f"[CACHE] Downloading {model_id} -> {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, use_auth_token=token
        )
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        logger.info("[CACHE] Download complete.")
    else:
        logger.info("[CACHE] Model already present locally.")

def load_data_from_batch_indices(batch_indices, output_folder="output"):
    data = []
    for idx in batch_indices:
        file_path = os.path.join(output_folder, f"{idx}.txt")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r'\[(\d+:\d+:\d+)\]\s*(\w+):\s*(.*)', line)
                    if match:
                        _, speaker, text = match.groups()
                        speaker_lower = speaker.lower()
                        if text and speaker_lower in ["operator", "caller", "other", "unknown"]:
                            role = "other" if speaker_lower == "unknown" else speaker_lower
                            data.append((text, role))
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        else:
            logger.warning(f"[!] File not found: {file_path}")
    logger.info(f"Loaded {len(data)} samples from batch files")
    return data

def load_data_json_as_data():
    data_json = []
    try:
        with open("data.json", "r") as f:
            cues = json.load(f)
        for cue in cues.get("CALLER_CUES", []):
            data_json.append((cue, "caller"))
        for cue in cues.get("OPERATOR_CUES", []):
            data_json.append((cue, "operator"))
        logger.info(f"Loaded {len(data_json)} samples from data.json")
    except FileNotFoundError:
        logger.error("[!] data.json not found")
    except Exception as e:
        logger.error(f"[!] Error loading data.json: {e}")
    return data_json

def load_all_output_data(output_folder="output"):
    data = []
    files = [f for f in os.listdir(output_folder) if f.endswith(".txt") and f[:-4].isdigit()]
    logger.info(f"Found {len(files)} output files to load")
    for filename in files:
        file_path = os.path.join(output_folder, filename)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'\[(\d+:\d+:\d+)\]\s*(\w+):\s*(.*)', line)
                if match:
                    _, speaker, text = match.groups()
                    speaker_lower = speaker.lower()
                    if text and speaker_lower in ["operator", "caller", "other", "unknown"]:
                        role = "other" if speaker_lower == "unknown" else speaker_lower
                        data.append((text, role))
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue
    logger.info(f"Loaded {len(data)} samples from all output files")
    return data

def load_additional_dialogs(hf_file="additional_911_dialogs.txt"):
    data = []
    dialogs = []  # We'll return both flat data and grouped dialogs
    if os.path.exists(hf_file):
        try:
            with open(hf_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Split on double newlines or '--' lines to separate situations/paragraphs
                blocks = re.split(r'\n\s*\n|\n\s*--\s*\n', content)
                for block in blocks:
                    block = block.strip()
                    if not block:
                        continue
                    current_dialog = []
                    lines = block.splitlines()
                    for line in lines:
                        line = line.strip()
                        if not line or line == '--':
                            continue
                        if ':' in line:
                            speaker, text = line.split(':', 1)
                            speaker = speaker.strip().lower()
                            text = text.strip()
                            if text and speaker in ["operator", "caller", "other", "unknown"]:
                                role = "other" if speaker == "unknown" else speaker
                                data.append((text, role))
                                current_dialog.append((text, role))
                    if current_dialog:
                        dialogs.append(current_dialog)
            logger.info(f"Loaded {len(data)} samples from {hf_file}")
        except Exception as e:
            logger.error(f"[!] Error loading {hf_file}: {e}")
    else:
        logger.warning(f"[!] {hf_file} not found, skipping additional dialogs.")
    return data, dialogs

def augment_text(text, label, augmenter, num_aug):
    augmented = [(text, label)]
    for _ in range(num_aug):
        try:
            aug_text_list = augmenter.augment(text)
            if aug_text_list and isinstance(aug_text_list, list):
                augmented.append((aug_text_list[0], label))
        except Exception as e:
            logger.warning(f"Augmentation error for '{text}': {e}")
    return augmented

def get_augmenter():
    return naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased', action='substitute', top_k=5
    )

class CustomEarlyStopCallback(TrainerCallback):
    def __init__(self, patience=2, max_lr=5e-5):
        self.patience = patience
        self.wait = 0
        self.best_loss = float("inf")
        self.max_lr = max_lr
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get("metrics", {})
        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.wait = 0
            else:
                self.wait += 1
                logger.info(f"[EarlyStop] Eval loss increased. Wait = {self.wait}/{self.patience}")
                if self.wait >= self.patience:
                    logger.info("[EarlyStop] Stopping training due to overfitting.")
                    control.should_training_stop = True
        return control

class MLflowLoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            mlflow.log_metrics(logs, step=state.global_step)

def clear_memory():
    gc.collect()
    if use_gpu:
        torch.cuda.empty_cache()

def finetune_main(batch_indices=None):
    # Use all available cores for fine-tuning
    total_cores = mp.cpu_count()
    torch.set_num_threads(total_cores)
    logger.info(f"Using {total_cores} cores for fine-tuning")

    # Initialize accelerator for distributed training if needed
    accelerator = Accelerator()
    if accelerator.is_main_process:
        logger.info("Starting data loading...")
    
    # Load data from this specific batch
    if batch_indices is None:
        if accelerator.is_main_process:
            logger.warning("[!] No batch indices provided, loading all output files...")
        batch_data = load_all_output_data()
    else:
        batch_data = load_data_from_batch_indices(batch_indices)
    
    # Load additional data from JSON
    data_json = load_data_json_as_data()
    # Load additional dialogs (now returns flat data and grouped dialogs)
    additional_data, dialogs = load_additional_dialogs()
    
    all_data = data_json + batch_data + additional_data  # list of (text, role)
    
    if not all_data:
        if accelerator.is_main_process:
            logger.error("[!] No data loaded. Ensure data.json has content or output files exist. Exiting.")
        return
    
    if accelerator.is_main_process:
        logger.info(f"Loaded total {len(all_data)} samples (text, role) for pseudo-labeling")
    
    # -----------------------------------------------------------
    # Ensure the base model is cached locally
    # -----------------------------------------------------------
    cache_base_emotion_model(token=HF_TOKEN)

    # -----------------------------------------------------------
    # Load pipeline from the local copy for emotion pseudo-labeling
    # -----------------------------------------------------------
    base_emo_pipe = hf_pipeline(
        "text-classification",
        model="./local_emotion_model",
        tokenizer="./local_emotion_model",
        top_k=None
    )
    
    # -----------------------------------------------------------
    # Pseudo-labeling with emotions and causes (grouped into dialogs)
    # -----------------------------------------------------------
    # Use dialogs from load_additional_dialogs; for simplicity, group all_data into pseudo-dialogs if needed
    if not dialogs:
        dialogs = [all_data]  # Treat as one big dialog if not grouped

    labeled_dialogs = []
    role_emotions = defaultdict(list)
    for dialog_id, dialog in enumerate(dialogs):
        texts = [text for text, role in dialog]
        roles = [role for text, role in dialog]
        try:
            results = base_emo_pipe(texts, batch_size=16)
            emotions = [result[0]["label"].lower() for result in results]  # Top emotion
            # Naive causes: Previous utterance as cause for non-neutral emotions
            causes = [""] * len(texts)
            cause_spans = [""] * len(texts)
            for i in range(1, len(texts)):
                if emotions[i] != "neutral":
                    causes[i] = texts[i-1]
                    cause_spans[i] = texts[i-1]  # Full previous as span
            for emo, cause, span, text, role in zip(emotions, causes, cause_spans, texts, roles):
                if emo != "neutral":
                    role_emotions[role].append(emo)
            labeled_dialogs.append(list(zip(texts, emotions, causes, cause_spans, roles)))
        except Exception as e:
            logger.error(f"Pseudo-labeling failed for dialog {dialog_id}: {e}")

    # Log emotion statistics per role
    if accelerator.is_main_process:
        for role, emos in role_emotions.items():
            counter = Counter(emos)
            logger.info(f"Emotions for {role}: {counter.most_common()}")
            # Optionally save to file
            with open(f"{role}_emotions.json", "w") as f:
                json.dump(dict(counter), f)
    
    if not labeled_dialogs:
        if accelerator.is_main_process:
            logger.error("[!] No labeled data after pseudo-labeling. Exiting.")
        return
    
    if accelerator.is_main_process:
        logger.info(f"Total dialogs after pseudo-labeling: {len(labeled_dialogs)}")
        logger.info("Starting augmentation...")
    
    augmenter = get_augmenter()
    augmented_data = []
    num_aug = 1
    with ThreadPoolExecutor(max_workers=total_cores) as executor:
        # Augment flat for simplicity, but only emotions
        flat_labeled = [(text, emo) for dialog in labeled_dialogs for text, emo, _, _, _ in dialog if emo != "neutral"]
        futures = [executor.submit(augment_text, item[0], item[1], augmenter, num_aug) for item in flat_labeled]
        for future in futures:
            augmented_data.extend(future.result())
    
    if accelerator.is_main_process:
        logger.info(f"Total samples after augmentation: {len(augmented_data)}")
    
    if not augmented_data:
        if accelerator.is_main_process:
            logger.error("[!] No data after augmentation. Exiting.")
        return
    
    clear_memory()  # Clear after augmentation
    
    # -----------------------------------------------------------
    # Prepare dataset for ECPE as QA span extraction
    # For each non-neutral utterance, create QA example: question = "What is the cause of [emotion] in [utterance]?", context = previous utterances concatenated
    # -----------------------------------------------------------
    qa_examples = []
    for dialog in labeled_dialogs:
        utterances = [f"{role.capitalize()}: {text}" for text, _, _, _, role in dialog]
        for i in range(len(dialog)):
            emo = dialog[i][1]
            if emo == "neutral":
                continue
            target_utterance = utterances[i]
            context = ' '.join(utterances[:i])  # History
            question = f"What is the cause of the emotion {emo} in this utterance: {target_utterance}?"
            answer_text = dialog[i][3]  # cause_span
            if not answer_text:
                continue
            # Find char start and end in context
            start_char = context.find(answer_text)
            if start_char == -1:
                continue  # Skip if not found
            end_char = start_char + len(answer_text) - 1
            qa_examples.append({
                "question": question,
                "context": context,
                "answers": {"text": [answer_text], "answer_start": [start_char]}
            })
    
    if not qa_examples:
        if accelerator.is_main_process:
            logger.error("[!] No QA examples generated. Exiting.")
        return

    dataset = Dataset.from_list(qa_examples)
    dataset = dataset.train_test_split(test_size=0.2)
    
    clear_memory()  # Clear after dataset creation
    
    if accelerator.is_main_process:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base", use_auth_token=HF_TOKEN)
        logger.info("Tokenizer loaded successfully.")
    
    def preprocess_qa(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = inputs["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            answer = answers[i]
            if len(answer["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0]) - 1
            token_start_index = 0
            while offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            token_end_index += 1

            # Detect if the answer is out of the span (in which case label as CLS)
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char + 1):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_positions.append(token_start_index - 1)
                end_positions.append(token_end_index + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    encoded = dataset.map(preprocess_qa, batched=True, remove_columns=dataset["train"].column_names, num_proc=total_cores)
    
    clear_memory()  # Clear after preprocessing
    
    model_path = "./fine_tuned_emotion_cause_model"
    temp_model_path = "./temp_fine_tuned_emotion_cause"
    
    if os.path.exists(model_path):
        if accelerator.is_main_process:
            logger.info("Loading from previous fine-tuned model...")
        model = AutoModelForQuestionAnswering.from_pretrained(model_path, use_auth_token=HF_TOKEN)
    else:
        if accelerator.is_main_process:
            logger.info("Starting from pre-trained model...")
        model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2", use_auth_token=HF_TOKEN)  # Use a QA-pretrained RoBERTa
    
    args = TrainingArguments(
        output_dir=temp_model_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8 if use_gpu else 2,  # Smaller for QA
        per_device_eval_batch_size=16 if use_gpu else 4,
        num_train_epochs=3,  # Fewer epochs for QA
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="mlflow",
        logging_dir="./logs",
        logging_steps=10,
        fp16=use_gpu,
        dataloader_num_workers=total_cores // 2,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if accelerator.is_main_process:
        mlflow.log_params({
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "weight_decay": args.weight_decay,
            "batch_size": len(batch_indices) if batch_indices else 0,
        })
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[CustomEarlyStopCallback(patience=2, max_lr=5e-5), MLflowLoggingCallback()],
    )
    
    if accelerator.is_main_process:
        logger.info("Starting emotion-cause model training...")
    trainer.train()
    
    clear_memory()  # Clear after training
    
    if accelerator.is_main_process:
        trainer.save_model(temp_model_path)
        tokenizer.save_pretrained(temp_model_path)
        shutil.rmtree(model_path, ignore_errors=True)
        os.rename(temp_model_path, model_path)
        
        logger.info("Evaluating emotion-cause model...")
        # For evaluation, use trainer.evaluate() or custom metrics for QA
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        
        mlflow.log_artifacts(model_path, artifact_path="emotion_cause_model")
    
    del model, trainer, encoded, dataset, tokenizer
    clear_memory()  # Final clear

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if len(sys.argv) > 1:
        try:
            batch_indices = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in sys.argv[1]: {sys.argv[1]}")
            sys.exit(1)
        if isinstance(batch_indices, int):
            batch_indices = [batch_indices]
    else:
        batch_indices = None
    finetune_main(batch_indices)
    gc.collect()
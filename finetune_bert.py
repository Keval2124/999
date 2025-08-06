# finetune_bert.py
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DistilBertConfig,
    DistilBertPreTrainedModel,
    DistilBertModel,
    DataCollatorWithPadding,
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
from accelerate import Accelerator
import shutil

# --- ADD YOUR TOKEN HERE ---
HF_TOKEN = "" # Replace with your actual token
# --- END ADDITION ---

# NEW -------------------------------------------------
mlflow.set_experiment("bert_finetune_experiment")
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

# --- REMOVED THE LINE THAT FORCED CPU ---
# os.environ["ACCELERATE_USE_CPU"] = "true"

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
                            label = "other" if speaker_lower == "unknown" else speaker_lower
                            data.append((text, label))
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
                        label = "other" if speaker_lower == "unknown" else speaker_lower
                        data.append((text, label))
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue
    logger.info(f"Loaded {len(data)} samples from all output files")
    return data

def load_additional_dialogs(hf_file="additional_911_dialogs.txt"):
    data = []
    if os.path.exists(hf_file):
        try:
            with open(hf_file, "r", encoding="utf-8") as f:
                hf_raw = f.read().strip().split("\n") # Fixed potential split issue
            for dialog in hf_raw:
                if not dialog.strip():
                    continue
                lines = dialog.splitlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if ':' in line:
                        speaker, text = line.split(':', 1)
                        speaker = speaker.strip().lower()
                        text = text.strip()
                        if text and speaker in ["operator", "caller", "other", "unknown"]:
                            label = "other" if speaker == "unknown" else speaker
                            data.append((text, label))
            logger.info(f"Loaded {len(data)} samples from {hf_file}")
        except Exception as e:
            logger.error(f"[!] Error loading {hf_file}: {e}")
    else:
        logger.warning(f"[!] {hf_file} not found, skipping additional dialogs.")
    return data

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

class WeightedDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Ensure class_weights is a tensor on the correct device eventually
        self.register_buffer('class_weights', class_weights if class_weights is not None else torch.ones(config.num_labels))
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # Loss calculation using the buffered weights
        loss = None
        if labels is not None:
             loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

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

def finetune_main(batch_indices=None):
    # Use all available cores for fine-tuning
    total_cores = mp.cpu_count()
    torch.set_num_threads(total_cores)
    logger.info(f"Using {total_cores} cores for fine-tuning (PyTorch internal parallelism)")

    # Initialize accelerator for distributed training if needed
    accelerator = Accelerator()
    if accelerator.is_main_process:
        logger.info("Starting data loading...")

    # --- DEVICE DETECTION AND FALLBACK LOGIC ---
    # Determine the best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA is available. Attempting to use GPU...")
        use_cpu_arg = False
        # Suggest using fp16 for GPU
        suggested_fp16 = True
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")
        use_cpu_arg = True
        suggested_fp16 = False # fp16 generally not beneficial on CPU

    # Load data from this specific batch
    if batch_indices is None:
        if accelerator.is_main_process:
            logger.warning("[!] No batch indices provided, loading all output files...")
        batch_data = load_all_output_data()
    else:
        batch_data = load_data_from_batch_indices(batch_indices)

    # Load additional data from JSON
    data_json = load_data_json_as_data()

    # Load additional dialogs
    additional_data = load_additional_dialogs()

    data = data_json + batch_data + additional_data

    if not data:
        if accelerator.is_main_process:
            logger.error("[!] No data loaded. Ensure data.json has content or output files exist. Exiting.")
        return

    if accelerator.is_main_process:
        logger.info(f"Loaded total {len(data)} samples for fine-tuning")
        logger.info("Starting augmentation...")

    augmenter = get_augmenter()
    augmented_data = []
    num_aug = 1
    with ThreadPoolExecutor(max_workers=total_cores) as executor:
        futures = [executor.submit(augment_text, item[0], item[1], augmenter, num_aug) for item in data]
        for future in futures:
            augmented_data.extend(future.result())

    if accelerator.is_main_process:
        logger.info(f"Total samples after augmentation: {len(augmented_data)}")

    if not augmented_data:
        if accelerator.is_main_process:
            logger.error("[!] No data after augmentation. Exiting.")
        return

    texts, labels = zip(*augmented_data)
    dataset = Dataset.from_dict({"text": texts, "label": labels})

    from datasets import ClassLabel
    label_feature = ClassLabel(num_classes=3, names=["operator", "caller", "other"])
    dataset = dataset.cast_column("label", label_feature)
    dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")

    if accelerator.is_main_process:
        logger.info("Training Random Forest baseline...")
        train_texts = [ex['text'] for ex in dataset['train']]
        train_labels = [ex['label'] for ex in dataset['train']]
        test_texts = [ex['text'] for ex in dataset['test']]
        test_labels = [ex['label'] for ex in dataset['test']]
        unique_test_labels_int = sorted(list(set(test_labels)))
        target_names_for_report = [label_feature.int2str(label_int) for label_int in unique_test_labels_int]
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, train_labels)
        rf_preds = rf.predict(X_test)
        logger.info("Random Forest Classification Report:")
        try:
            report_str = classification_report(
                test_labels,
                rf_preds,
                labels=unique_test_labels_int,
                target_names=target_names_for_report,
                zero_division=0
            )
            logger.info(report_str)
        except ValueError as e:
            logger.error(f"Error generating RF classification report: {e}")
        del rf, vectorizer, X_train, X_test
        gc.collect()

    if accelerator.is_main_process:
        logger.info("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", use_auth_token=HF_TOKEN)

    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, padding=False, max_length=128) # Let DataCollator handle padding
        enc["labels"] = batch["label"]
        return enc

    # Use remove_columns to avoid keeping raw text in memory
    encoded = dataset.map(preprocess, batched=True, remove_columns=["text"])

    # Class weights calculation
    all_classes = label_feature.names
    num_classes = len(all_classes)
    y_int = np.array(dataset["train"]["label"])
    n_samples = len(y_int)
    counts = np.bincount(y_int, minlength=num_classes)
    class_weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            class_weights[c] = n_samples / (num_classes * counts[c])
        else:
            class_weights[c] = 1.0
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    model_path = "./fine_tuned_bert_role_classifier"
    temp_model_path = "./temp_fine_tuned_bert"

    # --- MODEL LOADING WITH FALLBACK ---
    model_loaded = False
    model = None
    config = None

    # Determine torch_dtype based on the target device *before* loading
    torch_dtype_arg = torch.float16 if device.type == 'cuda' and suggested_fp16 else 'auto'
    logger.info(f"Using torch_dtype: {torch_dtype_arg} for model loading.")

    # Attempt to load model (potentially onto GPU)
    if os.path.exists(model_path):
        if accelerator.is_main_process:
            logger.info("Attempting to load from previous fine-tuned model...")
        try:
            # Load config first
            config = DistilBertConfig.from_pretrained(model_path, use_auth_token=HF_TOKEN)
            # Attempt to load model with specified dtype
            model = WeightedDistilBertForSequenceClassification.from_pretrained(
                model_path,
                class_weights=class_weights_tensor, # Pass custom class weights tensor
                use_auth_token=HF_TOKEN,
                torch_dtype='auto', # Specify dtype
            )
            # Explicitly try moving to device if not already
            model.to(device)
            model_loaded = True
            logger.info(f"Successfully loaded model from {model_path} onto {device}.")
        except Exception as e: # Catch CUDA OOM or other loading/moving errors
            logger.error(f"Error loading model onto {device} (likely CUDA OOM): {e}")
            if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                 logger.warning("Falling back to CPU due to GPU error.")
                 device = torch.device("cpu")
                 use_cpu_arg = True
                 suggested_fp16 = False
                 torch_dtype_arg = 'auto' # Or torch.float32, 'auto' is safer
                 logger.info(f"Retrying model load with torch_dtype: {torch_dtype_arg} for CPU.")
                 try:
                     # Retry loading for CPU
                     if os.path.exists(model_path):
                         config = DistilBertConfig.from_pretrained(model_path, use_auth_token=HF_TOKEN)
                         model = WeightedDistilBertForSequenceClassification.from_pretrained(
                             model_path,
                             class_weights=class_weights_tensor,
                             use_auth_token=HF_TOKEN,
                             torch_dtype=torch_dtype_arg, # Use appropriate dtype for CPU
                         )
                         # Model should load directly to CPU now
                         model_loaded = True
                         logger.info(f"Successfully loaded model from {model_path} onto CPU after fallback.")
                     else:
                         # Need to load from pre-trained if path doesn't exist anymore or wasn't used
                         config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=3, use_auth_token=HF_TOKEN)
                         model = WeightedDistilBertForSequenceClassification.from_pretrained(
                             "distilbert-base-uncased",
                             config=config,
                             class_weights=class_weights_tensor,
                             use_auth_token=HF_TOKEN,
                             torch_dtype=torch_dtype_arg,
                         )
                         model_loaded = True
                         logger.info("Successfully loaded pre-trained model onto CPU after fallback.")

                 except Exception as cpu_e:
                     logger.error(f"Failed to load model onto CPU after fallback: {cpu_e}")
                     return # Exit if loading fails on CPU too
            else:
                logger.error(f"Non-CUDA error during model loading: {e}")
                return # Exit if it's a different kind of error

    # If loading from existing path failed or path didn't exist, load from pre-trained
    if not model_loaded:
        if accelerator.is_main_process:
            logger.info("Starting from pre-trained model...")
        try:
            config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=3, use_auth_token=HF_TOKEN)
            model = WeightedDistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                config=config,
                class_weights=class_weights_tensor,
                use_auth_token=HF_TOKEN,
                torch_dtype=torch_dtype_arg,
            )
            # Try moving to device (likely GPU if not fallen back)
            model.to(device)
            logger.info(f"Successfully loaded pre-trained model onto {device}.")
        except Exception as e:
            logger.error(f"Error loading pre-trained model onto {device} (likely CUDA OOM): {e}")
            if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                logger.warning("Falling back to CPU for pre-trained model load.")
                device = torch.device("cpu")
                use_cpu_arg = True
                suggested_fp16 = False
                torch_dtype_arg = 'auto' # Or torch.float32
                logger.info(f"Retrying pre-trained model load with torch_dtype: {torch_dtype_arg} for CPU.")
                try:
                    # Retry loading for CPU
                    config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=3, use_auth_token=HF_TOKEN)
                    model = WeightedDistilBertForSequenceClassification.from_pretrained(
                        "distilbert-base-uncased",
                        config=config,
                        class_weights=class_weights_tensor,
                        use_auth_token=HF_TOKEN,
                        torch_dtype=torch_dtype_arg, # Use appropriate dtype for CPU
                    )
                    # Should load directly to CPU
                    logger.info("Successfully loaded pre-trained model onto CPU after fallback.")
                except Exception as cpu_e:
                    logger.error(f"Failed to load pre-trained model onto CPU after fallback: {cpu_e}")
                    return # Exit if loading fails on CPU too
            else:
                 logger.error(f"Non-CUDA error during pre-trained model loading: {e}")
                 return # Exit if it's a different kind of error

    # --- ADJUST TrainingArguments BASED ON FINAL DEVICE ---
    args = TrainingArguments(
        output_dir=temp_model_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="mlflow",
        logging_dir="./logs",
        logging_steps=10,
        fp16=False,
        use_cpu=use_cpu_arg, # Use the determined CPU arg
    )

    # Ensure dataset format is correct for Trainer (let Trainer handle device placement)
    # encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # ^ Not needed if columns are correctly specified and DataCollator handles padding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if accelerator.is_main_process:
        mlflow.log_params({
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "weight_decay": args.weight_decay,
            "batch_size": len(batch_indices) if batch_indices else 0,
            "device_used": device.type, # Log which device was ultimately used
            "fp16_used": args.fp16,
        })

    trainer = Trainer(
        model=model, # Pass the loaded (and potentially moved) model
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        data_collator=data_collator,
        callbacks=[CustomEarlyStopCallback(patience=2, max_lr=5e-5), MLflowLoggingCallback()],
        # tokenizer=tokenizer, # Optional, but good practice if not passed to DataCollator
    )

    if accelerator.is_main_process:
        logger.info("Starting BERT training...")
    trainer.train() # Trainer will handle device placement of batches etc.

    # --- POST-TRAINING ---
    if accelerator.is_main_process:
        # Use Trainer's save method for consistency
        trainer.save_model(temp_model_path)
        tokenizer.save_pretrained(temp_model_path)
        shutil.rmtree(model_path, ignore_errors=True)
        os.rename(temp_model_path, model_path)
        logger.info("Evaluating BERT model...")
        bert_preds = trainer.predict(encoded['test'])
        bert_logits = bert_preds.predictions
        bert_preds = np.argmax(bert_logits, axis=-1)

        # Get the true labels from the test set
        y_true_test = encoded['test']['labels']
        # Find unique labels present in the test set
        unique_labels_in_test = sorted(list(set(y_true_test)))
        # Get the corresponding target names for only those labels
        target_names_for_test = [label_feature.int2str(label_int) for label_int in unique_labels_in_test]

        # Generate the classification report using only the labels present in the test set
        report = classification_report(
            y_true_test,          # True labels from test set
            bert_preds,           # Predicted labels
            labels=unique_labels_in_test,  # Specify the labels present
            target_names=target_names_for_test, # Corresponding names
            output_dict=True,      # For logging metrics
            zero_division=0        # Handle potential division by zero
        )

        # Log metrics (only for classes present, handle potential missing keys)
        mlflow.log_metrics({
            "eval_accuracy": report.get('accuracy', 0.0), # Accuracy is overall
            "eval_precision_macro": report.get('macro avg', {}).get('precision', 0.0),
            "eval_recall_macro": report.get('macro avg', {}).get('recall', 0.0),
            "eval_f1_macro": report.get('macro avg', {}).get('f1-score', 0.0),
        })

        # Generate the string report for logging (using the same dynamic labels/names)
        report_str = classification_report(
            y_true_test,
            bert_preds,
            labels=unique_labels_in_test,
            target_names=target_names_for_test,
            zero_division=0
        )
        logger.info("BERT Classification Report:")
        logger.info(report_str)
        

        report_file = "classification_report.txt"
        with open(report_file, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_file)
        os.remove(report_file)
        mlflow.log_artifacts(model_path, artifact_path="bert_model")

    # Cleanup
    del model, trainer, encoded, dataset, tokenizer
    gc.collect()

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

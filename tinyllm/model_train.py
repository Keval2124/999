import os
import json
import glob
import warnings
import mlflow
import mlflow.transformers  # Add this import
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType

# Set tokenizers parallelism to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------
# 0. Paths
# ------------------------------------------------
BASE_DIR      = "/mnt/scratch/users/sgkshah/999"
TEXT_DIR      = os.path.join(BASE_DIR, "output")
JSON_FILE     = os.path.join(BASE_DIR, "sample.json")
MLFLOW_DIR    = os.path.join(BASE_DIR, "mlruns")
MODEL_DIR     = os.path.join(BASE_DIR, "tinyllm", "tinyllm")   # fallback base
OUTPUT_DIR    = os.path.join(BASE_DIR, "tinyllm-lora")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")

for d in (OUTPUT_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------
# 1. Data loading (999 transcripts)
# ------------------------------------------------
def load_and_augment_data():
    samples = []

    # 1) .txt transcripts ----------------------------------------------------
    for path in glob.glob(os.path.join(TEXT_DIR, "*.txt")):
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        if raw:
            # wrap so model learns start/end
            text = f"<|startoftext|>\n{raw}\n"
            samples.append({"text": text})

    # 2) optional JSON side info --------------------------------------------
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, encoding="utf-8") as f:
            js = json.load(f)
        for item in js:
            name    = item.get("name", "").strip()
            address = item.get("address", "").strip()
            if not (name and address):
                continue
            # Realistic mini-transcript ending with name & address exchange
            text = (
                f"<|startoftext|>\n"
                f"[0:00:00] operator: 999 Emergency, what is your emergency?\n"
                f"[0:00:02] caller: There’s a fire in the kitchen.\n"
                f"[0:00:05] operator: Are you at the address now?\n"
                f"[0:00:07] caller: Yes, I need help!\n"
                f"[0:00:09] operator: May I have your name and address, please?\n"
                f"[0:00:11] caller: My name is {name}. Address: {address}\n"
                f"<|endoftext|>"
            )
            samples.append({"text": text})

    print(f"[DATA] Loaded {len(samples)} transcript samples")
    ds = Dataset.from_list(samples).train_test_split(test_size=0.1)
    return ds["train"], ds["test"]

train_ds, eval_ds = load_and_augment_data()

# ------------------------------------------------
# 2. Tokenizer & model
# ------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=True,
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
base_model.gradient_checkpointing_enable()

# LoRA
lora_conf = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_conf)

# ------------------------------------------------
# 3. Tokenize
# ------------------------------------------------
def tok_fn(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tok_train = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
tok_eval  = eval_ds.map(tok_fn,  batched=True, remove_columns=["text"])

print(f"[TOKEN] train={len(tok_train)}  eval={len(tok_eval)}")

# ------------------------------------------------
# 4. Training args
# ------------------------------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=100,                    # raise ceiling
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    logging_dir=LOGS_DIR,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",      # what to watch
    logging_strategy="steps",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    gradient_checkpointing=True,
    report_to="mlflow",
)

class MLflowMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Compute perplexity from the evaluation loss
            eval_loss = metrics.get("eval_loss", None)
            if eval_loss is not None:
                metrics["eval_perplexity"] = np.exp(eval_loss)
            
            # Log all metrics (including the new eval_perplexity)
            mlflow.log_metrics(metrics, step=state.global_step)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log training loss and any other training metrics
            mlflow.log_metrics(logs, step=state.global_step)
            
class StopOnLrIncreaseCallback(TrainerCallback):
    """Stops training if learning rate increases"""
    def __init__(self):
        self.previous_lr = None
        
    def on_step_end(self, args, state, control, **kwargs):
        # Get current learning rate
        logs = kwargs.get('logs', {})
        current_lr = logs.get('learning_rate', None)
        
        if current_lr is None:
            return
            
        # Check if LR has increased
        if self.previous_lr is not None and current_lr > self.previous_lr:
            print(f"\nLearning rate increased from {self.previous_lr:.2e} to {current_lr:.2e}")
            print("Stopping training early")
            control.should_training_stop = True
            
        # Update previous LR
        self.previous_lr = current_lr

mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("tinyllm_finetuning")


with mlflow.start_run():
    mlflow.log_params({
        "model": "TinyLlama-LoRA",
        "dataset": "999_emergency",
        "lora_r": lora_conf.r,
        "lora_alpha": lora_conf.lora_alpha,
        "lora_dropout": lora_conf.lora_dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "max_length": 512,
        "epochs": args.num_train_epochs
    })

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        processing_class=tokenizer,     # Replaces tokenizer=
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2), 
            MLflowMetricsCallback(), 
            StopOnLrIncreaseCallback()
        ]
    )

    trainer.train()

    # save adapter
    adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(adapter_path)
    
    # log to MLflow using transformers flavor (avoids pickling issues)
    mlflow.transformers.log_model(
    transformers_model={"model": model, "tokenizer": tokenizer},
    name="lora_adapter",
    task="text-generation"  # Add this line
    )

    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(f"runs:/{run_id}/lora_adapter", "tinylama")
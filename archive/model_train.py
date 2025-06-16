import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import glob

# Step 1: Load your dataset
all_text = []
for file_path in glob.glob('output/*.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        all_text.append({"text": f.read()})

dataset = Dataset.from_list(all_text)

# Step 2: Load tokenizer and model
model_name = "tinyllm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Set training arguments
training_args = TrainingArguments(
    output_dir="tinyllm-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_dir="/content/logs",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    fp16=False, # Set to False as we are using CPU
    no_cuda=True, # Explicitly set to True for CPU-only training
    report_to="none"
)

# Step 5: Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Step 6: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 7: Train
trainer.train()

import os
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import mlflow
import mlflow.pytorch

# Config
data_file = 'input.txt'  # Single file with all data
tokenizer_path = 'bpe_tokenizer_v2.json'
vocab_size = 5000
end_of_text_token = "[EOT]"  # Changed from empty string
model_checkpoint_path = 'gpt_model_checkpoint.pth'
best_model_path = 'best_gpt_model.pth'
max_iters = 5000
eval_interval = 1000
eval_iters = 200
learning_rate = 3e-4
device = 'cpu'

# MLflow configuration
BASE_DIR = "/mnt/scratch/users/sgkshah/999"
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("911-call-gpt-training")

# Model hyperparameters
block_size = 256
batch_size = 64
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

def cleaning(text):
    clean_lines = []
    for line in text.split('\n'):
        # Skip name/address lines and empty lines
        if line.strip().startswith('Name:') or not line.strip():
            continue
        # Process call log lines
        match = re.match(r'^\[\d+:\d+:\d+\]\s*(caller|operator|unknown):\s*(.+)$', line.strip(), re.IGNORECASE)
        if match:
            speaker, dialogue = match.groups()
            if len(dialogue.strip()) > 3:
                clean_lines.append(f"{speaker.upper()}: {dialogue}")
        else:
            # Handle lines that don't match the timestamp pattern but might be dialogue
            if ':' in line and len(line.strip()) > 10:
                clean_lines.append(line.strip())
    return "\n".join(clean_lines)

print("Loading and cleaning data...")
with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
    raw_content = f.read()

print("Cleaning data...")
cleaned_content = cleaning(raw_content)
print(f"Raw content length: {len(raw_content)}")
print(f"Cleaned content length: {len(cleaned_content)}")
print(f"First 500 characters of cleaned content:\n{cleaned_content[:500]}")

if not cleaned_content:
    raise ValueError("No valid content found after cleaning. Check your data format.")

# Prepare all text with end of text tokens
all_text = f"\n{end_of_text_token}\n".join([cleaned_content, ''])  # Add empty string to ensure proper joining

def load_tokenizer(text, vocab_size, save_path):
    if os.path.exists(save_path):
        print("Loading existing tokenizer...")
        tokenizer = Tokenizer.from_file(save_path)
    else:
        print("Training new tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]", end_of_text_token]
        )
        tokenizer.train_from_iterator([text], trainer)
        tokenizer.save(save_path)
    return tokenizer

tokenizer = load_tokenizer(all_text, vocab_size, tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)

# Encode data
print("Encoding data...")
data = torch.tensor(encode(all_text), dtype=torch.long)
print(f"Encoded data length: {len(data)}")
print(f"First 20 tokens: {data[:20]}")

if len(data) < block_size:
    raise ValueError(f"Dataset is too small. Need at least {block_size} tokens, but only have {len(data)}.")

# Split data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Training data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")

def get_batch(split):
    d = train_data if split == 'train' else val_data
    
    # Check if we have enough data
    if len(d) <= block_size:
        raise ValueError(f"Dataset size ({len(d)}) is too small for block_size ({block_size})")
    
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class brain(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0, 
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.attn = brain(n_embd, n_head, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class gpt(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_params({
        "data_file": data_file,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "batch_size": batch_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "max_iters": max_iters,
        "learning_rate": learning_rate,
        "device": device,
        "data_length": len(data),
        "train_data_length": len(train_data),
        "val_data_length": len(val_data)
    })
    
    # Log data samples
    mlflow.log_text(cleaned_content[:1000], "data_sample.txt")
    
    # Execution
    print(f"Using device: {device}")
    model = gpt(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    start_iter = 0

    if os.path.exists(model_checkpoint_path):
        print(f"Resuming training from checkpoint: {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iter = checkpoint['iter'] + 1
        # Log resumed iteration
        mlflow.log_param("resumed_from_iter", start_iter - 1)
    else:
        print("Starting training from scratch.")

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Training Loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for iter in tqdm(range(start_iter, max_iters), desc="Training"):
        try:
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Log training loss
            mlflow.log_metric("train_loss", loss.item(), step=iter)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=iter)
            
            # Evaluate loss and Save Checkpoint
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                mlflow.log_metrics({
                    "val_loss": losses['val'],
                    "train_eval_loss": losses['train']
                }, step=iter)
                
                print(f"\nIter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
                
                # Save the model checkpoint
                print(f"Saving checkpoint to {model_checkpoint_path}")
                torch.save({
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': losses['val'],
                }, model_checkpoint_path)
                
                # Save best model
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    torch.save({
                        'iter': iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': losses['val'],
                    }, best_model_path)
                    # Log best model as artifact if it exists
                    if os.path.exists(best_model_path):
                        mlflow.log_artifact(best_model_path)
                
                # Log checkpoint as artifact every 10 eval intervals (only if file exists)
                if iter % (eval_interval * 10) == 0 and os.path.exists(model_checkpoint_path):
                    mlflow.log_artifact(model_checkpoint_path)
                    
        except Exception as e:
            print(f"Error at iteration {iter}: {e}")
            mlflow.log_text(str(e), f"error_iter_{iter}.txt")
            break

    # Log final artifacts (only if they exist)
    if os.path.exists(model_checkpoint_path):
        mlflow.log_artifact(model_checkpoint_path)
    if os.path.exists(tokenizer_path):
        mlflow.log_artifact(tokenizer_path)
    if os.path.exists(best_model_path):
        mlflow.log_artifact(best_model_path)
    
    # Log model to MLflow
    try:
        mlflow.pytorch.log_model(model, "final_model")
    except Exception as e:
        print(f"Could not log PyTorch model to MLflow: {e}")
    
    print("Training completed.")
    print(f"MLflow run ID: {run.info.run_id}")
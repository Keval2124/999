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

# Config 
data_dir = '/home/keval/Music/999/archive/output'
tokenizer_path = 'bpe_tokenizer_v2.json'
vocab_size = 5000
end_of_text_token = "<|endoftext|>"

model_checkpoint_path = 'gpt_model_checkpoint.pth' # File to save the model
max_iters = 5000 
eval_interval = 1000 
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        match = re.match(r'^(OPERATOR|CALLER):\s*(.+)$', line.strip())
        if match:
            speaker, dialogue = match.groups()
            if len(dialogue.strip()) > 3:
                clean_lines.append(f"{speaker}: {dialogue}")
    return "\n".join(clean_lines)

print("Cleaning data...")
all_documents = []
for fname in tqdm(os.listdir(data_dir), desc="Processing files"):
    if fname.endswith('.txt'):
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            cleaned_content = cleaning(f.read())
            if cleaned_content: all_documents.append(cleaned_content)
all_text = f"\n{end_of_text_token}\n".join(all_documents)

def load_tokenizer(text, vocab_size, save_path):
    if os.path.exists(save_path):
        tokenizer = Tokenizer.from_file(save_path)
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]", end_of_text_token])
        tokenizer.train_from_iterator([text], trainer)
        tokenizer.save(save_path)
    return tokenizer

tokenizer = load_tokenizer(all_text, vocab_size, tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)
data = torch.tensor(encode(all_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class brain(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn, self.c_proj = nn.Linear(n_embd, 3 * n_embd), nn.Linear(n_embd, n_embd)
        self.attn_dropout, self.resid_dropout = nn.Dropout(dropout), nn.Dropout(dropout)
        self.n_head, self.n_embd = n_head, n_embd
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k, q, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (k, q, v)]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = brain(n_embd, n_head, dropout)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class gpt(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(vocab_size, n_embd), wpe=nn.Embedding(block_size, n_embd), drop=nn.Dropout(dropout), h=nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]), ln_f=nn.LayerNorm(n_embd)))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb, pos_emb = self.transformer.wte(idx), self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h: x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# Execution 
print(f"Using device: {device}")
model = gpt(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

start_iter = 0
if os.path.exists(model_checkpoint_path):
    print(f"Resuming training from checkpoint: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_iter = checkpoint['iter'] + 1
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
for iter in tqdm(range(start_iter, max_iters), desc="Training"):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Evaluate loss and Save Checkpoint
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"\nIter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")

        # --- NEW: Save the model checkpoint ---
        print(f"Saving checkpoint to {model_checkpoint_path}")
        torch.save({
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': losses['val'],
        }, model_checkpoint_path)

# Text Generation
print("Training finished. Generating sample text...")

# Load the final model for generation
final_model = gpt(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
checkpoint = torch.load(model_checkpoint_path)
final_model.load_state_dict(checkpoint['model_state_dict'])

def generate(model, start_text, max_new_tokens=200):
    model.eval()
    idx = torch.tensor(encode(start_text), dtype=torch.long, device=device)[None, :]
    print(f"\nGenerating from prompt: '{start_text}'")
    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens), desc="Generating text"):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if next_id.item() == tokenizer.token_to_id(end_of_text_token):
                break
    return decode(idx[0].tolist())

prompt = "OPERATOR: 911, what is your emergency?"
generated_text = generate(final_model, prompt, max_new_tokens=300)

print("\n--- Prompt Script---")
print(generated_text)
print("------------------------")
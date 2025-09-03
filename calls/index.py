import os, math, time, json
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- DEVICE & SCALER --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    from torch.cuda.amp import GradScaler
else:
    try:                     # PyTorch ≥ 2.0 has CPU GradScaler
        from torch.amp import GradScaler
    except ImportError:      # < 2.0 fallback
        class GradScaler:
            def __init__(self, enabled=True): self.enabled = enabled
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass

from torch.amp import autocast

# -------------------- MODEL HYPERPARAMS --------------------
n_layer = 6
n_head = 6
kv_heads = 2
n_embd = 384
block_size = 512
dropout = 0.25  # Increased slightly to prevent overfitting
batch_size = 64
max_iters = 100000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-4  # Reduced LR for longer training
warmup_steps = 1000  # Increased warmup for stability
grad_clip = 1.0
ckpt_dir = "ckpt"
final_ckpt = "999_gpt.pt"
weight_decay = 1e-2  # Added weight decay for regularization
patience = 3  # For early stopping

# -------------------- SCENARIO TAGS --------------------
# Keeping similar scenarios but adapted for UK context (e.g., fire remains fire, but model will learn UK codes/terminology)
SCENARIOS = ["fire", "medical", "active_shooter", "traffic", "domestic", "robbery", "burglary", "assault", "overdose", "suicide", "hazardous_material", "missing_person", "child_abuse", "carbon_monoxide"]

# -------------------- DATA CLEANING & AUGMENTATION --------------------
raw_path   = "additional_911_dialogs.txt"  # USA 911 data
clean_path = "clean_input.txt"             # Augmented output with UK adaptations

# Load additional JSON data for UK adaptation
with open("address_name.json", "r", encoding="utf-8") as f:
    address_name_data = json.load(f)  # List of {"name":, "address":}

with open("uk_emergency_codes.json", "r", encoding="utf-8") as f:
    uk_codes_data = json.load(f)      # Dict of UK emergency codes

with open("data.json", "r", encoding="utf-8") as f:
    cues_data = json.load(f)          # {"CALLER_CUES": [...], "OPERATOR_CUES": [...]}

def flatten_uk_codes(data, prefix=""):
    """Flatten UK codes JSON into readable text for training."""
    flat = []
    if isinstance(data, dict):
        for k, v in data.items():
            flat.extend(flatten_uk_codes(v, f"{prefix}{k}: "))
    elif isinstance(data, str):
        flat.append(f"{prefix}{data}")
    return flat

uk_codes_text = "\n".join(flatten_uk_codes(uk_codes_data))
caller_cues = cues_data["CALLER_CUES"]
operator_cues = cues_data["OPERATOR_CUES"]

# Convert JSONs to text blocks separately
addresses_text = "uk addresses and names:\n" + "\n".join(f"name: {item['name']}, address: {item['address']}" for item in address_name_data)
codes_text = "uk emergency codes:\n" + uk_codes_text
cues_text = (
    "caller cues:\n" + "\n".join(caller_cues) + "\n"
    "operator cues:\n" + "\n".join(operator_cues)
)

# Combined JSON knowledge text (separate from dialogs)
json_knowledge = addresses_text + "\n" + codes_text + "\n" + cues_text + "\n"

def clean_logs(raw_file: str, out_file: str):
    if os.path.exists(out_file):
        logger.info(f"{out_file} already exists – skipping re-clean.")
        return
    lines = open(raw_file, encoding="utf-8").read().splitlines()
    cleaned = []
    buf = []
    for ln in lines:
        ln = ln.strip().lower()
        if not ln:
            continue
        # Adapt USA to UK: Replace "911" with "999"
        ln = ln.replace("911", "999").replace("9-1-1", "9-9-9")
        if ln.startswith("operator:"):
            buf.append("operator: " + ln[9:].strip())
        elif ln.startswith("caller:"):
            buf.append("caller: " + ln[7:].strip())
        else:
            buf.append("caller: " + ln)
    
    # split into chunks and tag
    chunk_size = 256
    for i in range(0, len(buf), chunk_size):
        chunk = buf[i:i+chunk_size]
        scenario = SCENARIOS[i % len(SCENARIOS)]
        cleaned.append(f"scenario:{scenario}\n" + "\n".join(chunk))
    open(out_file, "w", encoding="utf-8").write("\n".join(cleaned))
    logger.info(f"Wrote {len(cleaned)} chunks -> {out_file}")

clean_logs(raw_path, clean_path)

# -------------------- TOKENIZER --------------------
text = json_knowledge + open(clean_path, encoding="utf-8").read()  # Prepend JSON knowledge separately
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

# -------------------- DATA LOADER --------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# -------------------- ARCHITECTURE --------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class SwiGLU(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        hidden = int(8/3 * n_embd)
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs).to(device)
    sin = torch.sin(freqs).to(device)
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotate(q), rotate(k)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embd, kv_heads):
        super().__init__()
        assert n_head % kv_heads == 0
        self.n_head, self.kv_heads = n_head, kv_heads
        self.head_dim = n_embd // n_head
        self.q_per_kv = n_head // kv_heads
        total = n_embd + 2 * (n_embd // n_head * kv_heads)
        self.qkv = nn.Linear(n_embd, total, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        cos, sin = precompute_freqs_cis(self.head_dim, block_size)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.n_head*self.head_dim,
                             self.kv_heads*self.head_dim,
                             self.kv_heads*self.head_dim], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.q_per_kv, dim=1)
        v = v.repeat_interleave(self.q_per_kv, dim=1)
        cos, sin = self.cos[:T], self.sin[:T]
        q, k = apply_rotary_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, n_head, n_embd, kv_heads):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, n_embd, kv_heads)
        self.ffn  = SwiGLU(n_embd)
        self.ln1  = RMSNorm(n_embd)
        self.ln2  = RMSNorm(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_head, n_embd, kv_heads) for _ in range(n_layer)])
        self.ln_f    = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -------------------- TRAINING LOOP --------------------
model = GPT().to(device)
logger.info(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scaler = GradScaler(enabled=(device == 'cuda'))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, s/warmup_steps) if s<warmup_steps else 0.5*(1+math.cos(math.pi*(s-warmup_steps)/(max_iters-warmup_steps))))

best_val_loss = float('inf')
patience_counter = 0

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ('train','val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            with autocast(device_type=device, dtype=torch.bfloat16 if device=='cuda' else torch.float32):
                logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def save_ckpt(name):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if device=='cuda' else None,
                "stoi": stoi, "itos": itos}, f"{ckpt_dir}/{name}")

for iter in range(max_iters):
    start_time = time.time()
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        logger.info(f"step {iter}: train {losses['train']:.4f}  val {losses['val']:.4f}")
        save_ckpt(f"iter{iter}")
        
        # Early stopping check
        val_loss = losses['val']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_ckpt("best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at iteration {iter}: val loss did not improve for {patience} evaluations")
                break  # Stop training

    xb, yb = get_batch('train')
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type=device, dtype=torch.bfloat16 if device=='cuda' else torch.float32):
        logits, loss = model(xb, yb)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    end_time = time.time()
    logger.info(f"Iteration {iter} completed in {end_time - start_time:.2f} seconds")

save_ckpt(final_ckpt)
logger.info(f"Training complete! Checkpoint: {ckpt_dir}/{final_ckpt}")
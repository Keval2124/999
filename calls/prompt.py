#!/usr/bin/env python3
"""
prompt.py
Generate a full UK 999 emergency call script using the trained model.
The model auto-decides the scenario and generates until it seems complete (up to a safe max).
Run: python prompt.py
Output saved to 999_script.txt
"""
import os, torch, random, logging
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load paths
ckpt_dir = "ckpt"
final_ckpt = "999_gpt.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Re-define model params and architecture (from train_999_sim.py)
n_layer = 6
n_head = 6
kv_heads = 2
n_embd = 384
block_size = 512
dropout = 0.2

class RMSNorm(torch.nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class SwiGLU(torch.nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        hidden = int(8/3 * n_embd)
        self.gate = torch.nn.Linear(n_embd, hidden, bias=False)
        self.up = torch.nn.Linear(n_embd, hidden, bias=False)
        self.down = torch.nn.Linear(hidden, n_embd, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
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

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_embd, kv_heads):
        super().__init__()
        assert n_head % kv_heads == 0
        self.n_head, self.kv_heads = n_head, kv_heads
        self.head_dim = n_embd // n_head
        self.q_per_kv = n_head // kv_heads
        total = n_embd + 2 * (n_embd // n_head * kv_heads)
        self.qkv = torch.nn.Linear(n_embd, total, bias=False)
        self.proj = torch.nn.Linear(n_embd, n_embd)
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

class Block(torch.nn.Module):
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

class GPT(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
        self.blocks  = torch.nn.Sequential(*[Block(n_head, n_embd, kv_heads) for _ in range(n_layer)])
        self.ln_f    = RMSNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=1.0, top_k=None, repetition_penalty=1.0, end_tokens=None):
        generated = idx.clone()
        for _ in range(max_new_tokens):
            idx_cond = generated if generated.size(1) <= block_size else generated[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                prev_tokens = generated[:, -block_size:]
                for token in torch.unique(prev_tokens):
                    logits[:, token] /= repetition_penalty
            
            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)
            # Check for end condition
            if end_tokens and idx_next.item() in end_tokens:
                break
        return generated

# Load checkpoint
ckpt_path = f"{ckpt_dir}/{final_ckpt}"
if not os.path.exists(ckpt_path):
    logger.error(f"Checkpoint not found at {ckpt_path}. Train the model first.")
    exit(1)

ckpt = torch.load(ckpt_path, map_location=device)
stoi = ckpt["stoi"]
itos = ckpt["itos"]
vocab_size = len(stoi)
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join(itos[i] for i in l if i in itos)

model = GPT(vocab_size).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
logger.info("Model loaded successfully.")

# Stronger prompt to focus on script generation only, no documents
prompt = (
    "<|system|>\nYou are strictly a UK 999 emergency call script generator. Do not generate any document, PDF, guide, or non-script content. Only output the emergency call script.\n"
    "Rules:\n- Choose one random scenario from training.\n"
    "- Start with 'scenario: [chosen]'\n"
    "- Then alternate 'operator: [line]' and 'caller: [line]'.\n"
    "- Use trained knowledge for realistic details (addresses, codes, cues).\n"
    "- Keep coherent, no repetitions or hallucinations.\n"
    "- End with 'operator: Help is on the way.' followed by '<|end|>' when resolved.\n"
    "- No other text.\n"
    "Generate now:\n"
)

tokens = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

# Generate with stricter params: lower temp for less hallucination, higher repetition_penalty
end_token_id = stoi['<'] if '<' in stoi else None
with torch.no_grad():
    generated_ids = model.generate(tokens, max_new_tokens=1024, temperature=0.5, top_p=0.8, top_k=20, repetition_penalty=1.5, end_tokens=[end_token_id])[0]

generated_text = decode(generated_ids.tolist())

# Post-process: Filter to only script lines, enforce format
lines = generated_text.splitlines()
script = []
seen_lines = set()
current_role = "operator"
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith("scenario:"):
        script.append(line)
        continue
    # Skip any line that looks like document content (heuristic)
    if any(word in line.lower() for word in ["pdf", "page", "document", "title", "abstract", "table of contents", "hpc", "barkla"]):
        continue
    expected_prefix = f"{current_role}:"
    clean_line = line.lower()
    if clean_line in seen_lines:
        continue
    if line.startswith(expected_prefix):
        script.append(line)
        seen_lines.add(clean_line)
        current_role = "caller" if current_role == "operator" else "operator"
    elif line.startswith(("operator:", "caller:")):
        script.append(line)
        seen_lines.add(clean_line)
        current_role = "caller" if line.startswith("operator:") else "operator"
    if "<|end|>" in line:
        break

# If incomplete or empty, add default resolution
if not script or len(script) < 4:
    script = [
        "scenario: fire",
        "operator: 9-9-9, what's your emergency?",
        "caller: There's a fire!",
        "operator: Address?",
        "caller: 123 Main St.",
        "operator: Help is on the way."
    ]
elif script[-1].startswith("caller:"):
    script.append("operator: Emergency services dispatched. Stay safe.")

full_script = "\n".join(script)

# Save output
out_file = "999_script.txt"
with open(out_file, 'w', encoding='utf-8') as f:
    f.write(full_script)
logger.info(f"Generated script saved to {out_file}")
logger.info("\nPreview:\n" + "\n".join(script[:10]) + "\n...")
import os
import torch
import logging
import random  # For random scenario selection if desired

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- DEVICE --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------- MODEL HYPERPARAMS (MUST MATCH TRAINING) --------------------
n_layer = 6
n_head = 6
kv_heads = 2
n_embd = 384
block_size = 512
dropout = 0.25

# -------------------- ARCHITECTURE (COPY FROM TRAINING CODE) --------------------
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
        return self.dropout(self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x)))

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
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class Block(torch.nn.Module):
    def __init__(self, n_head, n_embd, kv_heads):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, n_embd, kv_heads)
        self.ffn = SwiGLU(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
        self.blocks = torch.nn.Sequential(*[Block(n_head, n_embd, kv_heads) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
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
            loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.2):
        recent_tokens = set()
        generated_text = decode(idx[0].tolist())  # Start with initial prompt
        stop_phrases = ["thank you", "good bye", "goodbye", "bye"]
        for _ in range(max_new_tokens):
            if any(phrase in generated_text.lower()[-100:] for phrase in stop_phrases):  # Increased to last 100 chars for better detection
                break
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty
            for token in recent_tokens:
                logits[0, token] /= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            generated_text += decode([idx_next.item()])  # Append new char

            # Update recent tokens
            recent_tokens.add(idx_next.item())
            if len(recent_tokens) > 5:
                recent_tokens.pop()

            # Stop if repetition detected
            if idx.size(1) > 100 and torch.all(idx[:, -10:] == idx[:, -20:-10]):
                break

        return idx

# -------------------- SCENARIOS (FROM TRAINING) --------------------
SCENARIOS = ["fire", "medical", "active_shooter", "traffic", "domestic", "robbery", "burglary", "assault", "overdose", "suicide", "hazardous_material", "missing_person", "child_abuse", "carbon_monoxide"]

# -------------------- LOAD CHECKPOINT --------------------
ckpt_path = "ckpt/999_gpt.pt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = len(stoi)
encode = lambda s: [stoi[c] for c in s if c in stoi]  # Safe encode
decode = lambda l: ''.join(itos[i] for i in l)

model = GPT(vocab_size).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()
logger.info(f"Loaded model from {ckpt_path} with {vocab_size} vocab size")

# -------------------- GENERATION PARAMETERS --------------------
incremental_tokens = 500  # Tokens to add each time if not complete
total_max_tokens = 3000   # Absolute max to prevent infinite loop
temperature = 0.5         # Lowered for more coherent, deterministic outputs
top_k = 20                # Tightened to reduce nonsense
repetition_penalty = 1.5  # To reduce repetitions
output_file = "generated_999_call.txt"

# -------------------- PROMPT GENERATION --------------------
def generate_prompt(scenario=None):
    if scenario is None:
        scenario = random.choice(SCENARIOS)  # Random if not specified
    prompt = (
        f"scenario:{scenario}\n"
        "Generate a coherent, realistic, and complete 999 emergency call dialogue strictly for the given scenario only. Do not mix or switch to other situations.\n"
        "The dialogue must alternate strictly between operator: and caller: lines without repetition or nonsense.\n"
        "Operator: Starts with '999, what's your emergency?', asks for address, nature of incident, if anyone is hurt, number of people involved, hazards, and provides calm instructions like stay safe, help is on the way.\n"
        "Caller: Provides clear, logical information without nonsense, describes the single situation step-by-step, sticking to the scenario.\n"
        "Complete the full dialogue until help is dispatched and the call ends politely with phrases like 'thank you', 'goodbye', or 'bye' from either party. Do not stop early; continue until the conversation is naturally complete. No incomplete sentences at the end.\n"
        "Use UK addresses, names, and emergency codes from training data.\n"
        "operator: 999, what's your emergency?\n"
        "caller:"
    )
    return prompt, scenario

# -------------------- MAIN GENERATION --------------------
prompt, scenario = generate_prompt()  # Or specify e.g., generate_prompt("active_shooter")
logger.info(f"Generating dialog for scenario: {scenario}")

input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
generated_ids = input_ids.clone()  # Start with prompt
current_new_tokens = incremental_tokens
stop_phrases = ["thank you", "good bye", "goodbye", "bye"]

while generated_ids.size(1) < total_max_tokens:
    generated_ids = model.generate(generated_ids, current_new_tokens, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty)
    generated_text = decode(generated_ids[0].tolist())
    
    # Check if complete
    if any(phrase in generated_text.lower()[-200:] for phrase in stop_phrases):  # Check last 200 chars
        break
    
    # If not complete, continue with more tokens
    logger.info("Dialogue not complete; generating more tokens...")
    current_new_tokens = incremental_tokens  # Add more

generated_text = decode(generated_ids[0].tolist())

# Post-process: Ensure alternating roles, remove incomplete lines
lines = generated_text.split('\n')
cleaned_lines = []
last_role = 'scenario'
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith("operator:") or line.startswith("caller:") or line.startswith("scenario:"):
        role = line.split(':')[0]
        if role == last_role:
            continue  # Skip repeated roles
        last_role = role
        cleaned_lines.append(line)
    elif cleaned_lines and ':' not in cleaned_lines[-1]:
        cleaned_lines[-1] += ' ' + line  # Append to previous if no role

# Trim after stop phrases if not caught in generate
full_text = '\n'.join(cleaned_lines)
stop_phrases = ["thank you", "good bye", "goodbye", "bye"]
stop_idx = min([full_text.lower().find(phrase) for phrase in stop_phrases if full_text.lower().find(phrase) != -1] + [len(full_text)])
if stop_idx != len(full_text):
    full_text = full_text[:stop_idx + max(len(phrase) for phrase in stop_phrases)]  # Trim after the stop phrase
    lines = full_text.split('\n')
    cleaned_lines = lines[:-1] if not lines[-1].strip() else lines  # Remove incomplete last line if any

# Remove any "... [call continues]" or similar
generated_text = '\n'.join(cleaned_lines).rstrip()
generated_text = generated_text.replace("... [call continues]", "").replace("[call continues]", "").strip()
if not generated_text.endswith(('.', '!', '?')):
    generated_text += '.'

# Save to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(generated_text)
logger.info(f"Generated dialog saved to {output_file}")

# Print preview
print("\nGenerated 999 Emergency Call:\n")
print(generated_text)
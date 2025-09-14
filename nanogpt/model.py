# model.py
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------------
#  RMSNorm 
# -------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, ndim: int, bias: bool = False, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# -------------------------------------------------------------------
#  SwiGLU MLP
# -------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(2/3 * 4 * config.n_embd) # 2/3 * 4 = 2.666...
        self.gate = nn.Linear(config.n_embd, hidden, bias=config.bias) 
        self.up   = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

# -------------------------------------------------------------------
#  CausalSelfAttention  (softmax  +  linear kernel branch)
# -------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.scale   = self.head_dim ** -0.5

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # flash attention present?
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,nh,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ---- choose branch ----
        if self.flash and T <= 512:          # fast softmax
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        elif T > 512:                        # linear attention
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            num = q @ k.transpose(-2, -1)               # (B,nh,T,T)
            denom = num.cumsum(dim=-1).clamp(min=1e-6)  # causal normaliser
            att = num / denom
            y = att @ v
        else:                                  # slow softmax
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

# -------------------------------------------------------------------
#  Transformer block  (RMSNorm + ReZero α)
# -------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)
        self.alpha = nn.Parameter(torch.zeros(1))      # ReZero

    def forward(self, x):
        x = x + self.alpha * self.attn(self.ln_1(x))
        x = x + self.alpha * self.mlp(self.ln_2(x))
        return x

# -------------------------------------------------------------------
#  Config & GPT model  
# -------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True 

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        device = idx.device
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)  # (b,t,n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t,n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # -------------  optimizer (8-bit ready)  -----------------
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Return 8-bit AdamW if bitsandbytes is available, else fused AdamW."""
        # --- parameter groups ---
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params  = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params= [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups  = [
            {"params": decay_params,  "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        print(f"num decayed tensors: {len(decay_params)}, nodesayed: {len(nodecay_params)}")
        use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        extra_args = dict(fused=True) if use_fused else dict()

        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(optim_groups, lr=learning_rate, betas=betas)
        except ImportError:
            pass
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    # -------------  generate  -----------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # -------------  from_pretrained  -----------------
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight"]

        assert len(sd_keys_hf) == len(sd_keys)
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
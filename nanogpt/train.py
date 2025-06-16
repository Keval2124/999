import os, time, math, pickle
import numpy as np # type: ignore
import torch # type: ignore
from contextlib import nullcontext # type: ignore
from torch.nn.parallel import DistributedDataParallel as DDP # type: ignore
from torch.distributed import init_process_group # type: ignore
from model import GPTConfig, GPT

# --- Config ---
out_dir, dataset, init_from = 'out', 'calls', 'scratch'
wandb_log, wandb_project, wandb_run_name = False, 'owt', 'gpt2'
eval_interval, log_interval, eval_iters, eval_only = 2000, 1, 200, False
always_save_checkpoint = True
batch_size, block_size = 12, 1024
n_layer, n_head, n_embd, dropout, bias = 12, 12, 768, 0.0, False
learning_rate, weight_decay = 6e-4, 1e-1
beta1, beta2, grad_clip = 0.9, 0.95, 1.0
max_iters, decay_lr, warmup_iters, lr_decay_iters, min_lr = 600000, True, 2000, 600000, 6e-5
device, dtype, compile = 'cpu', 'float32', False
backend, gradient_accumulation_steps = 'nccl', 5 * 8

config = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}

# --- DDP Setup ---
rank_env = os.environ.get('RANK', -1)
ddp = int(rank_env) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(rank_env)
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'; torch.cuda.set_device(device)
    master_process = ddp_rank == 0; seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True; seed_offset = 0; ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
if master_process: os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- Tokenization ---
data_dir = os.path.join('data', dataset)
start_tokenize = time.time()
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
tokenize_time = time.time() - start_tokenize
print(f"Tokenization took {tokenize_time:.2f} seconds")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return (x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)) if device_type == 'cuda' else (x.to(device), y.to(device))

# --- Model Init ---
iter_num, best_val_loss = 0, 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta_vocab_size = pickle.load(f)['vocab_size']

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
    model = GPT(GPTConfig(**model_args))
elif init_from == 'resume':
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    for k in model_args: model_args[k] = ckpt['model_args'][k]
    model = GPT(GPTConfig(**model_args))
    state_dict = {k[11:] if k.startswith('_orig_mod.') else k: v for k, v in ckpt['model'].items()}
    model.load_state_dict(state_dict)
    iter_num, best_val_loss = ckpt['iter_num'], ckpt['best_val_loss']
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))
    for k in model_args: model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float32'))

if init_from == 'resume':
    optimizer.load_state_dict(ckpt['optimizer'])

if compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# --- Eval ---
@torch.no_grad()
def estimate_loss():
    out = {}; model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx: _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train(); return out

def get_lr(it):
    if it < warmup_iters: return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters: return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# if wandb_log and master_process:
#     import wandb
#     wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# --- Training ---
X, Y = get_batch('train')
t0, local_iter_num, raw_model = time.time(), 0, model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups: param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # if wandb_log:
        #     wandb.log({"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'], "lr": lr, "mfu": running_mfu*100})
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only: break

    for micro_step in range(gradient_accumulation_steps):
        if ddp: model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y); loss /= gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1, dt = time.time(), time.time() - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1; local_iter_num += 1
    if iter_num > max_iters: break

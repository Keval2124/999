# train.py (or your main training script name)
import os, time, math, pickle
import numpy as np # type: ignore
import torch # type: ignore
from contextlib import nullcontext # type: ignore
from torch.nn.parallel import DistributedDataParallel as DDP # type: ignore
from torch.distributed import init_process_group # type: ignore

from model import GPTConfig, GPT

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  

# --- Config ---
# --- Updated Output Directory ---
out_dir = "nanogpt"
# --- Dataset Path ---
# Note: 'dataset' was previously used, but now data_dir is set directly.
# Keeping 'dataset' for potential future use or clarity, but it's not used in path construction anymore.
dataset = '/users/sgkshah/scratch/999/nanogpt/data/calls'
checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
if os.path.exists(checkpoint_path):
    init_from = 'resume'
    print(f"Found checkpoint at {checkpoint_path}. Setting init_from='resume'.")
else:
    init_from = 'scratch'
    print(f"No checkpoint found at {checkpoint_path}. Setting init_from='scratch'.")
dtype = 'bfloat16' # Changed from 'float32'
eval_interval, log_interval, eval_iters, eval_only = 2000, 1, 200, False
always_save_checkpoint = True
# --- Data & Model Hyperparameters ---
# --- Small Model Configuration with Adjusted Batch Settings ---
batch_size, block_size = 8, 512  # Batch size per accumulation step, sequence length
n_layer, n_head, n_embd, dropout, bias = 4, 4, 256, 0.1, False # Small model: 4 layers, 4 heads, 256 dims
# --- Optimization Hyperparameters ---
# --- Adjusted Learning Rate ---
learning_rate, weight_decay = 3e-4, 1e-1 # Slightly lower LR often helps with smaller/larger effective batch sizes
beta1, beta2, grad_clip = 0.9, 0.95, 1.0
# --- Learning Rate Schedule ---
max_iters, decay_lr, warmup_iters, lr_decay_iters, min_lr = 60000, True, 2000, 60000, 6e-5
device = 'cpu' # Training on CPU as per your environment variable setting
compile = False # Requires PyTorch 2.0+. Might cause issues with RMSNorm or fused ops if not set up correctly.
# --- DDP Settings ---
# --- CRITICAL FIX: Reduced Gradient Accumulation Steps ---
# Original was 5 * 8 = 40, leading to a very large effective batch size (8 * 40 = 320).
# This change significantly reduces the effective batch size (8 * 5 = 40).
backend, gradient_accumulation_steps = 'nccl', 5 # Adjust based on your setup (e.g., number of GPUs * desired local batch size)
# --- Early Stopping Config ---
patience = 5  # Number of evaluation intervals to wait for improvement
min_delta = 0.001 # Minimum change in validation loss to qualify as improvement
# --- Early Stopping State ---
best_val_loss_early_stop = float('inf')
patience_counter = 0
should_stop_training = False # Flag to potentially break the main loop
# --- End Early Stopping Config ---
# Collect config for logging
config = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}
# --- MLflow Setup ---
import mlflow
from mlflow.models import infer_signature
# --- Updated MLflow Tracking URI ---
mlflow.set_tracking_uri("file:///users/sgkshah/scratch/999/mlruns") # Ensure 'file://' prefix
mlflow.set_experiment("Nanogpt") # Specific experiment for your project
# --- DDP Setup --- (This part remains largely unchanged, but gradient_accumulation_steps is now different)
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

# Recalculate tokens per iteration with the new settings
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}") # Should now print 40 * 512 = 20,480


if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# --- Key Improvement: Ensure correct context for CPU/CUDA ---
# AMP context is only beneficial on CUDA devices.
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- Tokenization ---
# --- Updated Data Directory Path ---
# Corrected path construction: dataset variable now holds the full path
data_dir = dataset # '/users/sgkshah/scratch/999/nanogpt/data/calls'
print(f"Looking for data in: {data_dir}")
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}. Please ensure your tokenized data is in {data_dir}/train.bin and {data_dir}/val.bin")

start_tokenize = time.time()
train_data_path = os.path.join(data_dir, 'train.bin')
val_data_path = os.path.join(data_dir, 'val.bin')
meta_path = os.path.join(data_dir, 'meta.pkl') # Check for meta.pkl here too

if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
    raise FileNotFoundError(f"Train or Val data file not found in {data_dir}. Required: train.bin, val.bin")

train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
tokenize_time = time.time() - start_tokenize
print(f"Tokenization/data loading took {tokenize_time:.2f} seconds")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Ensure we don't sample beyond the available data
    max_start_idx = len(data) - block_size - 1
    if max_start_idx <= 0:
        raise ValueError(f"Data split '{split}' is too small for block_size={block_size}")
    ix = torch.randint(0, max_start_idx, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # Pin memory only if using CUDA for faster transfer
    try:
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA out of memory error: {e}")
            print("Falling back to CPU for this batch")
            x = x.to('cpu')
            y = y.to('cpu')
        else:
            raise
    return x, y

# --- Model Init ---
iter_num, best_val_loss = 0, 1e9 # best_val_loss for checkpointing
# --- Early Stopping State Initialization ---
best_val_loss_early_stop = float('inf') # best_val_loss for early stopping logic
patience_counter = 0
should_stop_training = False
# ---

meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        meta_vocab_size = meta.get('vocab_size')
        print(f"Found vocab_size={meta_vocab_size} in meta.pkl")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    # Use vocab size from meta.pkl or a default
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
    print(f"Initializing model from scratch with vocab_size={model_args['vocab_size']}")
    model = GPT(GPTConfig(**model_args))
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found at {ckpt_path}")
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # Restore model args from checkpoint
    ckpt_model_args = ckpt.get('model_args', {})
    # Override config with checkpoint args
    for k in model_args:
        if k in ckpt_model_args:
            model_args[k] = ckpt_model_args[k]
    print(f"Resuming with model args from checkpoint: {model_args}")
    model = GPT(GPTConfig(**model_args))
    state_dict = ckpt['model']
    # Handle potential prefix from DDP or torch.compile
    unwanted_prefixes = ['_orig_mod.', 'module.']
    for k in list(state_dict.keys()):
        for prefix in unwanted_prefixes:
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = ckpt.get('iter_num', 0)
    best_val_loss = ckpt.get('best_val_loss', 1e9)
    # Restore early stopping state if present (optional but good practice)
    best_val_loss_early_stop = ckpt.get('best_val_loss_early_stop', float('inf'))
    patience_counter = ckpt.get('patience_counter', 0)
    print(f"Resumed from iteration {iter_num}, best val loss {best_val_loss:.4f}, early stop best {best_val_loss_early_stop:.4f}, patience {patience_counter}")

elif init_from.startswith('gpt2'):
    print(f"Initializing from pretrained GPT-2 model: {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))
    # Update model_args with loaded config for consistency
    for k in model_args:
        if hasattr(model.config, k):
            model_args[k] = getattr(model.config, k)

# Crop block size if necessary
if block_size < model.config.block_size:
    print(f"Cropping model block size from {model.config.block_size} to {block_size}")
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)
raw_model = model.module if ddp else model # Keep reference to unwrapped model for optimizer

# --- Key Improvement: Use fused AdamW if available ---
print("Configuring optimizer...")
optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# Scaler is only strictly needed for float16, but harmless for bfloat16
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16' or dtype == 'bfloat16'))

# Load optimizer state if resuming
if init_from == 'resume' and 'optimizer' in ckpt:
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer state loaded from checkpoint.")
    except Exception as e:
        print(f"Warning: Could not load optimizer state: {e}. Starting from scratch optimizer state.")

# Optional compilation (use with caution)
if compile:
    print("Compiling the model (requires PyTorch 2.0+)...")
    try:
        model = torch.compile(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Warning: Model compilation failed: {e}. Proceeding without compilation.")

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # Update raw_model reference after DDP wrapping

# --- Start MLflow Run ---
if master_process:
    print("Starting MLflow run...")
    mlflow.start_run(run_name="Nanogpt-with-early-stopping")
    mlflow.log_params(config)
    print("MLflow run started and parameters logged.")

# --- Eval ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Fetch initial batch
X, Y = get_batch('train')
t0, local_iter_num = time.time(), 0
running_mfu = -1.0

print("Starting training loop...")
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        print(f"Running evaluation at iteration {iter_num}...")
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # --- Early Stopping Check ---
        # Check if the current validation loss is the best so far (considering min_delta)
        if losses['val'] < (best_val_loss_early_stop - min_delta):
            best_val_loss_early_stop = losses['val']
            patience_counter = 0 # Reset patience counter if improvement
            print(f"Early stopping counter reset. New best val loss: {best_val_loss_early_stop:.4f}")
        else:
            patience_counter += 1 # Increment counter if no sufficient improvement
            print(f"No significant improvement in val loss. Patience counter: {patience_counter}/{patience}")

        # If patience counter reaches the limit, trigger stop
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} evaluations without improvement.")
            should_stop_training = True # Set flag to break main loop

        # --- Log to MLflow (before potential stop) ---
        mlflow.log_metric("val/loss", losses['val'], step=iter_num)
        mlflow.log_metric("train/loss_eval", losses['train'], step=iter_num)
        mlflow.log_metric("lr", lr, step=iter_num) # Log LR here too for eval steps

        # Save best model based on *overall* best val loss (for checkpointing the best model found)
        # This logic now also checks for the early stopping condition
        if losses['val'] < best_val_loss or always_save_checkpoint:
            if losses['val'] < best_val_loss:
                print(f"New overall best validation loss: {losses['val']:.4f}")
            best_val_loss = losses['val'] # For checkpointing logic

            # --- Asynchronous Checkpoint Saving (Refined) ---
            # Create checkpoint dictionary *inside* the save function context or pass explicitly
            checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
            def save_checkpoint_thread_fn():
                """Function to run in the background thread."""
                # Recreate checkpoint dict inside thread fn to ensure latest state if needed,
                # though iter_num etc. are captured at call time.
                ckpt_data = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    # Include early stopping state for resuming correctly
                    'best_val_loss_early_stop': best_val_loss_early_stop,
                    'patience_counter': patience_counter,
                }
                if ddp:
                    ckpt_data['ddp_rank'] = ddp_rank
                try:
                    torch.save(ckpt_data, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                    # Optionally log artifact here if desired within thread
                    mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                except Exception as e:
                    print(f"Error saving checkpoint to {checkpoint_path}: {e}")

            import threading
            # Start a background thread to save the checkpoint
            threading.Thread(target=save_checkpoint_thread_fn, daemon=True).start()
            # --- End Asynchronous Checkpoint Saving ---


        # --- Check for Early Stop AFTER logging and potential saving ---
        if should_stop_training:
            print("Breaking main training loop due to early stopping.")
            break # Exit the main while loop

    if iter_num == 0 and eval_only:
        print("Eval only mode, exiting after initial eval.")
        break

    # --- Training Loop ---
    model.train() # Ensure model is in training mode
    optimizer.zero_grad(set_to_none=True) # Clear gradients at the start of accumulation

    lossf = 0.0 # For logging accumulated loss
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # Only sync gradients on the last micro-step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx: # AMP context
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # Scale loss for accumulation
            lossf += loss.detach() # Accumulate unscaled loss for logging

        # Backward pass
        # Use scaler only if dtype is float16 (bfloat16 usually doesn't need it, but scaler handles it)
        scaler.scale(loss).backward()

        # Fetch next batch while GPU is busy with backward pass
        X, Y = get_batch('train')

    # --- Gradient Clipping & Optimizer Step ---
    if grad_clip != 0.0:
        # Unscale gradients before clipping (needed when using scaler)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Optionally log grad norm
        if master_process and iter_num % log_interval == 0:
            mlflow.log_metric("grad_norm", grad_norm.item(), step=iter_num)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    # Note: optimizer.zero_grad is called at the beginning of the accumulation loop

    # --- Logging ---
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        # lossf already accumulated and detached during the loop
        if local_iter_num >= 5: # Wait a few steps for MFU to stabilize
            try:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            except Exception as e:
                # MFU estimation might fail, e.g., if model structure changes
                print(f"Warning: Could not estimate MFU: {e}")
                running_mfu = -1.0 # Reset if it fails
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        # Log to MLflow
        mlflow.log_metric("train/loss", lossf, step=iter_num) # Log accumulated loss
        # mlflow.log_metric("lr", lr, step=iter_num) # Already logged at eval interval
        if local_iter_num >= 5 and running_mfu != -1.0:
            mlflow.log_metric("mfu", running_mfu * 100, step=iter_num)


    iter_num += 1
    local_iter_num += 1
    # --- Check for max iters OR early stop flag ---
    if iter_num > max_iters or should_stop_training:
        break

# --- End MLflow Run ---
if master_process:
    final_status = "completed" if not should_stop_training else "stopped_early"
    print(f"Training {final_status}.")
    try:
        mlflow.log_param("final_status", final_status) # Log how training ended
        mlflow.end_run()
        print("MLflow run ended.")
    except Exception as e:
        print(f"Warning: Could not end MLflow run cleanly: {e}")
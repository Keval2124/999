import os, time, math, pickle
import numpy as np
import torch
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from model import GPT, GPTConfig
import logging

# Set up logging with corrected path
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nanogpt')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Atomic save helper function to prevent corruption
def safe_save_checkpoint(ckpt_data, checkpoint_path):
    tmp_path = checkpoint_path + '.tmp'
    try:
        # Save to temp with legacy serialization to avoid ZIP issues
        torch.save(ckpt_data, tmp_path, _use_new_zipfile_serialization=False)
        # Atomic rename
        os.rename(tmp_path, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Verify immediately (quick CPU load)
        test_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info("Checkpoint verified (load test passed)")
        return True
    except Exception as e:
        logger.error(f"Save/verification failed: {e}. Removing partial file.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return False

# Config
out_dir = "nanogpt"
dataset = 'data/calls'  # Changed to directory containing train.bin, val.bin, meta.pkl
checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
if os.path.exists(checkpoint_path):
    init_from = 'resume'
    logger.info(f"Found checkpoint at {checkpoint_path}. Setting init_from='resume'.")
else:
    init_from = 'scratch'
    logger.info(f"No checkpoint found at {checkpoint_path}. Setting init_from='scratch'.")
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
eval_interval, log_interval, eval_iters, eval_only = 2000, 1, 200, False
always_save_checkpoint = True
batch_size, block_size = 8, 1024
n_layer, n_head, n_embd, dropout, bias = 4, 4, 256, 0.1, False
learning_rate, weight_decay = 3e-4, 1e-1
beta1, beta2, grad_clip = 0.9, 0.95, 1.0
max_iters, decay_lr, warmup_iters, lr_decay_iters, min_lr = 60000, True, 2000, 60000, 6e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = False
patience = 5
min_delta = 0.001
best_val_loss_early_stop = float('inf')
patience_counter = 0
should_stop_training = False
config = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}

# MLflow Setup
import mlflow
mlflow.set_tracking_uri(os.path.join(os.getcwd(), "999/mlruns"))  # Use absolute path for reliability
mlflow.set_experiment("Nanogpt")

# DDP Setup
import torch.distributed as dist

gradient_accumulation_steps = 5

rank_env = os.environ.get('RANK', -1)
ddp = int(rank_env) != -1
if ddp:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'; torch.cuda.set_device(device)
    master_process = ddp_rank == 0; seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    # Initialize process group only if in DDP mode
    try:
        init_process_group(backend='gloo')
        logger.info("Gloo selected.")
    except RuntimeError:
        try:
            init_process_group(backend='nccl')
            logger.info("NCCL selected.")
        except RuntimeError:
            raise RuntimeError("Neither Gloo nor NCCL available.")
else:
    master_process = True; seed_offset = 0; ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
# Tokenization
data_dir = dataset
logger.info(f"Looking for data in: {data_dir}")
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}. Please ensure your tokenized data is in {data_dir}/train.bin and {data_dir}/val.bin")

start_tokenize = time.time()
train_data_path = os.path.join(data_dir, 'train.bin')
val_data_path = os.path.join(data_dir, 'val.bin')
meta_path = os.path.join(data_dir, 'meta.pkl')

if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
    raise FileNotFoundError(f"Train or Val data file not found in {data_dir}. Required: train.bin, val.bin")

train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
tokenize_time = time.time() - start_tokenize
logger.info(f"Tokenization/data loading took {tokenize_time:.2f} seconds")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    max_start_idx = len(data) - block_size - 1
    if max_start_idx <= 0:
        raise ValueError(f"Data split '{split}' is too small for block_size={block_size}")
    ix = torch.randint(0, max_start_idx, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    try:
        if device == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"CUDA out of memory error: {e}")
            logger.warning("Falling back to CPU for this batch")
            x = x.to('cpu')
            y = y.to('cpu')
        else:
            raise
    return x, y

# Model Init
iter_num, best_val_loss = 0, 1e9
best_val_loss_early_stop = float('inf')
patience_counter = 0
should_stop_training = False

meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        meta_vocab_size = meta.get('vocab_size')
        logger.info(f"Found vocab_size={meta_vocab_size} in meta.pkl")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
    logger.info(f"Initializing model from scratch with vocab_size={model_args['vocab_size']}")
    model = GPT(GPTConfig(**model_args))
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found at {ckpt_path}")
    logger.info(f"Loading checkpoint from {ckpt_path}")
    loaded = False
    for w_only in [False, True]:  # Try legacy first
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=w_only)
            logger.info(f"Loaded with weights_only={w_only}")
            loaded = True
            break
        except RuntimeError as e:
            if "PytorchStreamReader" in str(e):
                logger.warning(f"Retry with weights_only={w_only} failed: {e}")
                continue
            raise e
    if not loaded:
        raise RuntimeError("Failed to load checkpoint—delete and retrain from scratch.")
    ckpt_model_args = ckpt.get('model_args', {})
    for k in model_args:
        if k in ckpt_model_args:
            model_args[k] = ckpt_model_args[k]
    logger.info(f"Resuming with model args from checkpoint: {model_args}")
    model = GPT(GPTConfig(**model_args))
    state_dict = ckpt['model']
    unwanted_prefixes = ['_orig_mod.', 'module.']
    for k in list(state_dict.keys()):
        for prefix in unwanted_prefixes:
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = ckpt.get('iter_num', 0)
    best_val_loss = ckpt.get('best_val_loss', 1e9)
    best_val_loss_early_stop = ckpt.get('best_val_loss_early_stop', float('inf'))
    patience_counter = ckpt.get('patience_counter', 0)
    logger.info(f"Resumed from iteration {iter_num}, best val loss {best_val_loss:.4f}, early stop best {best_val_loss_early_stop:.4f}, patience {patience_counter}")
elif init_from.startswith('gpt2'):
    logger.info(f"Initializing from pretrained GPT-2 model: {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))
    for k in model_args:
        if hasattr(model.config, k):
            model_args[k] = getattr(model.config, k)
else:
    logger.error(f"Unknown init_from option: {init_from}")

if block_size < model.config.block_size:
    logger.info(f"Cropping model block size from {model.config.block_size} to {block_size}")
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)
raw_model = model.module if ddp else model

# Print model size information
num_params = sum(p.numel() for p in raw_model.parameters())
logger.info(f"Model has {num_params:,} parameters ({num_params / 1e6:.2f}M parameters)")

# Rough estimate of model size in MB (assuming float32 for calculation; actual size depends on dtype and storage)
param_bytes = num_params * 4  # float32 bytes per param
model_size_mb = param_bytes / (1024 * 1024)
logger.info(f"Approximate model size (float32): {model_size_mb:.2f} MB")

logger.info("Configuring optimizer...")
optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16' or dtype == 'bfloat16'))

if init_from == 'resume' and 'ckpt' in locals() and 'optimizer' in ckpt:
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info("Optimizer state loaded from checkpoint.")
    except Exception as e:
        logger.warning(f"Could not load optimizer state: {e}. Starting from scratch optimizer state.")

if compile:
    logger.info("Compiling the model (requires PyTorch 2.0+)...")
    try:
        model = torch.compile(model)
        logger.info("Model compiled successfully.")
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}. Proceeding without compilation.")

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

# Start MLflow Run
if master_process:
    logger.info("Starting MLflow run...")
    mlflow.start_run(run_name="Nanogpt-with-early-stopping")
    mlflow.log_params(config)
    logger.info("MLflow run started and parameters logged.")

# Eval
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

X, Y = get_batch('train')
t0, local_iter_num = time.time(), 0
running_mfu = -1.0

logger.info("Starting training loop...")
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        logger.info(f"Running evaluation at iteration {iter_num}...")
        losses = estimate_loss()
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < (best_val_loss_early_stop - min_delta):
            best_val_loss_early_stop = losses['val']
            patience_counter = 0
            logger.info(f"Early stopping counter reset. New best val loss: {best_val_loss_early_stop:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No significant improvement in val loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} evaluations without improvement.")
            should_stop_training = True

        mlflow.log_metric("val/loss", losses['val'], step=iter_num)
        mlflow.log_metric("train/loss_eval", losses['train'], step=iter_num)
        mlflow.log_metric("lr", lr, step=iter_num)

        if losses['val'] < best_val_loss or always_save_checkpoint:
            if losses['val'] < best_val_loss:
                logger.info(f"New overall best validation loss: {losses['val']:.4f}")
            best_val_loss = losses['val']

            checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
            ckpt_data = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'best_val_loss_early_stop': best_val_loss_early_stop,
                'patience_counter': patience_counter,
            }
            if ddp:
                ckpt_data['ddp_rank'] = ddp_rank
            
            # Synchronous safe save
            save_success = safe_save_checkpoint(ckpt_data, checkpoint_path)
            if save_success and master_process:
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                logger.info(f"Checkpoint artifact logged to MLflow")
            else:
                logger.warning("Checkpoint save failed—continuing without backup")

        if should_stop_training:
            logger.info("Breaking main training loop due to early stopping.")
            break

    if iter_num == 0 and eval_only:
        logger.info("Eval only mode, exiting after initial eval.")
        break

    model.train()
    optimizer.zero_grad(set_to_none=True)

    lossf = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
            lossf += loss.detach()

        scaler.scale(loss).backward()

        X, Y = get_batch('train')

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if master_process and iter_num % log_interval == 0:
            mlflow.log_metric("grad_norm", grad_norm.item(), step=iter_num)

    scaler.step(optimizer)
    scaler.update()

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        if local_iter_num >= 5:
            try:
                # Fix: Manual MFU estimation since raw_model might not have estimate_mfu
                step_time = dt / gradient_accumulation_steps
                mfu = batch_size * block_size * gradient_accumulation_steps * 100 / (step_time * num_params * 6)  # Simplified MFU
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            except Exception as e:
                logger.warning(f"Could not estimate MFU: {e}")
                running_mfu = -1.0
        logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        mlflow.log_metric("train/loss", lossf.item(), step=iter_num)
        if local_iter_num >= 5 and running_mfu != -1.0:
            mlflow.log_metric("mfu", running_mfu * 100, step=iter_num)

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters or should_stop_training:
        break

# Final save after loop
if master_process and (iter_num % eval_interval != 0):  # If didn't save in last eval
    checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
    ckpt_data = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
        'best_val_loss_early_stop': best_val_loss_early_stop,
        'patience_counter': patience_counter,
    }
    if ddp:
        ckpt_data['ddp_rank'] = ddp_rank
    save_success = safe_save_checkpoint(ckpt_data, checkpoint_path)
    if save_success:
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

# Register the model with MLflow at the end
if master_process:
    final_status = "completed" if not should_stop_training else "stopped_early"
    logger.info(f"Training {final_status}.")
    try:
        mlflow.log_param("final_status", final_status)
        mlflow.log_metric("final_train_loss", lossf.item(), step=iter_num)
        mlflow.log_metric("final_val_loss", losses['val'], step=iter_num) if 'losses' in locals() else logger.warning("No final val loss to log")
        mlflow.log_metric("final_lr", lr, step=iter_num)
        mlflow.log_metric("final_mfu", running_mfu * 100, step=iter_num) if running_mfu != -1.0 else logger.warning("No final MFU to log")

        # Log the PyTorch model
        model_uri = mlflow.pytorch.log_model(
            pytorch_model=raw_model,
            artifact_path="nanogpt_model",
            registered_model_name="nanogpt_emergency_calls"  # Registered model name
        )
        logger.info(f"Model logged and registered as 'nanogpt_emergency_calls'. URI: {model_uri}")
        
        mlflow.end_run()
        logger.info("MLflow run ended.")
    except Exception as e:
        logger.warning(f"Could not end MLflow run or register model cleanly: {e}")
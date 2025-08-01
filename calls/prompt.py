import torch
from tokenizers import Tokenizer

# Load checkpoint
model_checkpoint_path = 'gpt_model_checkpoint.pth'
tokenizer_path = 'bpe_tokenizer_v2.json'

checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
vocab_size = checkpoint['model_state_dict']['transformer.wte.weight'].shape[0]
print(f"Model vocab size: {vocab_size}")

# Check tokenizer
try:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print("First 10 tokens in tokenizer:")
    vocab = tokenizer.get_vocab()
    for i, (token, id) in enumerate(list(vocab.items())[:10]):
        print(f"  {id}: '{token}'")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Check what's in the checkpoint
print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  {key}")

if 'model_state_dict' in checkpoint:
    print("\nModel state dict keys (first 10):")
    for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:10]):
        print(f"  {key}")
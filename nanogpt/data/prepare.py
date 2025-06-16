import os
import sys
import pickle
import numpy as np
import tiktoken
import time

def prepare(config):
    # Ensure dataset attribute exists
    if not hasattr(config, 'dataset'):
        raise ValueError("Config file must define 'dataset' attribute")

    # Use parent directory of script to resolve data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up one level from data/ to llm/
    data_dir = os.path.join(base_dir, 'data', config.dataset)
    input_file = os.path.join(data_dir, 'input.txt')

    # Debug: Print paths
    print(f"Script directory: {script_dir}")
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Input file path: {input_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}")

    # Initialize BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"Length of dataset in characters: {len(data):,}")

    # Tokenize with BPE
    print("Tokenizing with BPE...")
    start_time = time.time()
    encoded_data = np.array(tokenizer.encode(data), dtype=np.int64)
    print(f"Tokenization took {time.time() - start_time:.2f} seconds")
    print(f"Length of dataset in tokens: {len(encoded_data):,}")

    # Get vocabulary size
    vocab_size = tokenizer.n_vocab
    print("Vocabulary size:", vocab_size)

    # Split into train and validation
    n = len(encoded_data)
    train_data = encoded_data[:int(n * 0.9)]
    val_data = encoded_data[int(n * 0.9):]

    # Save tokenized data and metadata
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'train.bin'), 'wb') as f:
        np.array(train_data, dtype=np.uint16).tofile(f)
    with open(os.path.join(data_dir, 'val.bin'), 'wb') as f:
        np.array(val_data, dtype=np.uint16).tofile(f)
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': vocab_size}, f)

    print("Saved train.bin, val.bin, meta.pkl")

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].startswith("config_path="):
        print("Error: Please provide the config file path as 'config_path=/path/to/config.py'")
        sys.exit(1)
    
    config_path = sys.argv[1].split("=")[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at: {config_path}")
        sys.exit(1)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None:
        print(f"Error: Failed to load config file: {config_path}")
        sys.exit(1)
    
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    prepare(config)
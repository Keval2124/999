import os
import sys
import pickle
import numpy as np
import tiktoken
import time
import json 

def extract_text_from_json(obj):
    """
    Helper to extract text from a JSON object.
    Customize this based on your JSON structure.
    - If dict with 'text' key: return obj['text']
    - If list of dicts: concatenate all 'text' fields
    - If flat dict: join values as strings
    - Returns empty string if no text found.
    """
    if isinstance(obj, dict):
        if 'text' in obj:
            return str(obj['text'])  # Or 'content', 'description', etc.
        # Fallback: join all string values
        text_parts = [str(v) for v in obj.values() if isinstance(v, str)]
        return ' '.join(text_parts)
    elif isinstance(obj, list):
        # Recurse on list items
        texts = []
        for item in obj:
            texts.append(extract_text_from_json(item))
        return '\n\n'.join([t for t in texts if t.strip()])
    else:
        return str(obj)  # Last resort: stringify

def prepare(config):
    # Ensure required attributes exist
    if not hasattr(config, 'dataset'):
        raise ValueError("Config file must define 'dataset' attribute")
    if not hasattr(config, 'TEXT_DIR'):
        raise ValueError("Config file must define 'TEXT_DIR' attribute")
    # JSON_FILE is optional; we'll use it if needed

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

    # Create data_dir if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Generate input.txt if it doesn't exist
    if not os.path.exists(input_file):
        print(f"Input file not found. Generating from config sources...")
        
        data = ""
        
        # Option 1: Concatenate all .txt files from TEXT_DIR (assumed primary source)
        if os.path.exists(config.TEXT_DIR):
            print(f"Concatenating text files from {config.TEXT_DIR}")
            for filename in sorted(os.listdir(config.TEXT_DIR)):  # Sorted for reproducibility
                if filename.endswith('.txt'):
                    file_path = os.path.join(config.TEXT_DIR, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data += f.read() + "\n\n"  # Add separators between files
        else:
            print(f"Warning: TEXT_DIR {config.TEXT_DIR} does not exist.")
        
        print(f"Scanning for JSON files in {script_dir} to add to dataset...")
        json_files = [f for f in sorted(os.listdir(script_dir)) if f.endswith('.json')]
        if json_files:
            for filename in json_files:
                json_path = os.path.join(script_dir, filename)
                print(f"Processing {filename}...")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    extracted_text = extract_text_from_json(json_data)
                    if extracted_text.strip():
                        data += extracted_text + "\n\n"  # Add separator
                        print(f"Added {len(extracted_text):,} characters from {filename}")
                    else:
                        print(f"No extractable text found in {filename}")
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in {filename}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing {filename}: {e}")
        else:
            print("No JSON files found in script directory.")

        if not data and hasattr(config, 'JSON_FILE') and os.path.exists(config.JSON_FILE):
            print(f"Extracting text from {config.JSON_FILE}")
            with open(config.JSON_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if 'text' in entry:
                            data += entry['text'] + "\n\n"
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON line in {config.JSON_FILE}")
        
        if not data:
            raise ValueError("No data found in TEXT_DIR, JSON files, or JSON_FILE to generate input.txt")
        
        # Write the generated data to input.txt
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(data)
        print(f"Generated input.txt with {len(data):,} characters")

    # Proceed with original logic
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
'''
(999) [sgkshah@viz2[barkla2] nanogpt]$ python data/prepare.py "config_path=config/train_calls.py"
Script directory: /mnt/scratch/users/sgkshah/999/nanogpt/data
Base directory: /mnt/scratch/users/sgkshah/999/nanogpt
Data directory: /mnt/scratch/users/sgkshah/999/nanogpt/data/calls
Input file path: /mnt/scratch/users/sgkshah/999/nanogpt/data/calls/input.txt
Length of dataset in characters: 1,360,099
Tokenizing with BPE...
Tokenization took 0.10 seconds
Length of dataset in tokens: 409,708
Vocabulary size: 50257
Saved train.bin, val.bin, meta.pkl
'''    
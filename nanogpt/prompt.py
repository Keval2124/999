import torch
import os
import pickle
import tiktoken  # For BPE fallback; pip install if missing
import random  # For random simple emergencies
from model import GPTConfig, GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = 'nanogpt/ckpt.pt'
meta_path = 'data/calls/meta.pkl'  # From project root

# Check checkpoint
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint missing at {ckpt_path}.")

file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
print(f"Loading {ckpt_path} ({file_size_mb:.1f} MB)")

# Load checkpoint
try:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
except:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load meta and handle tokenization
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

if 'stoi' in meta and 'itos' in meta:
    # Char-level tokenization
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
    tokenizer_name = 'char'
    print("Using char-level tokenization from meta.pkl")
else:
    # BPE/tokenizer fallback (assume GPT-2 style)
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = meta.get('vocab_size', 50304)
    if enc.n_vocab != vocab_size:
        print(f"Warning: tiktoken GPT-2 vocab size {enc.n_vocab} != meta vocab_size {vocab_size}. Using GPT-2 anyway.")
    encode = enc.encode
    decode = enc.decode
    tokenizer_name = 'gpt2-bpe'
    print(f"Using GPT-2 BPE tokenization (vocab_size={vocab_size})")

# Simple random emergencies (no file dependency; pure strings for coherence)
random_emergencies = [
    "There's a fire in my kitchen! I can smell smoke everywhere and flames are spreading quickly.",
    "My chest hurts badly, I can't breathe, and I'm feeling dizzy. I think it's a heart attack!",
    "Someone is breaking into my house! I hear glass shattering downstairs.",
    "I've been in a car accident on the motorway. The car is smoking and I can't move.",
    "A child is choking on food at home. They're turning blue!"
]

# Interactive: Get user input or random
user_input = input("Enter emergency description (or press Enter for random): ").strip()
if not user_input:
    emergency_scenario = random.choice(random_emergencies)
    scenario_key = "random"
    print(f"Random scenario selected: {emergency_scenario}")
else:
    emergency_scenario = user_input
    scenario_key = "user-defined"
    if len(emergency_scenario) > 200:  # Validate length to avoid overload
        emergency_scenario = emergency_scenario[:200] + "..."
        print(f"Input truncated for generation: {emergency_scenario}")

# Improved prompt with instructions for coherence
prompt = f"""You are generating a realistic transcript of a 999 emergency call between two participants only: the OPERATOR (professional, calm, asking for details like address, situation, and dispatching help) and the CALLER (panicked but responding to questions). Alternate turns strictly (OPERATOR then CALLER). Use UK English. No stage directions like (sighs) or (crying). Keep it focused on the emergency. End when help is dispatched or the situation is resolved.

OPERATOR: 999, what's your emergency?
CALLER: {emergency_scenario}
OPERATOR:"""

print(f"\nGenerating 999 Emergency Call Script...")
print(f"Scenario: {emergency_scenario}\n")
print(f"Tokenizer: {tokenizer_name}\n")

context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    out = model.generate(
        context, 
        max_new_tokens=500,  # For a full back-and-forth dialogue
        temperature=0.6,     # Lower for more coherent output
        # If your model.generate supports these, uncomment after editing model.py:
        # top_k=40,
        # repetition_penalty=1.1
    )

response = decode(out[0].tolist())
generated_script = response[len(prompt):].strip()

# Post-process for better formatting and cleanup (split by speaker, remove repeats)
full_script = prompt + generated_script
lines = full_script.split('\n')
formatted_script = []
current_speaker = ""
repeat_count = 0
for line in lines:
    line = line.strip()
    if not line:
        continue
    upper_line = line.upper()
    if 'OPERATOR:' in upper_line or 'CALLER:' in upper_line:
        if current_speaker and len(current_speaker) > 10:  # Avoid empty lines
            formatted_script.append(current_speaker.strip())
        current_speaker = line
        repeat_count = 0
    elif current_speaker and len(line) > 3 and repeat_count < 3:  # Add to speaker if not repetitive
        current_speaker += " " + line
        repeat_count += 1 if line in current_speaker else 0  # Simple repeat check
    else:
        repeat_count = 0  # Reset if new

if current_speaker and len(current_speaker) > 10:
    formatted_script.append(current_speaker.strip())

print("=== GENERATED 999 EMERGENCY CALL SCRIPT ===\n")
print('\n'.join(formatted_script[:20]))  # Limit to first 20 lines for brevity; adjust as needed

# Save to file
output_file = f'generated_999_call_{scenario_key.replace(" ", "_")}.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(formatted_script))
print(f"\nFull script saved to {output_file}")
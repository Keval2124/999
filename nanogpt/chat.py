import torch
import os
import pickle
import tiktoken
import json
import random
import re
from model import GPTConfig, GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = 'nanogpt/ckpt.pt'
meta_path = 'data/calls/meta.pkl'

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
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
    tokenizer_name = 'char'
    print("Using char-level tokenization from meta.pkl")
else:
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = meta.get('vocab_size', 50257)
    if enc.n_vocab != vocab_size:
        print(f"Warning: tiktoken GPT-2 vocab size {enc.n_vocab} != meta vocab_size {vocab_size}. Using GPT-2 anyway.")
    encode = enc.encode
    decode = enc.decode
    tokenizer_name = 'gpt2-bpe'
    print(f"Using GPT-2 BPE tokenization (vocab_size={vocab_size})")

# Load emergency codes for random scenarios
emergency_codes_path = 'data/uk_emergency_codes.json'
scenario_categories = []

if os.path.exists(emergency_codes_path):
    with open(emergency_codes_path, 'r') as f:
        codes_data = json.load(f)
    
    # Extract realistic emergency scenarios from the codes
    if 'police_codes' in codes_data and 'ten_codes' in codes_data['police_codes']:
        for code, description in codes_data['police_codes']['ten_codes']['codes'].items():
            if any(keyword in description.lower() for keyword in ['accident', 'robbery', 'shooting', 'assault', 'burglary']):
                scenario_categories.append(f"Police incident: {description}")
    
    if 'ambulance_codes' in codes_data and 'response_categories' in codes_data['ambulance_codes']:
        for category, details in codes_data['ambulance_codes']['response_categories']['codes'].items():
            if 'description' in details:
                scenario_categories.append(f"Medical emergency: {details['description']}")
    
    if 'fire_service_codes' in codes_data and 'fire_response_codes' in codes_data['fire_service_codes']:
        if 'priority_codes' in codes_data['fire_service_codes']['fire_response_codes']:
            for code, description in codes_data['fire_service_codes']['fire_response_codes']['priority_codes'].items():
                if 'fire' in description.lower() or 'emergency' in description.lower():
                    scenario_categories.append(f"Fire emergency: {description}")
    
    if 'coastguard_codes' in codes_data:
        scenario_categories.append("Maritime emergency: Vessel in distress at sea")
        scenario_categories.append("Coastal emergency: Person in trouble in water")
    
    print(f"Loaded {len(scenario_categories)} scenario categories from UK emergency codes.")
else:
    # Fallback scenarios if codes file is missing
    scenario_categories = [
        "Police incident: There's been a burglary at my house!",
        "Medical emergency: My husband is having a heart attack!",
        "Fire emergency: There's a fire in my kitchen!",
        "Coastal emergency: I see someone in trouble in the water!"
    ]

# Interactive: Get user input or random
user_input = input("Enter emergency description (or press Enter for random): ").strip()
if user_input:
    emergency_scenario = user_input
    scenario_key = "user-defined"
else:
    emergency_scenario = random.choice(scenario_categories)
    scenario_key = "random"

# Extract emergency type for more focused prompting
emergency_type = "general"
if "police" in emergency_scenario.lower():
    emergency_type = "police"
elif "medical" in emergency_scenario.lower():
    emergency_type = "medical"
elif "fire" in emergency_scenario.lower():
    emergency_type = "fire"
elif "coastal" in emergency_scenario.lower() or "maritime" in emergency_scenario.lower():
    emergency_type = "coastguard"

# More specific and constrained prompt
prompt = f"""Generate a realistic UK 999 emergency call between an OPERATOR and a CALLER.
Focus strictly on this emergency: {emergency_scenario}

Rules:
1. Stay focused on this specific emergency type
2. The operator should ask appropriate questions for this emergency type
3. The caller should provide consistent information
4. Do not introduce unrelated emergencies or change the scenario
5. End the call naturally when help is dispatched

OPERATOR: 999, what's your emergency?
CALLER: {emergency_scenario}
OPERATOR:"""

print(f"\nGenerating 999 Emergency Call Script...")
print(f"Scenario: {emergency_scenario}\n")
print(f"Emergency Type: {emergency_type}\n")
print(f"Tokenizer: {tokenizer_name}\n")

context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    out = model.generate(
        context, 
        max_new_tokens=300,
        temperature=0.6,  # Lower temperature for more focused output
        top_k=30,         # Lower top_k for more focused output
        repetition_penalty=1.2  # Add repetition penalty if supported
    )

response = decode(out[0].tolist())
generated_script = response[len(prompt):].strip()

# Enhanced post-processing with consistency checks
full_script = prompt + generated_script
lines = full_script.split('\n')
formatted_script = []
current_speaker = ""
ending_phrases = ["thank you", "goodbye", "bye", "help is on the way", "assistance is coming", "stay on the line"]
call_ended = False

# Track the emergency type to detect inconsistencies
expected_emergency_type = emergency_type
inconsistency_detected = False

for line in lines:
    line = line.strip()
    
    # Check for inconsistencies in the emergency type
    if "police" in line.lower() and expected_emergency_type != "police":
        inconsistency_detected = True
    elif "medical" in line.lower() and expected_emergency_type != "medical":
        inconsistency_detected = True
    elif "fire" in line.lower() and expected_emergency_type != "fire":
        inconsistency_detected = True
    elif ("coastguard" in line.lower() or "maritime" in line.lower()) and expected_emergency_type != "coastguard":
        inconsistency_detected = True
    
    # If inconsistency is detected, stop processing
    if inconsistency_detected:
        break
    
    # Check if this line indicates the call should end
    if 'OPERATOR:' in line.upper() or 'CALLER:' in line.upper():
        speaker_text = line.split(':', 1)[1].strip().lower() if ':' in line else ""
        if any(phrase in speaker_text for phrase in ending_phrases):
            call_ended = True
    
    if call_ended:
        break
        
    if 'OPERATOR:' in line.upper() or 'CALLER:' in line.upper():
        if current_speaker:
            formatted_script.append(current_speaker.strip())
        current_speaker = line
    elif current_speaker:
        current_speaker += " " + line

if current_speaker and not call_ended and not inconsistency_detected:
    formatted_script.append(current_speaker.strip())

# If inconsistency was detected, use a fallback script
if inconsistency_detected:
    formatted_script = [
        "OPERATOR: 999, what's your emergency?",
        f"CALLER: {emergency_scenario}",
        "OPERATOR: Okay, I need your exact location first.",
        "CALLER: [Location details would be provided here]",
        "OPERATOR: And what's happening right now?",
        "CALLER: [Description of the emergency]",
        "OPERATOR: Help is on the way. Please stay where you are and wait for emergency services."
    ]
    print("Note: Generated script was inconsistent. Using fallback script.")

# Add a proper ending if the call didn't end naturally
elif not call_ended and formatted_script:
    last_line = formatted_script[-1]
    if 'OPERATOR:' in last_line.upper():
        formatted_script.append("OPERATOR: Help is on the way. Please stay where you are and wait for emergency services.")
    elif 'CALLER:' in last_line.upper():
        formatted_script.append("OPERATOR: Understood. Emergency services have been dispatched. Please stay on the line until they arrive.")

# Filter out any lines that don't match the expected pattern
filtered_script = []
for line in formatted_script:
    if re.match(r'^(OPERATOR|CALLER):', line, re.IGNORECASE):
        filtered_script.append(line)

print("=== GENERATED 999 EMERGENCY CALL SCRIPT ===\n")
print('\n'.join(filtered_script))

# Save to file
output_file = f'generated_999_call_{scenario_key.replace(" ", "_")}.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(filtered_script))
print(f"\nScript saved to {output_file}")
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import torch  # Added explicit import for torch.no_grad()

# Load the fine-tuned merged model and tokenizer
merged_model_path = "tinyllm-merged"  # Update if needed, e.g., "/mnt/scratch/users/sgkshah/999/tinyllm-merged"

model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# Set pad_token_id if not already set (common for Llama models)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the UK emergency codes JSON and create condensed summary (as before, to avoid token overflow)
with open('uk_emergency_codes.json', 'r', encoding='utf-8') as f:
    uk_codes_json = json.load(f)

summary = """
UK Emergency Codes Summary (use optionally for realism in dialogue, e.g., operator referencing '10-50 for vehicle accident' or 'Category 2 response'):

Police 10-Codes (relevant to accidents):
- 10-50: Vehicle Accident
- 10-52: Dispatch Ambulance
- 10-54: Hit and Run Accident
- 10-55: Intoxicated Driver
- 10-17: En route
- 10-23: Arrived at Scene
- 10-33: Need Immediate Assistance

Ambulance Response Categories (NHS UK):
- Category 1: Life-threatening (e.g., unconscious, major trauma); 7 min target, lights/sirens.
- Category 2: Emergency (e.g., road traffic collisions, chest pain); 18 min average, lights/sirens.
- Category 3: Urgent (e.g., falls, fractures); 120 min target.
- Category 4: Less urgent; 180 min target.

Police State Codes:
- State 5: En route to incident
- State 6: At scene

Fire/Other: Code 3 for life-threatening (lights/sirens). General: 999 for emergencies.
"""

# Base instructions (shared)
base_instructions = (
    f"You are assisting in generating a realistic UK emergency call transcript. "
    f"You have access to this UK emergency service codes summary for optional reference (use it if it enhances realism naturally, "
    f"e.g., operator mentioning 'Category 2' for urgency or '10-50' for accident, but only if it fits—do not force it; prioritize natural dialogue):\n\n"
    f"{summary}\n\n"
    "This is a transcript of a 999 emergency call between only two participants: "
    "the OPERATOR and the CALLER. Respond using only these two roles. Do not include stage directions like (sighs) or (smiling). "
    "Keep the dialogue realistic and focused on the emergency. Optionally incorporate relevant details from the summary above "
    "if it fits naturally, but it's entirely up to you. Continue the conversation naturally until the emergency is resolved "
    "and the operator ends the call (e.g., with 'thank you', 'goodbye', or 'help is on the way').\n\n"
)

# Initial prompt (starts the transcript)
initial_prompt = base_instructions + "OPERATOR: 999, what's your emergency?\nCALLER:"

# Full generated text (starts empty, will build incrementally)
full_text = initial_prompt

# Generation parameters for chunks (small to allow checking after each)
gen_params = {
    "max_new_tokens": 100,      # Small chunks for incremental generation and checking
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 0.85,
    "repetition_penalty": 1.3,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

# Define end conditions: Keywords that indicate the operator is ending the call
end_keywords = ["thank you", "goodbye", "bye", "hang up", "call ended", "help is on the way", "units dispatched", "ambulance en route"]
# Regex to detect if last speaker is OPERATOR and contains an end keyword (case-insensitive)
end_pattern = re.compile(r"OPERATOR:\s*(.*?)(?=\n|$)", re.IGNORECASE | re.DOTALL)

# NEW: End-encouraging instruction to append when approaching max iterations
end_instruction = "\n\n[SYSTEM NOTE: The call has been ongoing; now wrap up the conversation naturally. Have the OPERATOR end the call soon with a goodbye, thank you, or confirmation that help is dispatched.]"

# Loop for incremental generation until end condition met
max_iterations = 10  # Safety limit to prevent infinite loop
iteration = 0

while iteration < max_iterations:
    # NEW: Add end-encouraging logic starting from iteration 8 (so by 9th, it's influenced)
    current_prompt = full_text
    if iteration >= 8:
        current_prompt += end_instruction
        print(f"Iteration {iteration + 1}: Added end instruction to prompt.")
    
    # Token check: Ensure current prompt fits in model limit (truncate history if needed)
    prompt_tokens = len(tokenizer.encode(current_prompt))
    if prompt_tokens > 1800:  # Leave buffer for generation (model max ~2048)
        print(f"Warning: Prompt too long ({prompt_tokens} tokens). Truncating early history.")
        # Simple truncate: Keep instructions + last 1000 tokens worth of dialogue
        # Estimate: Keep from a point ~1000 tokens back
        truncated_text = base_instructions
        remaining_tokens = 1500 - len(tokenizer.encode(truncated_text))
        # Find a cut point in the dialogue
        dialogue_start = full_text.find("OPERATOR: 999")
        if dialogue_start != -1:
            dialogue = full_text[dialogue_start:]
            # Rough truncate: Take last characters equivalent to remaining tokens (approx 4 chars/token)
            max_chars = remaining_tokens * 4
            truncated_dialogue = dialogue[-max_chars:] if len(dialogue) > max_chars else dialogue
            current_prompt = truncated_text + truncated_dialogue
        prompt_tokens = len(tokenizer.encode(current_prompt))
    
    print(f"Iteration {iteration + 1}: Generating chunk... (prompt tokens: {prompt_tokens})")
    
    # Generate next chunk using direct model.generate for better control
    inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():  # For efficiency
        outputs = model.generate(**inputs, **gen_params)
    
    # Decode the new tokens (exclude the input prompt)
    new_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    if not new_text:  # If no new text (e.g., EOS hit early), break
        break
    
    # Append to full text (note: we append to full_text, not current_prompt, to avoid duplicating the end_instruction in output)
    full_text += new_text
    
    # Clean up any trailing incomplete parts (e.g., partial sentences)
    lines = full_text.split('\n')
    while lines and (lines[-1].strip() in ["OPERATOR:", "CALLER:"] or not lines[-1].strip()):
        lines.pop()
    full_text = '\n'.join(lines) + '\n' if lines else full_text
    
    # Check for end condition: Look for last OPERATOR line containing end keywords
    match = end_pattern.search(full_text)
    if match:
        last_operator_line = match.group(1).lower()
        if any(keyword in last_operator_line for keyword in end_keywords):
            print("End condition met: Operator ended the call.")
            break
    
    iteration += 1

# Final cleanup: Ensure it ends properly (remove any trailing system note if accidentally appended)
if end_instruction in full_text:
    full_text = full_text.replace(end_instruction, "").strip()

full_text = full_text.strip()

# Print token length for final output
final_tokens = len(tokenizer.encode(full_text))
print(f"Final transcript token length: {final_tokens}")

# Save output
with open('999_UK_Caller_Complete.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

# Print for viewing
print("\n" + "="*50)
print("Generated Transcript:")
print("="*50)
print(full_text)
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned merged model and tokenizer
merged_model_path = "tinyllm-merged"  # Update if needed, e.g., "/mnt/scratch/users/sgkshah/999/tinyllm-merged"

model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# Set pad_token_id if not already set (common for Llama models)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Create the text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = (
    "This is a transcript of a 911 emergency call between only two participants: "
    "the OPERATOR and the CALLER. Respond using only these two roles. Do not include stage directions like (sighs) or (smiling). Keep the dialogue realistic and focused on the emergency.\n\n"
    "OPERATOR: 911, what's your emergency?\n"
    "CALLER:"
)


result = generator(
    prompt,
    max_new_tokens=200,       # Generate up to 200 new tokens (adjust to control length)
    do_sample=True,           # Enable sampling for variety
    temperature=0.5,          # Lower for more focused, realistic outputs
    top_p=0.85,               # Nucleus sampling for better diversity
    repetition_penalty=1.2,   # Penalize repetitions (e.g., sighs, smiles)
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

print(result[0]['generated_text'])
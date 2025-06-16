from transformers import pipeline

# Assuming `model` and `tokenizer` are already loaded

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = (
    "This is a transcript of a 911 emergency call between only two participants: "
    "the OPERATOR and the CALLER. Respond using only these two roles.\n\n"
    "OPERATOR: 911, what's your emergency?\n"
    "CALLER:"
)

result = generator(
    prompt,
    max_length=2000,          # Total token length, including prompt
    do_sample=True,           # Enable sampling
    temperature=0.6,          # Lower = more deterministic
    top_p=0.9,                # Nucleus sampling
    eos_token_id=tokenizer.eos_token_id,  # Optional but good for stopping
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
)

print(result[0]['generated_text'])

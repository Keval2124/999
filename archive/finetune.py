from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_path = "tinyllm"

base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "/content/tinyllm-lora")

# Merge and unload the adapter
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("/content/tinyllm-merged")
tokenizer.save_pretrained("/content/tinyllm-merged")

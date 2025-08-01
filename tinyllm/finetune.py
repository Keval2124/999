import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging
import os
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_merge_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    low_memory_mode: bool = False,
    trust_remote_code: bool = False
) -> None:
    """
    Load a base model, apply a LoRA adapter, merge, and save the result.

    Args:
        base_model_path: Path to the base model.
        lora_adapter_path: Path to the LoRA adapter.
        output_path: Path to save the merged model.
        torch_dtype: Data type for model loading (e.g., torch.float16).
        device_map: Device placement strategy (e.g., 'auto').
        low_memory_mode: Enable low-memory loading for large models.
        trust_remote_code: Allow execution of remote code from model hub.
    """
    try:
        logger.info(f"Loading base model from {base_model_path}")
        # Load base model with optimized settings
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_memory_mode,
            trust_remote_code=trust_remote_code
        )

        logger.info(f"Loading tokenizer from {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code
        )

        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

        logger.info("Merging adapter with base model")
        # Merge and unload adapter
        merged_model = model.merge_and_unload()

        # Free memory
        del model
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info(f"Saving merged model to {output_path}")
        # Save merged model and tokenizer
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)

        logger.info("Model merging and saving completed successfully")

    except Exception as e:
        logger.error(f"Error during model loading/merging: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    config = {
        "base_model_path": "tinyllm",
        "lora_adapter_path": "tinyllm-lora",
        "output_path": "tinyllm-merged",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "low_memory_mode": True,
        "trust_remote_code": False
    }
    load_and_merge_model(**config)
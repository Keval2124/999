from huggingface_hub import snapshot_download

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  

try:
    snapshot_download(
        repo_id=model_name,
        local_dir="tinyllm",
        resume_download=True
    )
    print("Download")
except Exception as e:
    print(f"Error during download: {e}")

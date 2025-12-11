import os
import sys
from huggingface_hub import snapshot_download

if len(sys.argv) < 2:
    print("Usage: python download_model.py <model_id>")
    exit(1)

model_id = sys.argv[1]
hf_token = os.environ.get("HF_TOKEN")  # from CodeBuild env variable
local_path = f"models/{model_id.replace('/', '_')}"

print(f"Downloading model: {model_id}")

snapshot_download(
    repo_id=model_id,
    local_dir=local_path,
    local_dir_use_symlinks=False,
    token=hf_token,
    resume_download=True
)

print(f"Model is downloaded to {local_path}")

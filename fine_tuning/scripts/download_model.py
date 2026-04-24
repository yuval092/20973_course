import os
from huggingface_hub import hf_hub_download

# Create directories
os.makedirs('models', exist_ok=True)

# Download sac-FetchPickAndPlace-v4.zip
try:
    model_path = hf_hub_download(
        repo_id='IntelliGrow/FetchPickAndPlace-v4',
        filename='sac-FetchPickAndPlace-v4.zip',
        local_dir='models'
    )
    print(f'Model downloaded to: {model_path}')
except Exception as e:
    print(f'Error downloading model: {e}')
    exit(1)

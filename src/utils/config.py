import yaml
import os

def load_config(config_name: str) -> dict:
    # Resolve path relative to this file
    # This assumes the file is at src/utils/config.py
    # So .. goes to src/, and ../.. goes to root.
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', f'{config_name}.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

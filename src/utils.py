import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)

def create_directory(path):
    os.makedirs(path, exist_ok=True)

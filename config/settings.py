from pathlib import Path
import yaml

def load_config():
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / 'config.yml'
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
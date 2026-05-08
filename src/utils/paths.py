import yaml
from pathlib import Path


def load_paths(config_path: str):
    """Загружает пути из paths.yaml и возвращает как dict Path."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    return {k: Path(v) for k, v in data.items()}

import os
import zipfile
import yaml
from pathlib import Path


def extract_zip_from_drive(zip_path: Path, extract_dir: Path) -> None:
    """Распаковывает zip из Google Drive в указанную папку."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
        zip_ref.extractall(str(extract_dir))


def load_yaml(p: Path) -> dict:
    """Загружает data.yaml."""
    with open(p) as f:
        return yaml.safe_load(f)

import os
import shutil
from pathlib import Path


def copy_files_to_folder(src_dir: Path, dst_dir: Path) -> None:
    """Копирует все файлы из src_dir в dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file_path in src_dir.iterdir():
        if file_path.is_file():
            shutil.copy(file_path, dst_dir)

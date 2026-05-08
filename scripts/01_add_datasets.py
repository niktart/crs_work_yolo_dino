
import os
import zipfile
import argparse
from pathlib import Path


DRIVE_DIR = "/content/drive/MyDrive/term_work/combined_dataset"
EXTRACT_DIR = "/content/data"
image_extensions = {".jpg", ".jpeg", ".png"}


def count_images(root_dir):
    root = Path(root_dir)
    total = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in image_extensions:
            total += 1
    print(f"📸 Всего изображений: {total}")
    return total


def main(drive_dir=DRIVE_DIR, extract_dir=EXTRACT_DIR):
    needed_zips = {
        "random_dataset.v1i.yolov8.zip",
        "final_good_detection.v1i.yolov8.zip",
    }

    root = Path(extract_dir)
    root.mkdir(parents=True, exist_ok=True)

    for zip_name in needed_zips:
        zip_path = Path(drive_dir) / zip_name
        if not zip_path.exists():
            print(f"⚠️ Нет архива: {zip_name}")
            continue

        ds_name = zip_name.replace(".zip", "")
        extract_path = root / ds_name
        extract_path.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)
        print(f"✅ Распакован: {zip_name}")

    count_images(extract_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--drive_dir", default=DRIVE_DIR)
    p.add_argument("--extract_dir", default=EXTRACT_DIR)
    args = p.parse_args()
    main(args.drive_dir, args.extract_dir)

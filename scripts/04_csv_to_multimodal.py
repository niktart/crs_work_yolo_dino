
"""
Конвертация train/val_coco.json в train/val_annotations.csv
Копирование train/val/images в multimodal-data
"""


import os
import json
import csv
from collections import defaultdict
import shutil


def coco_to_csv(coco_json_path, output_csv_path):
    with open(coco_json_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "label_name",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "image_name"
        ])

        for ann in coco["annotations"]:
            image_info = images[ann["image_id"]]
            category_name = categories[ann["category_id"]]
            x, y, w, h = ann["bbox"]

            writer.writerow([
                category_name,
                int(x),
                int(y),
                int(w),
                int(h),
                image_info["file_name"]
            ])


if __name__ == "__main__":
    DATA_ROOT = "/content/data/combined_clean_bbox"

    train_json = os.path.join(DATA_ROOT, "train_coco.json")
    val_json = os.path.join(DATA_ROOT, "val_coco.json")

    train_csv = os.path.join(DATA_ROOT, "train_annotations.csv")
    val_csv = os.path.join(DATA_ROOT, "val_annotations.csv")

    coco_to_csv(train_json, train_csv)
    coco_to_csv(val_json, val_csv)

    # === копирование в multimodal-data ===
    multimodal_root = os.path.join(DATA_ROOT, "multimodal-data")
    os.makedirs(multimodal_root, exist_ok=True)

    for split in ["train", "val"]:
        images_dir = os.path.join(multimodal_root, f"{split}_images")
        os.makedirs(images_dir, exist_ok=True)
        src_images = os.path.join(DATA_ROOT, split, "images")
        for file in os.listdir(src_images):
            shutil.copy(os.path.join(src_images, file), images_dir)

    # копирование CSV
    ann_dir = os.path.join(multimodal_root, "annotation")
    os.makedirs(ann_dir, exist_ok=True)
    shutil.copy(train_csv, os.path.join(ann_dir, "train_annotations.csv"))
    shutil.copy(val_csv, os.path.join(ann_dir, "val_annotations.csv"))

    print("✅ COCO → CSV и multimodal-data готовы")

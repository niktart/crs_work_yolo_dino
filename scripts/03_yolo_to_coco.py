
"""
Конвертация YOLO датасета в COCO JSON
Используется для подготовки данных для Grounding DINO.

Пусть DATASET_ROOT = "/content/data/combined_clean_bbox"
"""
import os
import json
import yaml
from PIL import Image
from tqdm import tqdm


def yolo_to_coco_bbox(xc, yc, w, h, img_w, img_h):
    x_min = (xc - w / 2) * img_w
    y_min = (yc - h / 2) * img_h
    width = w * img_w
    height = h * img_h
    return [x_min, y_min, width, height]


def convert_split(split_name, class_names):
    images = []
    annotations = []

    DATASET_ROOT = os.getenv("DATASET_ROOT", "/content/data/combined_clean_bbox")
    image_dir = os.path.join(DATASET_ROOT, split_name, "images")
    label_dir = os.path.join(DATASET_ROOT, split_name, "labels")

    ann_id = 1
    img_id = 1

    for img_file in tqdm(sorted(os.listdir(image_dir)), desc=f"Processing {split_name}"):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

        width, height = Image.open(img_path).size

        images.append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    bbox = yolo_to_coco_bbox(xc, yc, w, h, width, height)

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls + 1,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    })
                    ann_id += 1

        img_id += 1

    categories = [
        {
            "id": i + 1,
            "name": str(name)
        }
        for i, name in enumerate(class_names)
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


def main():
    DATASET_ROOT = os.getenv("DATASET_ROOT", "/content/data/combined_clean_bbox")
    SPLITS = ["train", "val", "test"]

    with open(os.path.join(DATASET_ROOT, "data.yaml")) as f:
        data_yaml = yaml.safe_load(f)

    class_names = data_yaml["names"]

    for split in SPLITS:
        split_path = os.path.join(DATASET_ROOT, split)
        if not os.path.isdir(split_path):
            continue

        coco_dict = convert_split(split, class_names)
        output_path = os.path.join(DATASET_ROOT, f"{split}_coco.json")

        with open(output_path, "w") as f:
            json.dump(coco_dict, f, indent=2)

        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

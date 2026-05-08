
"""
Fine‑tuning только bbox и class_embed слоёв GroundingDINO
"""


import os
import torch
import torch.optim as optim
import random
import csv
from collections import defaultdict
from groundingdino.models import build_model
from groundingdino.util.inference import load_image
from groundingdino.util.config import get_config


def read_dataset(ann_file, split='train'):
    ann_dict = defaultdict(lambda: defaultdict(list))

    if split == 'train':
        images_dir = "multimodal-data/train_images"
    else:
        images_dir = "multimodal-data/val_images"

    print(f"Читаем аннотации {split} из {ann_file}")
    print(f"Изображения из {images_dir}")

    with open(ann_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(images_dir, row["image_name"])
            if not os.path.exists(img_path):
                print(f"Файл не найден: {img_path}")
                continue
            x1 = int(row["bbox_x"])
            y1 = int(row["bbox_y"])
            w = int(row["bbox_width"])
            h = int(row["bbox_height"])
            x2 = x1 + w
            y2 = y1 + h
            label = row["label_name"]

            ann_dict[img_path]["boxes"].append([x1, y1, x2, y2])
            ann_dict[img_path]["captions"].append(label)

    print(f"Найдено {len(ann_dict)} уникальных изображений")
    return ann_dict


def train(model, ann_file, split='train', epochs=30, save_path='weights/model_', save_epoch=5):
    ann_dict = read_dataset(ann_file, split=split)

    print(f"Обучение на {len(ann_dict)} изображениях")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-6
    )

    model.train()
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        total_loss = 0
        processed_images = 0

        items = list(ann_dict.items())
        random.shuffle(items)

        for idx, (image_path, vals) in enumerate(items):
            try:
                image_source, image = load_image(image_path)

                boxes = vals['boxes']
                captions = vals['captions']

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    loss = train_image(
                        model=model,
                        image_source=image_source,
                        image=image,
                        caption_objects=captions,
                        box_target=boxes,
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                processed_images += 1

                if idx % 20 == 0:
                    print(f"[Epoch {epoch+1}] {idx}/{len(ann_dict)} Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Ошибка {image_path}: {e}")
                continue

        if processed_images > 0:
            avg_loss = total_loss / processed_images
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: 0 изображений")

        if (epoch + 1) % save_epoch == 0:
            torch.save(model.state_dict(), f"{save_path}_{split}_{epoch+1}.pth")


if __name__ == "__main__":
    config = get_config()
    model = build_model(config)

    # загружаем предобученные веса
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "weights", "groundingdino_swint_ogc.pth"
    )
    model.load_state_dict(torch.load(weights_path))

    # замораживаем всё, кроме bbox и class_embed
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "class_embed" in name or "bbox_embed" in name:
            param.requires_grad = True

    train(
        model=model,
        ann_file="multimodal-data/annotation/train_annotations.csv",
        split='train',
        epochs=30,
        save_path="weights/model",
        save_epoch=1
    )


import os
import cv2
import random
from pathlib import Path
import yaml
import albumentations as A
import numpy as np
from tqdm import tqdm


def main():
    TRAIN_IMG_DIR = Path("/content/data/combined_clean_bbox/train/images")
    TRAIN_LBL_DIR = Path("/content/data/combined_clean_bbox/train/labels")

    # Параметры (можно вынести в configs/augmentation.yaml)
    MIN_IMAGES = 250

    # Трансформации
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo", min_visibility=0.1, label_fields=["class_labels"]
        ),
    )

    # 1. Собираем классы
    cfg = yaml.safe_load((Path("/content/data/combined_clean_bbox/data.yaml")).read_text())
    class_names = cfg["names"]

    # Считаем изображения по классам
    class_images = {}

    for lbl_file in os.listdir(TRAIN_LBL_DIR):
        if not lbl_file.endswith(".txt"):
            continue

        img_file = lbl_file.replace(".txt", ".jpg")
        img_path = TRAIN_IMG_DIR / img_file
        if not img_path.is_file():
            continue

        with open(TRAIN_LBL_DIR / lbl_file) as f:
            for line in f:
                cls_id = int(float(line.split()[0]))
                if cls_id not in class_images:
                    class_images[cls_id] = set()
                class_images[cls_id].add(img_path)

    # 2. Аугментация
    for cls_id, cls_name in enumerate(class_names):
        current_count = len(class_images.get(cls_id, set()))
        needed = max(0, MIN_IMAGES - current_count)

        if needed == 0:
            print(f"✅ {cls_name}: {current_count} изображений")
            continue

        print(f"⚠️ {cls_name}: {current_count} → нужно {needed} аугментаций")

        # Собираем все примеры класса
        class_samples = []

        for img_path in class_images[cls_id]:
            lbl_name = img_path.with_suffix(".txt").name
            lbl_path = TRAIN_LBL_DIR / lbl_name
            if not lbl_path.is_file():
                continue

            bboxes = []
            classes = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_in_file = int(float(parts[0]))
                    if cls_in_file == cls_id:
                        bboxes.append([float(x) for x in parts[1:5]])
                        classes.append(cls_id)

            if bboxes:
                class_samples.append((img_path, bboxes, classes))

        # Аугментации
        aug_count = 0
        while aug_count < needed:
            random.shuffle(class_samples)

            for img_path, bboxes, classes in class_samples:
                if aug_count >= needed:
                    break

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                try:
                    augmented = transform(image=img, bboxes=bboxes, class_labels=classes)

                    # Ограничение bbox и фильтрация
                    safe_bboxes = []
                    safe_classes = []

                    for bbox, cls in zip(augmented["bboxes"], augmented["class_labels"]):
                        bbox = np.clip(bbox, 0, 1)
                        x, y, w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                        safe_bboxes.append([x, y, w, h])
                        safe_classes.append(cls)

                    if not safe_bboxes:
                        continue

                    aug_name = f"aug_{cls_name}_{aug_count}_{img_path.name}"
                    aug_path = TRAIN_IMG_DIR / aug_name
                    cv2.imwrite(
                        str(aug_path),
                        cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR),
                    )

                    lbl_name = aug_path.with_suffix(".txt").name
                    with open(TRAIN_LBL_DIR / lbl_name, "w") as f:
                        for bbox, cls in zip(safe_bboxes, safe_classes):
                            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

                    aug_count += 1

                except Exception as e:
                    print(f"Ошибка аугментации для {img_path}: {e}")
                    continue

    print("✅ Аугментация завершена!")


if __name__ == "__main__":
    main()

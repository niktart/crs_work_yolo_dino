import os
import cv2
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from pathlib import Path


def load_labels_for_dataset(ds_path: Path):
    """Собирает изображения по классам для конкретного датасета."""
    images_per_class = defaultdict(list)  # cls_name -> [(img_path, ds)]

    img_dir = ds_path / "val" / "images"
    lbl_dir = ds_path / "val" / "labels"

    if not lbl_dir.is_dir():
        return images_per_class

    class_names = []
    yaml_path = ds_path / "data.yaml"
    if yaml_path.is_file():
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        class_names = data.get("names", [])

    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue

        lbl_path = lbl_dir / lbl_file
        img_file = lbl_file.replace(".txt", ".jpg")
        img_path = img_dir / img_file
        if not img_path.is_file():
            continue

        with open(lbl_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id >= len(class_names):
                continue
            cls_name = class_names[cls_id]

            images_per_class[cls_name].append((img_path, ds_path.name))

    return images_per_class


def main():
    IMAGES_PER_CLASS = 6
    EXTRACT_DIR = Path("/content/data")

    all_classes = set()
    dataset_classes = {}

    for ds in os.listdir(EXTRACT_DIR):
        ds_path = EXTRACT_DIR / ds
        yaml_path = ds_path / "data.yaml"
        if not yaml_path.is_file():
            continue

        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        dataset_classes[ds] = data.get("names", [])
        for cls in data.get("names", []):
            all_classes.add(cls.strip())

    all_classes = sorted(all_classes)

    # Нормализация и поиск похожих названий
    def normalize(s):
        return s.lower().replace("-", " ").replace("_", " ").strip()

    normalized_classes = {cls: normalize(cls) for cls in all_classes}
    substring_groups = defaultdict(set)
    SIMILARITY_THRESHOLD = 0.8

    from difflib import SequenceMatcher

    for cls_a, norm_a in normalized_classes.items():
        for cls_b, norm_b in normalized_classes.items():
            if cls_a == cls_b:
                continue
            if len(norm_a) < 4 or len(norm_b) < 4:
                continue

            if norm_a in norm_b or norm_b in norm_a:
                substring_groups[norm_a].add(cls_a)
                substring_groups[norm_a].add(cls_b)
                continue

            similarity = SequenceMatcher(None, norm_a, norm_b).ratio()
            if similarity >= SIMILARITY_THRESHOLD:
                substring_groups[norm_a].add(cls_a)
                substring_groups[norm_a].add(cls_b)

    substring_groups = {
        k: sorted(v) for k, v in substring_groups.items() if len(v) > 1
    }

    if not substring_groups:
        print("✅ Нет спорных групп для визуальной проверки.")
        return

    # Глобально собираем все изображения по классам
    images_per_class = defaultdict(list)

    for ds_path in EXTRACT_DIR.iterdir():
        if not ds_path.is_dir():
            continue
        images_per_class.update(
            load_labels_for_dataset(ds_path)
        )

    # Визуализация
    for base, classes in substring_groups.items():
        print(f"\n🧩 СПОРНАЯ ГРУППА: '{base}'")

        total_classes = len(classes)
        plt.figure(figsize=(16, 4 * total_classes))

        for row_idx, cls_name in enumerate(classes):
            imgs = images_per_class.get(cls_name, [])
            if not imgs:
                print(f"⚠️ Нет изображений для класса: {cls_name}")
                continue

            imgs = random.sample(imgs, min(IMAGES_PER_CLASS, len(imgs)))

            for col_idx, (img_path, ds_name) in enumerate(imgs):
                plt.subplot(total_classes, IMAGES_PER_CLASS, row_idx * IMAGES_PER_CLASS + col_idx + 1)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.title(f"{cls_name}\n{ds_name}", fontsize=8)
                plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

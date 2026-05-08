
import os
import json
import shutil
import yaml
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import argparse

from _constants import (
    EXTRACT_DIR,
    OUTPUT_DIR,
    SPLIT_JSON,
    rename_map,
    SPLIT_RATIOS,
    RANDOM_STATE
)

image_extensions = {".jpg", ".jpeg", ".png"}

def count_images(root_dir):
    root = Path(root_dir)
    total = sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions)
    print(f"📸 Всего изображений: {total}")
    return total

def load_samples(extract_dir):
    samples = []
    all_classes = set()

    print("🔍 Сканируем датасеты...")
    for ds in sorted(os.listdir(extract_dir)):
        ds_path = os.path.join(extract_dir, ds)
        yaml_path = os.path.join(ds_path, "data.yaml")

        if not os.path.isfile(yaml_path):
            print(f"⚠️ Нет data.yaml в {ds}")
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        class_list = [
            rename_map.get(name, name).lower()
            for name in data["names"]
        ]

        for split in ["train", "val", "test"]:
            img_dir = os.path.join(ds_path, split, "images")
            lbl_dir = os.path.join(ds_path, split, "labels")

            if not os.path.isdir(img_dir):
                continue

            for lbl_file in sorted(os.listdir(lbl_dir)):
                if not lbl_file.endswith(".txt"):
                    continue

                img_file = lbl_file.replace(".txt", ".jpg")
                img_path = os.path.join(img_dir, img_file)
                lbl_path = os.path.join(lbl_dir, lbl_file)

                if not os.path.isfile(img_path):
                    continue

                with open(lbl_path) as f:
                    lines = f.readlines()

                objects = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    cls_name = class_list[cls_id]
                    objects.append((cls_name, parts[1:]))
                    all_classes.add(cls_name)

                if objects:
                    samples.append((img_path, ds, objects))

    print(f"✅ Найдено {len(samples)} фото, {len(all_classes)} классов")
    return samples, sorted(list(all_classes))

def analyze_split_stats(splits, final_names):
    stats = {cls: {
        'train_bbox': 0, 'train_images': 0,
        'val_bbox': 0, 'val_images': 0,
        'test_bbox': 0, 'test_images': 0
    } for cls in final_names}

    stats['TOTAL'] = {
        'train_bbox': 0, 'train_images': 0,
        'val_bbox': 0, 'val_images': 0,
        'test_bbox': 0, 'test_images': 0
    }

    split_names = ['train', 'val', 'test']

    for split_name in split_names:
        split_samples = splits[split_name]
        img_set = set()

        for img_path, _, objects in split_samples:
            img_set.add(img_path)

            # уникальные классы в изображении
            classes_in_image = set()

            for cls_name, _ in objects:
                stats[cls_name][f'{split_name}_bbox'] += 1
                classes_in_image.add(cls_name)

            # считаем images по классам
            for cls_name in classes_in_image:
                stats[cls_name][f'{split_name}_images'] += 1

        # TOTAL
        stats['TOTAL'][f'{split_name}_images'] = len(img_set)
        stats['TOTAL'][f'{split_name}_bbox'] = sum(
            stats[cls][f'{split_name}_bbox'] for cls in final_names
        )

    # DataFrame
    rows = []
    for cls in final_names:
        row = stats[cls].copy()
        row['class_name'] = cls
        row['total_bbox'] = sum(stats[cls][f'{s}_bbox'] for s in split_names)
        row['total_images'] = sum(stats[cls][f'{s}_images'] for s in split_names)
        rows.append(row)

    total_row = stats['TOTAL'].copy()
    total_row['class_name'] = 'TOTAL'
    total_row['total_bbox'] = sum(total_row[f'{s}_bbox'] for s in split_names)
    total_row['total_images'] = sum(total_row[f'{s}_images'] for s in split_names)

    rows.append(total_row)

    df = pd.DataFrame(rows)

    df = df[['class_name', 'train_bbox', 'train_images',
             'val_bbox', 'val_images',
             'test_bbox', 'test_images',
             'total_bbox', 'total_images']]

    df_non_total = df[df['class_name'] != 'TOTAL'].sort_values('total_bbox', ascending=False)
    df_total = df[df['class_name'] == 'TOTAL']

    df = pd.concat([df_non_total.reset_index(drop=True), df_total], ignore_index=True)

    return df

def make_split(samples, final_names, split_json):
    """Создает или загружает разбиение датасета"""
    final_map = {cls: i for i, cls in enumerate(final_names)}

    X = [(img_path, ds, objects) for img_path, ds, objects in samples]
    y = []
    for _, _, objects in samples:
        label_vector = [0] * len(final_names)
        for cls_name, _ in objects:
            label_vector[final_map[cls_name]] = 1
        y.append(label_vector)

    if not os.path.exists(split_json):
        print("🔀 Создаем новое разбиение...")

        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - SPLIT_RATIOS["train"],
            random_state=RANDOM_STATE
        )
        train_idx, temp_idx = next(msss1.split(X, y))

        X_train = [X[i] for i in train_idx]
        X_temp = [X[i] for i in temp_idx]
        y_temp = [y[i] for i in temp_idx]

        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]),
            random_state=RANDOM_STATE
        )
        val_idx, test_idx = next(msss2.split(X_temp, y_temp))

        X_val = [X_temp[i] for i in val_idx]
        X_test = [X_temp[i] for i in test_idx]

        splits = {
            "train": X_train,
            "val": X_val,
            "test": X_test,
        }

        split_paths = {k: [item[0] for item in v] for k, v in splits.items()}
        with open(split_json, "w") as f:
            json.dump(split_paths, f, indent=2)
        print(f"💾 Разбиение сохранено: {split_json}")

    else:
        print("📂 Загружаем существующее разбиение...")
        with open(split_json) as f:
            split_paths = json.load(f)

        splits = {"train": [], "val": [], "test": []}
        path_to_sample = {s[0]: s for s in X}

        for split_name in splits:
            for path in split_paths[split_name]:
                if path in path_to_sample:
                    splits[split_name].append(path_to_sample[path])

    for split_name, split_samples in splits.items():
        print(f"  {split_name}: {len(split_samples)} файлов")

    return splits

def make_name(ds, img_path):
    base = Path(img_path).stem
    return f"{ds}__{base}"

def save_split(splits, output_dir, final_map):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in splits.items():
        img_out = output_path / split_name / "images"
        lbl_out = output_path / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Сохраняем {split_name} ({len(split_samples)} файлов)...")

        for img_path, ds, objects in tqdm(split_samples, desc=split_name):
            name = make_name(ds, img_path)

            shutil.copy(img_path, img_out / f"{name}.jpg")

            new_lines = []
            for cls_name, bbox in objects:
                cls_id = final_map[cls_name]
                new_lines.append(" ".join([str(cls_id)] + bbox))

            with open(lbl_out / f"{name}.txt", "w") as f:
                f.write("\n".join(new_lines) + "\n")

    count_images(output_dir)

def create_data_yaml(output_dir, final_names):
    output_path = Path(output_dir)

    data_yaml = {
        "path": str(output_path.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": final_names,
        "nc": len(final_names),
    }

    yaml_path = output_path / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w", encoding='utf-8') as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print(f"\n✅ data.yaml создан: {yaml_path}")

def main(extract_dir=EXTRACT_DIR, output_dir=OUTPUT_DIR, split_json=SPLIT_JSON):
    print("🚀 Начинаем разбиение датасета")

    samples, final_names = load_samples(extract_dir)
    final_map = {cls: i for i, cls in enumerate(final_names)}

    print(f"\n🎯 Финальные классы ({len(final_names)}):")
    for i, cls in enumerate(final_names):
        print(f"  {i}: {cls}")

    splits = make_split(samples, final_names, split_json)

    # 📊 Статистика
    print("\n📊 Финальная статистика:")
    print("=" * 120)
    stats_df = analyze_split_stats(splits, final_names)
    print(stats_df.to_string(index=False, float_format='%.0f'))
    print("=" * 120)

    save_split(splits, output_dir, final_map)
    create_data_yaml(output_dir, final_names)

    print(f"\n🎉 Готово! Датасет: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разбиение датасета + статистика + data.yaml")
    parser.add_argument("--extract_dir", default=EXTRACT_DIR, help="Директория с датасетами")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Выходная директория")
    parser.add_argument("--split_json", default=SPLIT_JSON, help="JSON с разбиением")
    args = parser.parse_args()

    main(args.extract_dir, args.output_dir, args.split_json)


import time
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLOWorld


def main(config_path):
    data_yaml = Path("/content/data/combined_clean_bbox/data.yaml")
    weights_out_dir = Path("/content/drive/MyDrive/term_work/weights/yolo_world")

    if not data_yaml.is_file():
        raise RuntimeError(f"Не найден data.yaml: {data_yaml}")

    if not config_path.is_file():
        raise RuntimeError(f"Не найден конфиг: {config_path}")

    weights_out_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем конфиг
    with open(config_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    print("📄 Загружен конфиг:")
    for k, v in train_cfg.items():
        print(f"  {k}: {v}")

    # Загружаем модель
    model = YOLOWorld("yolov8s-worldv2.pt")

    print("\n🚀 Запускаем обучение...")
    start_time = time.time()

    # Обучение с параметрами из YAML
    model.train(
        data=str(data_yaml),
        project=str(weights_out_dir),
        name="yolo_world_exp",
        **train_cfg
    )

    # Таймер
    elapsed_seconds = time.time() - start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)

    print(f"\n⏱️ Обучение заняло: {hours} ч {minutes} мин {seconds} сек")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOWorld с YAML конфигом")
    parser.add_argument(
        "--config",
        default="configs/yolo_world_train.yaml",
        help="Путь к YAML конфигу"
    )
    args = parser.parse_args()

    main(Path(args.config))

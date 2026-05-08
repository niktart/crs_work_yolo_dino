
import os
import torch
from groundingdino.util.inference import load_model


def main():
    repo_root = "/content/Grounding-Dino-FineTuning-main"

    config_path = os.path.join(
        repo_root,
        "groundingdino",
        "config",
        "GroundingDINO_SwinT_OGC.py"
    )

    weights_path = os.path.join(
        repo_root,
        "weights",
        "groundingdino_swint_ogc.pth"
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Не найден config: {config_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Не найдены веса: {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔄 Загружаем GroundingDINO...")
    model = load_model(config_path, weights_path, device=device)
    print(f"✅ Модель загружена на {device}")


if __name__ == "__main__":
    main()

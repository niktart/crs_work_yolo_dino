
"""
Генерация TEXT_PROMPT для GroundingDINO из data.yaml
"""
import os
import yaml


if __name__ == "__main__":
    DATASET_ROOT = os.getenv("DATASET_ROOT", "/content/data/combined_clean_bbox")

    with open(os.path.join(DATASET_ROOT, "data.yaml")) as f:
        data_yaml = yaml.safe_load(f)

    class_names = data_yaml["names"]
    class_names = [name.replace("_", " ") for name in class_names]
    TEXT_PROMPT = " . ".join(class_names) + " ."

    print("Всего классов:", len(class_names))
    print("TEXT_PROMPT:", TEXT_PROMPT)

    # Сохранение в файл, чтобы использовать в обучении и инференсе
    prompt_file = os.path.join(DATASET_ROOT, "text_prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(TEXT_PROMPT)


EXTRACT_DIR = "/content/data"
OUTPUT_DIR = "/content/data/combined_clean_bbox"
SPLIT_JSON = "/content/split.json"

rename_map = {
    "Apel": "apple",
    "Apel Fuji": "apple",
    "Apel Honey Crips": "apple",
    "Apel Malang": "apple",
    "Lemon": "lemon",
    "Tomat": "tomato",
}

SPLIT_RATIOS = {"train": 0.75, "val": 0.15, 'test': 0.1}

RANDOM_STATE = 42

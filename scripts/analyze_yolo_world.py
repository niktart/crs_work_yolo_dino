
import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import torch
import yaml

from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLOWorld



CONF_THRESHOLD = 0.5
MATCH_IOU = 0.1
IOU_THRESHOLD = 0.5
MAX_VISUALIZE = 15

ERROR_NAMES = {
    "fn": "Пропуски",
    "fp": "Лишние",
    "cls": "Не тот класс",
    "loc": "Локализация"
}

WEIGHTS_PATH = "/content/drive/MyDrive/term_work/compare/weights/clear_dataset/new_clear_food.pt"
DATASET_ROOT = "/content/data/combined_clean_bbox"

IMAGES_DIR = os.path.join(DATASET_ROOT, "val/images")
LABELS_DIR = os.path.join(DATASET_ROOT, "val/labels")



def compute_strict_statistics(strict_results):

    all_confidences = []
    all_ious = []

    for result in strict_results:

        gt_boxes = result["gt_boxes"]
        pred_boxes = result["pred_boxes"]
        pred_conf = result["pred_confidences"]

        matches, _, _ = match_predictions(gt_boxes, pred_boxes)

        for _, _, iou in matches:
            all_ious.append(iou)

        all_confidences.extend(pred_conf)

    mean_conf = np.mean(all_confidences) if len(all_confidences) > 0 else 0
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0

    stats_df = pd.DataFrame({
        "Metric": ["Mean Confidence", "Mean IoU"],
        "Value": [round(mean_conf, 4), round(mean_iou, 4)]
    })

    return stats_df


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions(gt_boxes, pred_boxes, match_iou=0.1):
    matched_gt = set()
    matched_pred = set()
    matches = []

    for i, gt in enumerate(gt_boxes):
        best_iou = 0.0
        best_j = -1

        for j, pred in enumerate(pred_boxes):
            if j in matched_pred:
                continue

            iou = calculate_iou(gt, pred)

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j != -1 and best_iou >= match_iou:
            matches.append((i, best_j, best_iou))
            matched_gt.add(i)
            matched_pred.add(best_j)

    return matches, matched_gt, matched_pred


def confused_pairs_to_df(confused_pairs, top_k=10):
    total_confusions = sum(confused_pairs.values())

    rows = []
    for i, ((true_cls, pred_cls), count) in enumerate(
        confused_pairs.most_common(top_k), start=1
    ):
        perc = (count / total_confusions * 100) if total_confusions > 0 else 0

        rows.append({
            "   True Class   ": true_cls,
            "   Pred Class   ": pred_cls,
            "   Count   ": count,
            "  Percentage": round(perc, 2)
        })

    return pd.DataFrame(rows)


def load_label(label_path):
    gt_boxes, gt_classes = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:5])
                    gt_boxes.append([x_center, y_center, w, h])
                    gt_classes.append(cls_id)
    return np.array(gt_boxes), np.array(gt_classes)


def yolo_to_bbox(yolo_box, img_w, img_h):
    x_c, y_c, w, h = yolo_box
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]


def process_image(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]

    gt_boxes_yolo, gt_classes = load_label(label_path)
    gt_boxes = [yolo_to_bbox(b, w, h) for b in gt_boxes_yolo]

    with torch.no_grad():
        preds = model.predict(
            image_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=50,
            verbose=False
        )[0]

    pred_boxes, pred_classes, pred_confidences = [], [], []
    if preds.boxes is not None:
        for box in preds.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            if conf >= CONF_THRESHOLD:
                pred_boxes.append([x1, y1, x2, y2])
                pred_classes.append(cls)
                pred_confidences.append(conf)

    matches, matched_gt, matched_pred = match_predictions(gt_boxes, pred_boxes)

    errors = {'fn': [], 'fp': [], 'loc': [], 'cls': []}

    errors['fn'] = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    errors['fp'] = [j for j in range(len(pred_boxes)) if j not in matched_pred]

    for i, j, iou in matches:
        if iou < IOU_THRESHOLD:
            errors['loc'].append((i, j))
        elif gt_classes[i] != pred_classes[j]:
            errors['cls'].append((i, j))

    return {
        "image_path": image_path,
        "gt_boxes": gt_boxes,
        "gt_classes": gt_classes,
        "pred_boxes": pred_boxes,
        "pred_classes": pred_classes,
        "pred_confidences": pred_confidences,
        "matches": matches,
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "errors": errors
    }


def run_full_analysis():
    image_files = sorted([f for f in os.listdir(IMAGES_DIR)
                          if f.endswith(('.jpg', '.png', '.jpeg'))])

    results = []

    for img_file in tqdm(image_files, desc="Processing ALL"):
        image_path = os.path.join(IMAGES_DIR, img_file)
        label_path = os.path.join(LABELS_DIR, Path(img_file).with_suffix(".txt"))

        if not os.path.exists(label_path):
            continue

        result = process_image(image_path, label_path)
        if result is not None:
            results.append(result)

    print(f"\nВсего изображений: {len(results)}")
    return results


def analyze_results(results):
    error_images = {'fn': set(), 'fp': set(), 'loc': set(), 'cls': set()}
    confused_pairs = Counter()

    total_images = len(results)

    for idx, result in enumerate(results):
        gt_boxes = result["gt_boxes"]
        pred_boxes = result["pred_boxes"]
        gt_classes = result["gt_classes"]
        pred_classes = result["pred_classes"]

        matches, matched_gt, matched_pred = match_predictions(gt_boxes, pred_boxes)

        if len(gt_boxes) > 0:
            if any(i not in matched_gt for i in range(len(gt_boxes))):
                error_images['fn'].add(idx)

        if len(pred_boxes) > 0:
            if any(j not in matched_pred for j in range(len(pred_boxes))):
                error_images['fp'].add(idx)

        for i, j, iou in matches:
            if iou < IOU_THRESHOLD:
                error_images['loc'].add(idx)

        for i, j, iou in matches:
            if iou >= IOU_THRESHOLD and gt_classes[i] != pred_classes[j]:
                error_images['cls'].add(idx)

                confused_pairs[
                    (class_names[gt_classes[i]], class_names[pred_classes[j]])
                ] += 1

    stats_df = pd.DataFrame({
        "Error Type": ["FN", "FP", "LOC", "CLS"],
        "Images with error": [
            len(error_images["fn"]),
            len(error_images["fp"]),
            len(error_images["loc"]),
            len(error_images["cls"]),
        ],
    })

    stats_df["% of images"] = (
        stats_df["Images with error"] / total_images * 100
    ).round(2)

    return stats_df, confused_pairs, error_images


def draw_boxes_resized(image, boxes, classes=None, highlight_idx=None, correct_idx=None, size=(640,640)):
    img = cv2.resize(image, size)
    h_scale = size[0] / image.shape[1]
    v_scale = size[1] / image.shape[0]

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        x1 = int(x1 * h_scale)
        x2 = int(x2 * h_scale)
        y1 = int(y1 * v_scale)
        y2 = int(y2 * v_scale)

        if highlight_idx and idx in highlight_idx:
            color = (255,0,0)
            thickness = 7
        elif correct_idx and idx in correct_idx:
            color = (0,255,0)
            thickness = 2
        else:
            color = (0,255,0)
            thickness = 2

        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)

        if classes is not None:
            label = class_names[classes[idx]]
            cv2.putText(img, label, (x1, max(y1-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def main():
    global model, class_names

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLOWorld(WEIGHTS_PATH)
    model.to(device)
    model.eval()

    with open(f"{DATASET_ROOT}/data.yaml", 'r') as f:
        dataset_config = yaml.safe_load(f)

    class_names = dataset_config['names']

    print(f"Number of classes: {len(class_names)}")

    results = run_full_analysis()

    strict_stats_df = compute_strict_statistics(results)
    print(strict_stats_df)

    stats_df, confused_pairs, error_images = analyze_results(results)

    total_errors = stats_df['Images with error'].sum()
    stats_df['% of errors'] = (
        stats_df['Images with error'] / total_errors * 100
    ).round(2)

    stats_df["Error Type"] = stats_df["Error Type"].replace({
        "FN": "Пропуски",
        "FP": "Лишние",
        "CLS": "Не тот класс",
        "LOC": "Локализация"
    })

    print(stats_df)

    confused_df = confused_pairs_to_df(confused_pairs)
    print(confused_df)


if __name__ == "__main__":
    main()

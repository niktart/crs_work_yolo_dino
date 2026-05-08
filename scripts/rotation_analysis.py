
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import cv2
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

WEIGHTS_PATH = "/content/drive/MyDrive/term_work/compare/weights/new_clear_food+new_flips3.pt"
DATASET_ROOT = "/content/data/combined_clean_bbox_уууу"
IMAGES_DIR = f"{DATASET_ROOT}/val/images"
LABELS_DIR = f"{DATASET_ROOT}/val/labels"

model = YOLO(WEIGHTS_PATH)

def load_label(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append((class_id, x_center, y_center, width, height))
    return boxes, None

def yolo_to_bbox(yolo_box, img_w, img_h):
    _, x_c, y_c, w, h = yolo_box
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def apply_augmentation(img, aug_id):
    h, w = img.shape[:2]

    if aug_id == 0:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    elif aug_id == 1:
        return cv2.flip(img, 1)

    elif aug_id == 2:
        return cv2.flip(img, 0)

    elif aug_id == 3:
        img = cv2.flip(img, 1)
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    elif aug_id == 4:
        scale = random.uniform(1.1, 1.3)
        resized = cv2.resize(img, None, fx=scale, fy=scale)
        return resized[:h, :w]

    elif aug_id == 5:
        scale = random.uniform(0.7, 0.9)
        resized = cv2.resize(img, None, fx=scale, fy=scale)
        canvas = np.zeros_like(img)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        return canvas

    elif aug_id == 6:
        angle = random.uniform(-20, 20)
        scale = random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        return cv2.warpAffine(img, M, (w, h))

    elif aug_id == 7:
        img = cv2.flip(img, 1)
        scale = random.uniform(0.8, 1.2)
        return cv2.resize(img, None, fx=scale, fy=scale)[:h, :w]

    elif aug_id == 8:
        angle = random.uniform(-45, 45)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    elif aug_id == 9:
        img = cv2.flip(img, 1)
        angle = random.uniform(-30, 30)
        scale = random.uniform(0.7, 1.3)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        return cv2.warpAffine(img, M, (w, h))

    return img

def sample_images(images_dir, n=50):
    files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return random.sample(files, min(n, len(files)))

def calculate_detection_rate_with_aug(
    model,
    images_dir,
    labels_dir,
    image_list,
    aug_id,
    conf_threshold=0.5,
    iou_threshold=0.5
):
    total_images = 0
    detected_images = 0

    for img_file in image_list:
        image_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(
            labels_dir,
            Path(img_file).with_suffix(".txt").name
        )

        if not os.path.exists(label_path):
            continue

        gt_boxes_yolo, _ = load_label(label_path)
        if len(gt_boxes_yolo) == 0:
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        img = apply_augmentation(img, aug_id)

        h, w = img.shape[:2]
        gt_boxes = [yolo_to_bbox(box, w, h) for box in gt_boxes_yolo]
        n_gt = len(gt_boxes)

        total_images += 1

        with torch.no_grad():
            preds = model.predict(
                img,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]

        pred_boxes = []
        if preds.boxes is not None:
            for box in preds.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2, y2])

        n_pred = len(pred_boxes)

        matched_gt = set()
        matched_pred = set()

        for i, gt in enumerate(gt_boxes):
            best_iou = 0
            best_j = -1

            for j, pred in enumerate(pred_boxes):
                if j in matched_pred:
                    continue

                iou = calculate_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold:
                matched_gt.add(i)
                matched_pred.add(best_j)

        is_detected = (
            len(matched_gt) == n_gt and
            n_pred == n_gt
        )

        if is_detected:
            detected_images += 1

    return detected_images / total_images if total_images > 0 else 0

def main():
    aug_descriptions = {
        0: "rotation ±10°",
        1: "horizontal flip",
        2: "vertical flip",
        3: "hflip + rotation ±15°",
        4: "zoom in (1.1–1.3)",
        5: "zoom out (0.7–0.9)",
        6: "rotation ±20° + zoom (0.8–1.2)",
        7: "hflip + zoom (0.8–1.2)",
        8: "rotation ±45°",
        9: "hflip + rotation ±30° + zoom (0.7–1.3)"
    }

    image_files = sample_images(IMAGES_DIR, n=50)
    results = []

    for aug_id in range(10):
        rate = calculate_detection_rate_with_aug(
            model, IMAGES_DIR, LABELS_DIR, image_files, aug_id
        )
        results.append({
            "augmentation": aug_descriptions[aug_id],
            "detection_rate": round(rate, 2)
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

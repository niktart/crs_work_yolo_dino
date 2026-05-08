import os
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from ultralytics import YOLOWorld
from tqdm import tqdm


def load_label(label_path: str):
    """Загрузка GT аннотаций в формате YOLO"""
    gt_boxes = []
    gt_classes = []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    gt_boxes.append([x_center, y_center, width, height])
                    gt_classes.append(class_id)
    return np.array(gt_boxes), np.array(gt_classes)


def yolo_to_bbox(yolo_box, img_width, img_height):
    """YOLO → absolute xyxy"""
    x_center, y_center, width, height = yolo_box
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    """IoU двух bbox"""
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


def calculate_detection_rate(
    model,
    images_dir: str,
    labels_dir: str,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.5
):
    """
    Определяет долю изображений, где все объекты детектированы на 100%.
    """
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    total_images = 0
    detected_images = 0
    results = []

    print(f"Found {len(image_files)} images")

    for img_file in tqdm(image_files, desc="Calculating detection rate"):
        image_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(
            labels_dir,
            Path(img_file).with_suffix(".txt").name
        )

        if not os.path.exists(label_path):
            continue

        # GT
        gt_boxes_yolo, _ = load_label(label_path)
        if len(gt_boxes_yolo) == 0:
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gt_boxes = [yolo_to_bbox(box, w, h) for box in gt_boxes_yolo]
        n_gt = len(gt_boxes)

        total_images += 1

        # Inference
        with torch.no_grad():
            preds = model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]

        pred_boxes = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for box in preds.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2, y2])
        n_pred = len(pred_boxes)

        # Matching
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

        all_gt_matched = len(matched_gt) == n_gt
        same_number_boxes = n_pred == n_gt
        is_detected = all_gt_matched and same_number_boxes

        if is_detected:
            detected_images += 1

        results.append({
            "image": img_file,
            "gt_objects": n_gt,
            "pred_objects": n_pred,
            "matched": len(matched_gt),
            "all_gt_matched": all_gt_matched,
            "same_number_boxes": same_number_boxes,
            "detected": is_detected,
        })

    rate = detected_images / total_images if total_images > 0 else 0

    print("\n" + "="*50)
    print(f"🎯 STRICT DETECTION RATE (100% only): {rate:.2%}")
    print("="*50)
    print(f"Total: {total_images}")
    print(f"Detected: {detected_images}")
    print(f"Missed: {total_images - detected_images}")

    df = pd.DataFrame(results)
    return rate, df

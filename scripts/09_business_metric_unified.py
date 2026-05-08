
import os
import cv2
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable, Optional

import torch
from ultralytics import YOLOWorld


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes in xyxy format."""
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


def load_gt_boxes(label_path: str) -> List[List[float]]:
    """Load ground truth boxes from YOLO format label file."""
    gt_boxes = []

    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                gt_boxes.append([class_id, x_center, y_center, width, height])

    return gt_boxes


def yolo_to_xyxy(box: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert YOLO format (cx, cy, w, h) to xyxy format."""
    _, x_center, y_center, width, height = box
    x1 = (x_center - width / 2) * img_w
    y1 = (y_center - height / 2) * img_h
    x2 = (x_center + width / 2) * img_w
    y2 = (y_center + height / 2) * img_h
    return [x1, y1, x2, y2]


class YOLOWorldInference:
    """Inference wrapper for YOLO-World model."""
    
    def __init__(self, weights_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading YOLO-World model from {weights_path}...")
        self.model = YOLOWorld(weights_path)
        self.model.to(device)
        self.model.eval()
    
    def infer(self, image_path: str, conf_threshold: float = 0.3, **kwargs) -> List[List[float]]:
        """Run inference and return boxes in xyxy format."""
        results = self.model.predict(
            image_path,
            conf=conf_threshold,
            verbose=False
        )[0]
        
        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                pred_boxes.append([x1, y1, x2, y2])
        
        return pred_boxes


class GroundingDINOInference:
    """Inference wrapper for Grounding DINO model."""
    
    def __init__(self, weights_path: str, config_path: str = None, device: str = None):
        from groundingdino.util.inference import load_model
        from groundingdino.config import get_config
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Grounding DINO model from {weights_path}...")
        
        # Load model using GroundingDINO's API
        self.model = load_model(
            model_config_path=config_path or "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path=weights_path,
            device=device
        )
        self.device = device
    
    def infer(self, image_path: str, conf_threshold: float = 0.3, 
              text_threshold: float = 0.25, text_prompt: str = None,
              **kwargs) -> List[List[float]]:
        """Run inference and return boxes in xyxy format."""
        from groundingdino.util.inference import load_image, predict
        import torch
        
        if text_prompt is None:
            raise ValueError("text_prompt is required for GroundingDINO inference")
        
        image_source, image = load_image(image_path)
        
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=conf_threshold,
                text_threshold=text_threshold,
            )
        
        if boxes is None or len(boxes) == 0:
            return []
        
        h, w = image_source.shape[:2]
        
        # GroundingDINO returns normalized cxcywh boxes
        boxes = boxes.cpu()
        boxes_xyxy = []
        
        for b in boxes:
            cx, cy, bw, bh = b.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes_xyxy.append([x1, y1, x2, y2])
        
        return boxes_xyxy


def calculate_detection_rate(
    inference_model: Any,
    images_dir: str,
    labels_dir: str,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    **infer_kwargs
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate strict detection rate (all GT objects matched with correct count).
    
    Args:
        inference_model: Model wrapper with infer() method
        images_dir: Directory with images
        labels_dir: Directory with YOLO format labels
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for matching
        **infer_kwargs: Additional arguments for inference
    
    Returns:
        detection_rate: Strict detection rate (0-1)
        results_df: DataFrame with detailed results
    """
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    
    total_images = 0
    detected_images = 0
    results = []
    
    print(f"📁 Found {len(image_files)} images")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, Path(img_file).with_suffix(".txt").name)
        
        # Skip if no label
        if not os.path.exists(label_path):
            continue
        
        # Load ground truth boxes
        gt_boxes_yolo = load_gt_boxes(label_path)
        if len(gt_boxes_yolo) == 0:
            continue
        
        # Load image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Warning: Cannot read image {image_path}")
            continue
        
        h, w = img.shape[:2]
        gt_boxes = [yolo_to_xyxy(box, w, h) for box in gt_boxes_yolo]
        n_gt = len(gt_boxes)
        
        total_images += 1
        
        # Run inference
        try:
            pred_boxes = inference_model.infer(
                image_path=image_path,
                conf_threshold=conf_threshold,
                **infer_kwargs
            )
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
            pred_boxes = []
        
        if pred_boxes is None:
            pred_boxes = []
        
        n_pred = len(pred_boxes)
        
        # Match GT with predictions
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
        
        # Strict criterion: all GT objects matched AND same number of boxes
        all_gt_matched = (len(matched_gt) == n_gt)
        same_number_boxes = (n_pred == n_gt)
        is_detected = all_gt_matched and same_number_boxes
        
        if is_detected:
            detected_images += 1
        
        results.append({
            "image": img_file,
            "gt_objects": n_gt,
            "pred_objects": n_pred,
            "matched_objects": len(matched_gt),
            "all_gt_matched": all_gt_matched,
            "same_number_boxes": same_number_boxes,
            "detected": is_detected
        })
    
    detection_rate = detected_images / total_images if total_images > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 STRICT DETECTION RATE (all objects detected, same count)")
    print("=" * 60)
    print(f"🎯 Detection Rate: {detection_rate:.2%} ({detected_images}/{total_images})")
    print(f"✅ Fully detected: {detected_images} images")
    print(f"❌ Missed/Partial: {total_images - detected_images} images")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    return detection_rate, results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluation script for YOLO-World and Grounding DINO")
    parser.add_argument("--model_type", type=str, required=True, choices=["yolo_world", "grounding_dino"],
                        help="Type of model to evaluate")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--dataset_root", type=str, 
                        default="/content/data/combined_clean_bbox",
                        help="Root directory of dataset")
    parser.add_argument("--output_dir", type=str,
                        default="/content/drive/MyDrive/term_work/outputs",
                        help="Output directory for results")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for matching")
    
    # Grounding DINO specific arguments
    parser.add_argument("--text_prompt", type=str, default="food . bread . vegetable . fruit . dish",
                        help="Text prompt for Grounding DINO (comma-separated)")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="Text threshold for Grounding DINO")
    parser.add_argument("--config_path", type=str, 
                        default="groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Config path for Grounding DINO")
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path(args.dataset_root) / "val" / "images"
    labels_dir = Path(args.dataset_root) / "val" / "labels"
    
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise RuntimeError(f"Dataset directories not found: {images_dir}, {labels_dir}")
    
    # Initialize model
    if args.model_type == "yolo_world":
        inference_model = YOLOWorldInference(args.weights_path)
        infer_kwargs = {}
    else:  # grounding_dino
        inference_model = GroundingDINOInference(args.weights_path, args.config_path)
        infer_kwargs = {
            "text_prompt": args.text_prompt,
            "text_threshold": args.text_threshold
        }
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print(f"🚀 Starting evaluation for {args.model_type}")
    print(f"📂 Images: {images_dir}")
    print(f"🏷️ Labels: {labels_dir}")
    print(f"🎯 Conf threshold: {args.conf_threshold}")
    print(f"🔗 IoU threshold: {args.iou_threshold}")
    print("=" * 60 + "\n")
    
    detection_rate, results_df = calculate_detection_rate(
        inference_model=inference_model,
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        **infer_kwargs
    )
    
    # Save results
    output_file = output_dir / f"business_metric_{args.model_type}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Save summary
    summary_file = output_dir / f"summary_{args.model_type}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Weights: {args.weights_path}\n")
        f.write(f"Confidence threshold: {args.conf_threshold}\n")
        f.write(f"IoU threshold: {args.iou_threshold}\n")
        f.write(f"Detection rate: {detection_rate:.2%}\n")
        f.write(f"Total images: {len(results_df)}\n")
        f.write(f"Fully detected: {results_df['detected'].sum()}\n")
        f.write(f"Missed: {(~results_df['detected']).sum()}\n")
    
    print(f"💾 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

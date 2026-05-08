

import time
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import os
import sys
import argparse

import onnxruntime as ort
from ultralytics import YOLOWorld
import nncf
from nncf import Dataset

try:
    from openvino import Core, convert_model, save_model
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("Warning: OpenVINO not available. Install with: pip install openvino")


DEFAULT_CONFIG = {
    "img_size": 640,
    "warmup": 10,
    "conf_threshold": 0.5,
    "iou_threshold": 0.5,
}

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOWorld inference benchmark")
    parser.add_argument("--weights", type=str,
                        default="/content/drive/MyDrive/term_work/compare/weights/clear_dataset/new_clear_food.pt",
                        help="Path to model weights")
    parser.add_argument("--data", type=str,
                        default="/content/data/combined_clean_bbox_уууу",
                        help="Path to dataset root")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to use")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Image size for inference")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to use (for quick testing)")
    parser.add_argument("--backends", nargs="+",
                        default=["pytorch_fp32", "pytorch_int8", "onnx_fp32", "onnx_fp16",
                                "onnx_int8", "openvino_fp32", "openvino_fp16", "openvino_int8"],
                        help="Backends to benchmark")
    return parser.parse_args()

# =========================
# UTILITY FUNCTIONS
# =========================
def preprocess_image(img_path, img_size):
    """Load and preprocess image for inference"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1) / 255.0  # HWC to CHW, normalize
    return img.astype(np.float32)[None]  # Add batch dimension

def load_gt_annotations(label_path):
    """Load ground truth annotations in YOLO format"""
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
    """Convert YOLO format to absolute xyxy coordinates"""
    x_center, y_center, width, height = yolo_box
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
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

def calculate_detection_rate(predict_func, images_list, labels_dir, conf_threshold=0.5, iou_threshold=0.5):
    """
    Calculate strict detection rate (100% detection required)
    """
    total_images = 0
    detected_images = 0
    results = []

    for img_path in tqdm(images_list, desc="Calculating detection rate"):
        img_file = Path(img_path).name
        label_path = os.path.join(labels_dir, Path(img_file).with_suffix(".txt").name)

        if not os.path.exists(label_path):
            continue

        gt_boxes_yolo, _ = load_gt_annotations(label_path)
        if len(gt_boxes_yolo) == 0:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gt_boxes = [yolo_to_bbox(box, w, h) for box in gt_boxes_yolo]
        n_gt = len(gt_boxes)

        total_images += 1

        # Get predictions
        try:
            pred_boxes = predict_func(img_path)
        except Exception as e:
            print(f"Error predicting {img_path}: {e}")
            pred_boxes = []

        n_pred = len(pred_boxes)

        # Greedy matching (1-1 assignment)
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

        # Check 100% detection condition
        all_gt_matched = (len(matched_gt) == n_gt)
        same_number_boxes = (n_pred == n_gt)
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
            "detected": is_detected
        })

    detection_rate = detected_images / total_images if total_images > 0 else 0
    results_df = pd.DataFrame(results)

    return detection_rate, results_df

def benchmark_inference(name, predict_func, images_list, warmup=10):
    """Benchmark inference speed"""
    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(min(warmup, len(images_list))):
        predict_func(images_list[_])

    # Benchmark
    start = time.perf_counter()
    for img_path in images_list:
        predict_func(img_path)
    total_time = time.perf_counter() - start

    fps = len(images_list) / total_time
    latency_ms = (total_time / len(images_list)) * 1000
    print(f"  {name:<20} | Time: {total_time:.2f}s | FPS: {fps:.2f} | Latency: {latency_ms:.0f}ms")

    return {
        "Backend": name,
        "Time(s)": total_time,
        "FPS": fps,
        "Latency(ms)": latency_ms
    }

class PyTorchBackend:
    def __init__(self, weights_path, img_size=640, conf=0.5, iou=0.5):
        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.model = YOLOWorld(weights_path)
        self.model.cpu()
        self.model.eval()

    def predict(self, img_path):
        with torch.no_grad():
            results = self.model.predict(
                img_path,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                imgsz=self.img_size
            )[0]

        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2, y2])
        return pred_boxes

class PyTorchINT8Backend:
    def __init__(self, weights_path, img_size=640, conf=0.5, iou=0.5):
        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.model = YOLOWorld(weights_path)
        self.model.cpu()
        self.model.eval()

        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
        except Exception as e:
            print(f"  Quantization warning: {e}")

    def predict(self, img_path):
        with torch.no_grad():
            results = self.model.predict(
                img_path,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                imgsz=self.img_size
            )[0]

        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2, y2])
        return pred_boxes

class ONNXBackend:
    def __init__(self, model_path, img_size=640, conf=0.5, iou=0.5, use_fp16=False):
        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.use_fp16 = use_fp16

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img_path):
        if self.use_fp16:
            img = preprocess_image(img_path, self.img_size).astype(np.float16)
        else:
            img = preprocess_image(img_path, self.img_size)

        outputs = self.session.run(None, {self.input_name: img})

        detections = outputs[0][0]  # (36, 8400)
        detections = detections.transpose(1, 0)  # (8400, 36)

        img_original = cv2.imread(img_path)
        h_img, w_img = img_original.shape[:2]

        pred_boxes = []

        for det in detections:
            x_center, y_center, width, height = det[:4]
            class_scores = det[4:]
            max_conf = np.max(class_scores)

            if max_conf >= self.conf:
                scale_x = w_img / self.img_size
                scale_y = h_img / self.img_size

                x1 = (x_center - width / 2) * scale_x
                y1 = (y_center - height / 2) * scale_y
                x2 = (x_center + width / 2) * scale_x
                y2 = (y_center + height / 2) * scale_y

                pred_boxes.append([float(x1), float(y1), float(x2), float(y2)])

        if len(pred_boxes) > 0:
            boxes = np.array(pred_boxes)
            scores = np.array([1.0] * len(pred_boxes))

            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.conf,
                self.iou
            )

            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                pred_boxes = boxes[indices].tolist()

        return pred_boxes

class OpenVINOBackend:
    def __init__(self, model_path, img_size=640, conf=0.5, iou=0.5):
        self.img_size = img_size
        self.conf = conf
        self.iou = iou

        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO not available")

        core = Core()
        self.model = core.read_model(str(model_path))
        self.compiled_model = core.compile_model(self.model, "CPU")

    def predict(self, img_path):
        img = preprocess_image(img_path, self.img_size)
        result = self.compiled_model([img])[0]

        if result.dtype == np.float16:
            result = result.astype(np.float32)

        detections = result[0]
        detections = detections.transpose(1, 0)

        img_original = cv2.imread(img_path)
        h_img, w_img = img_original.shape[:2]

        pred_boxes = []

        for det in detections:
            x_center, y_center, width, height = det[:4]
            class_scores = det[4:]
            max_conf = np.max(class_scores)

            if max_conf >= self.conf:
                scale_x = w_img / self.img_size
                scale_y = h_img / self.img_size

                x1 = (x_center - width / 2) * scale_x
                y1 = (y_center - height / 2) * scale_y
                x2 = (x_center + width / 2) * scale_x
                y2 = (y_center + height / 2) * scale_y

                pred_boxes.append([float(x1), float(y1), float(x2), float(y2)])

        if len(pred_boxes) > 0:
            boxes = np.array(pred_boxes)
            scores = np.array([1.0] * len(pred_boxes))

            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.conf,
                self.iou
            )

            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                pred_boxes = boxes[indices].tolist()

        return pred_boxes

def export_to_onnx(pt_model, onnx_path, img_size=640):
    """Export PyTorch model to ONNX"""
    if not onnx_path.exists():
        print(f"  Exporting to ONNX: {onnx_path}")
        pt_model.export(format="onnx", imgsz=img_size, opset=12, simplify=True, name=str(onnx_path))
    return onnx_path

def convert_to_openvino(onnx_path, ov_path, img_size=640):
    """Convert ONNX to OpenVINO"""
    if not ov_path.exists() and OPENVINO_AVAILABLE:
        print(f"  Converting to OpenVINO: {ov_path}")
        ov_model = convert_model(str(onnx_path))
        save_model(ov_model, str(ov_path))
    return ov_path


def main():
    args = parse_args()

    # Setup paths
    weights_path = Path(args.weights)
    dataset_root = Path(args.data)
    images_dir = dataset_root / args.split / "images"
    labels_dir = dataset_root / args.split / "labels"
    model_name = weights_path.stem
    model_dir = weights_path.parent

    # Create model directories
    onnx_fp32_path = model_dir / f"{model_name}.onnx"
    onnx_fp16_path = model_dir / f"{model_name}_fp16.onnx"
    onnx_int8_path = model_dir / f"{model_name}_int8.onnx"
    ov_fp32_path = model_dir / f"{model_name}_openvino" / f"{model_name}.xml"
    ov_fp16_path = model_dir / f"{model_name}_openvino" / f"{model_name}_fp16.xml"
    ov_int8_path = model_dir / f"{model_name}_openvino" / f"{model_name}_int8.xml"

    # Load images
    images = sorted(glob.glob(os.path.join(images_dir, "*")))
    images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if args.max_images:
        images = images[:args.max_images]

    print(f"\n{'='*60}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Weights: {weights_path}")
    print(f"  Dataset: {dataset_root}")
    print(f"  Split: {args.split}")
    print(f"  Images: {len(images)}")
    print(f"  Image size: {args.img_size}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Warmup: {args.warmup}")

    # Load base model for exports
    print(f"\n{'='*60}")
    print(f"LOADING BASE MODEL")
    print(f"{'='*60}")
    base_model = YOLOWorld(str(weights_path))
    base_model.cpu()
    base_model.eval()

    # Export models if needed
    if any(backend in args.backends for backend in ["onnx_fp32", "onnx_fp16", "onnx_int8",
                                                      "openvino_fp32", "openvino_fp16", "openvino_int8"]):
        export_to_onnx(base_model, onnx_fp32_path, args.img_size)

    # Results storage
    all_results = []

    print(f"\n{'='*60}")
    print(f"RUNNING BENCHMARKS")
    print(f"{'='*60}")

    # PyTorch FP32
    if "pytorch_fp32" in args.backends:
        print(f"\n📌 PyTorch FP32")
        backend = PyTorchBackend(str(weights_path), args.img_size, args.conf, args.iou)
        bench_result = benchmark_inference("PyTorch FP32", backend.predict, images, args.warmup)
        rate, _ = calculate_detection_rate(backend.predict, images, str(labels_dir), args.conf, args.iou)
        print(f"  ✅ PyTorch FP32 Detection Rate: {rate:.2%}")
        all_results.append({**bench_result, "Detection Rate": rate})

    # PyTorch INT8
    if "pytorch_int8" in args.backends:
        print(f"\n📌 PyTorch INT8")
        backend = PyTorchINT8Backend(str(weights_path), args.img_size, args.conf, args.iou)
        bench_result = benchmark_inference("PyTorch INT8", backend.predict, images, args.warmup)
        rate, _ = calculate_detection_rate(backend.predict, images, str(labels_dir), args.conf, args.iou)
        print(f"  ✅ PyTorch INT8 Detection Rate: {rate:.2%}")
        all_results.append({**bench_result, "Detection Rate": rate})

    # ONNX FP32
    if "onnx_fp32" in args.backends:
        print(f"\n📌 ONNX FP32")
        backend = ONNXBackend(onnx_fp32_path, args.img_size, args.conf, args.iou, use_fp16=False)
        bench_result = benchmark_inference("ONNX FP32", backend.predict, images, args.warmup)
        rate, _ = calculate_detection_rate(backend.predict, images, str(labels_dir), args.conf, args.iou)
        print(f"  ✅ ONNX FP32 Detection Rate: {rate:.2%}")
        all_results.append({**bench_result, "Detection Rate": rate})

    # ONNX FP16
    if "onnx_fp16" in args.backends and onnx_fp16_path.exists():
        print(f"\n📌 ONNX FP16")
        backend = ONNXBackend(onnx_fp16_path, args.img_size, args.conf, args.iou, use_fp16=True)
        bench_result = benchmark_inference("ONNX FP16", backend.predict, images, args.warmup)
        rate, _ = calculate_detection_rate(backend.predict, images, str(labels_dir), args.conf, args.iou)
        print(f"  ✅ ONNX FP16 Detection Rate: {rate:.2%}")
        all_results.append({**bench_result, "Detection Rate": rate})

    # OpenVINO FP32
    if "openvino_fp32" in args.backends and OPENVINO_AVAILABLE:
        convert_to_openvino(onnx_fp32_path, ov_fp32_path, args.img_size)
        print(f"\n📌 OpenVINO FP32")
        backend = OpenVINOBackend(ov_fp32_path, args.img_size, args.conf, args.iou)
        bench_result = benchmark_inference("OpenVINO FP32", backend.predict, images, args.warmup)
        rate, _ = calculate_detection_rate(backend.predict, images, str(labels_dir), args.conf, args.iou)
        print(f"  ✅ OpenVINO FP32 Detection Rate: {rate:.2%}")
        all_results.append({**bench_result, "Detection Rate": rate})

    # OpenVINO FP16
    if "openvino_fp16" in args.backends and OPENVINO_AVAILABLE and ov_fp16_path.exists():
        print(f"\n📌 OpenVINO FP16")
        backend = OpenVINOBackend(ov_fp16_path, args.img_size, args.conf, args.iou)
        bench_result = benchmark_inference("OpenVINO FP16", backend.predict, images, args.warmup)
        rate, _ = calculate_detection_rate(backend.predict, images, str(labels_dir), args.conf, args.iou)
        print(f"  ✅ OpenVINO FP16 Detection Rate: {rate:.2%}")
        all_results.append({**bench_result, "Detection Rate": rate})


    print(f"{'Backend':<20} | {'Detection Rate':<15} | {'FPS':<10} | {'Latency(ms)':<12}")
    print("-" * 65)

    # Calculate relative performance
    baseline_fps = None
    for result in all_results:
        if result["Backend"] == "PyTorch FP32":
            baseline_fps = result["FPS"]
            break

    for result in all_results:
        backend = result["Backend"]
        rate = result["Detection Rate"]
        fps = result["FPS"]
        latency = result["Latency(ms)"]
        rel_perf = fps / baseline_fps if baseline_fps else 1.0

        print(f"{backend:<20} | {rate:>14.2%} | {fps:>9.2f} | {latency:>11.0f}ms")

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("benchmark_results.csv", index=False)
    print(f"\n✅ Results saved to benchmark_results.csv")

    return all_results

if __name__ == "__main__":
    main()

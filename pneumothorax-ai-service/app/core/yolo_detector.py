import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
from app.core.config import YOLO_B_WEIGHTS_PATH, YOLO_A_WEIGHTS_PATH, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, VINDR_CLASSES
from app.core.clinical_metrics import determine_lung_zone

class YOLODetector:
    def __init__(self, weights_path: Path = YOLO_B_WEIGHTS_PATH, device: str = None):
        self.weights_path = weights_path
        self.device = device
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.weights_path):
            print(f"[YOLODetector] Warning: Weights file {self.weights_path} not found.")
            return

        print(f"[YOLODetector] Loading YOLO weights from {self.weights_path}...")
        self.model = YOLO(str(self.weights_path))
        print("[YOLODetector] YOLO model loaded successfully.")

    def predict(
        self,
        img_bgr: np.ndarray,
        conf_thr: float = YOLO_CONF_THRESHOLD,
        iou_thr: float = YOLO_IOU_THRESHOLD,
        imgsz: int = 1280
    ) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
        """
        Runs YOLO multi-lesion detection.
        Returns:
            detections: List of dicts with box, confidence, class_id, class_name, lung_zone.
            heart_box: [x1, y1, x2, y2] for CTR computation (or None if no cardiomegaly detected).
        """
        orig_h, orig_w = img_bgr.shape[:2]
        if self.model is None:
            return [], None

        results = self.model.predict(
            source=img_bgr,
            conf=conf_thr,
            iou=iou_thr,
            imgsz=imgsz,
            verbose=False,
            device=self.device
        )

        raw_detections = []
        cardiomegaly_boxes = []

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxy, confs, clss):
                    class_name = VINDR_CLASSES.get(cls_id, res.names.get(cls_id, f"Class {cls_id}"))
                    # Normalize underscore
                    class_name = class_name.replace("_", " ")

                    x1, y1, x2, y2 = [float(v) for v in box]
                    det = {
                        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "confidence": round(float(conf), 3),
                        "class_id": int(cls_id),
                        "class_name": class_name,
                        "lung_zone": determine_lung_zone([x1, y1, x2, y2], orig_h)
                    }

                    if "Cardiomegaly" in class_name:
                        cardiomegaly_boxes.append(det)
                    else:
                        raw_detections.append(det)

        # Clinical Single Cardiac Envelope Post-Processing:
        # Enforce exactly one consolidated cardiac envelope box if cardiomegaly is detected
        heart_box = None
        if cardiomegaly_boxes:
            min_x1 = min(b["box"][0] for b in cardiomegaly_boxes)
            min_y1 = min(b["box"][1] for b in cardiomegaly_boxes)
            max_x2 = max(b["box"][2] for b in cardiomegaly_boxes)
            max_y2 = max(b["box"][3] for b in cardiomegaly_boxes)
            max_conf = max(b["confidence"] for b in cardiomegaly_boxes)

            unified_heart = {
                "box": [min_x1, min_y1, max_x2, max_y2],
                "confidence": max_conf,
                "class_id": 0,
                "class_name": "Cardiomegaly",
                "lung_zone": "Cardiac / Mediastinal Compartment"
            }
            raw_detections.insert(0, unified_heart)
            heart_box = [min_x1, min_y1, max_x2, max_y2]

        return raw_detections, heart_box

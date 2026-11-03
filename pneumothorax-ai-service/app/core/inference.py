import os
import cv2
import uuid
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

from app.core.config import RESULTS_DIR
from app.core.unet_segmenter import UNetSegmenter
from app.core.yolo_detector import YOLODetector
from app.core.clinical_metrics import calculate_ctr, calculate_pleural_collapse, determine_triage_alert
from app.core.visualizer import render_medical_visualizations

class MedicalInferencePipeline:
    def __init__(self):
        print("[MedicalInferencePipeline] Initializing models...")
        self.unet = UNetSegmenter()
        self.yolo = YOLODetector()
        print("[MedicalInferencePipeline] Initialization complete.")

    def run(self, img_input: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Executes end-to-end multi-modal CXR diagnostic inference.
        """
        start_time = time.time()

        # 1. Image Ingestion
        if isinstance(img_input, (str, Path)):
            img_bgr = cv2.imread(str(img_input))
            source_name = Path(img_input).name
        elif isinstance(img_input, bytes):
            nparr = np.frombuffer(img_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            source_name = "uploaded_cxr.png"
        elif isinstance(img_input, np.ndarray):
            img_bgr = img_input
            source_name = "direct_array.png"
        else:
            raise ValueError("Unsupported input format for inference.")

        if img_bgr is None:
            raise ValueError("Could not decode image.")

        h, w = img_bgr.shape[:2]
        exam_id = f"EXAM-{uuid.uuid4().hex[:8].upper()}"

        # 2. Model Inference
        # Phase I: U-Net++ Pneumothorax Segmentation
        t_unet_start = time.time()
        ptx_mask, ptx_raw_stats = self.unet.predict(img_bgr, use_tta=True)
        t_unet = round((time.time() - t_unet_start) * 1000, 1)

        # Phase II: YOLO Multi-Lesion Detection
        t_yolo_start = time.time()
        detections, heart_box = self.yolo.predict(img_bgr, conf_thr=0.25, iou_thr=0.45, imgsz=1280)
        t_yolo = round((time.time() - t_yolo_start) * 1000, 1)

        # 3. Clinical Biomarker Quantification
        ctr_stats = calculate_ctr(heart_box, img_w=w)
        collapse_stats = calculate_pleural_collapse(ptx_mask)
        ptx_stats = {
            "detected": ptx_raw_stats["detected"],
            "area_px": ptx_raw_stats["area_px"],
            "confidence_mean": ptx_raw_stats["confidence_mean"],
            **collapse_stats
        }

        triage = determine_triage_alert(ptx_stats, detections, ctr_stats)

        # 4. Rendering Visual Overlays
        composite, seg_only, det_only = render_medical_visualizations(
            img_bgr, detections, ptx_mask, mask_opacity=0.45
        )

        # Save artifacts to results/
        composite_filename = f"{exam_id}_composite.png"
        seg_filename = f"{exam_id}_seg.png"
        det_filename = f"{exam_id}_det.png"
        raw_filename = f"{exam_id}_raw.png"

        cv2.imwrite(str(RESULTS_DIR / composite_filename), composite)
        cv2.imwrite(str(RESULTS_DIR / seg_filename), seg_only)
        cv2.imwrite(str(RESULTS_DIR / det_filename), det_only)
        cv2.imwrite(str(RESULTS_DIR / raw_filename), img_bgr)

        total_time_ms = round((time.time() - start_time) * 1000, 1)

        # 5. Synthesize Structured Clinical Report
        report = {
            "exam_id": exam_id,
            "source_image": source_name,
            "dimensions": f"{w} x {h} px",
            "latency": {
                "total_ms": total_time_ms,
                "unet_ms": t_unet,
                "yolo_ms": t_yolo
            },
            "triage_alert": triage,
            "biomarkers": {
                "cardiothoracic_ratio": ctr_stats,
                "pleural_collapse": ptx_stats
            },
            "findings": {
                "pneumothorax_segmentation": ptx_stats,
                "localized_lesions": detections,
                "total_lesions_count": len(detections) + (1 if ptx_stats["detected"] else 0)
            },
            "artifacts": {
                "raw_url": f"/results/{raw_filename}",
                "composite_url": f"/results/{composite_filename}",
                "seg_url": f"/results/{seg_filename}",
                "det_url": f"/results/{det_filename}"
            }
        }

        return report

# Global pipeline instance
pipeline = MedicalInferencePipeline()

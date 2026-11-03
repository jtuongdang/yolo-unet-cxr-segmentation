import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from app.core.config import CLASS_COLORS

def render_medical_visualizations(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    ptx_mask: np.ndarray,
    mask_opacity: float = 0.45
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renders 3 medical visual layers:
    1. Composite (Boxes + Mask)
    2. Segmentation Mask Only
    3. Detection Bounding Boxes Only
    """
    h, w = img_bgr.shape[:2]

    # --- 1. SEGMENTATION MASK LAYER ---
    seg_overlay = img_bgr.copy()
    if ptx_mask is not None and np.sum(ptx_mask > 0) > 0:
        colored_mask = np.zeros_like(img_bgr)
        # BGR: Spring Green (0, 255, 127)
        mask_color_bgr = (127, 255, 0)
        colored_mask[ptx_mask > 0] = mask_color_bgr
        
        # Blend
        cv2.addWeighted(colored_mask, mask_opacity, seg_overlay, 1.0, 0, seg_overlay)
        
        # Draw contour border
        contours, _ = cv2.findContours(ptx_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_overlay, contours, -1, (0, 255, 127), 2)

    # --- 2. DETECTION BOUNDING BOX LAYER ---
    det_overlay = img_bgr.copy()
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        class_name = det["class_name"]
        conf = det["confidence"]

        # Color mapping (RGB to BGR)
        rgb_color = CLASS_COLORS.get(class_name, (0, 255, 255))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        # Draw box
        cv2.rectangle(det_overlay, (x1, y1), (x2, y2), bgr_color, 2)

        # Label badge
        label = f"{class_name}: {conf:.2f}"
        font_scale = 0.55
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        badge_y1 = max(0, y1 - label_h - 8)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = min(w, x1 + label_w + 10)

        # Draw filled badge background
        cv2.rectangle(det_overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), bgr_color, -1)
        # White or black text
        text_color = (0, 0, 0) if (rgb_color[0]*0.299 + rgb_color[1]*0.587 + rgb_color[2]*0.114) > 180 else (255, 255, 255)
        cv2.putText(det_overlay, label, (badge_x1 + 5, badge_y2 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # --- 3. COMPOSITE LAYER (Both mask and boxes) ---
    composite = seg_overlay.copy()
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        class_name = det["class_name"]
        conf = det["confidence"]

        rgb_color = CLASS_COLORS.get(class_name, (0, 255, 255))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        cv2.rectangle(composite, (x1, y1), (x2, y2), bgr_color, 2)

        label = f"{class_name}: {conf:.2f}"
        font_scale = 0.55
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        badge_y1 = max(0, y1 - label_h - 8)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = min(w, x1 + label_w + 10)

        cv2.rectangle(composite, (badge_x1, badge_y1), (badge_x2, badge_y2), bgr_color, -1)
        text_color = (0, 0, 0) if (rgb_color[0]*0.299 + rgb_color[1]*0.587 + rgb_color[2]*0.114) > 180 else (255, 255, 255)
        cv2.putText(composite, label, (badge_x1 + 5, badge_y2 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return composite, seg_overlay, det_overlay

import numpy as np
from typing import Dict, List, Any, Optional

def calculate_ctr(heart_box: Optional[List[float]], img_w: int) -> Dict[str, Any]:
    """
    Computes purely objective Cardiothoracic Ratio (CTR = W_heart / W_thorax).
    W_thorax is approximated from standard PA chest radiograph rib cage boundaries (~85% of image width).
    No qualitative severity labels assigned - pure mathematical measurement for physician interpretation.
    """
    if heart_box is None:
        return {
            "ctr_value": None,
            "heart_width_px": 0,
            "thorax_width_px": int(img_w * 0.85)
        }
        
    x1, y1, x2, y2 = heart_box
    heart_w = max(0, x2 - x1)
    estimated_thorax_w = img_w * 0.85
    
    ctr = min(1.0, max(0.0, heart_w / estimated_thorax_w))
    ctr_rounded = round(float(ctr), 3)
    
    return {
        "ctr_value": ctr_rounded,
        "heart_width_px": round(heart_w, 1),
        "thorax_width_px": round(estimated_thorax_w, 1)
    }

def calculate_pleural_collapse(ptx_mask: np.ndarray) -> Dict[str, Any]:
    """
    Calculates the purely objective pleural collapse fraction Phi = (ptx_area / estimated_hemithorax_lung_area) * 100%
    In PA chest radiographs, the ipsilateral hemithorax lung area is ~20% of total image pixels.
    No qualitative severity judgments assigned - pure volumetric measurements.
    """
    h, w = ptx_mask.shape[:2]
    total_pixels = h * w
    ptx_pixels = int(np.sum(ptx_mask > 0))
    estimated_lung_pixels = total_pixels * 0.20
    
    if ptx_pixels == 0:
        return {
            "collapse_percentage": 0.0,
            "ptx_pixels": 0,
            "estimated_hemithorax_px": int(estimated_lung_pixels)
        }
        
    fraction = min(100.0, (ptx_pixels / estimated_lung_pixels) * 100.0)
    fraction_rounded = round(float(fraction), 1)
    
    return {
        "collapse_percentage": fraction_rounded,
        "ptx_pixels": ptx_pixels,
        "estimated_hemithorax_px": int(estimated_lung_pixels)
    }

def determine_lung_zone(box: List[float], img_h: int) -> str:
    """Maps bounding box center to Apical, Mid, or Basal lung zone."""
    x1, y1, x2, y2 = box
    center_y = (y1 + y2) / 2.0
    relative_y = center_y / img_h
    
    if relative_y < 0.38:
        return "Apical / Upper Zone"
    elif relative_y < 0.68:
        return "Mid Lung Zone"
    else:
        return "Basal / Lower Zone"

def determine_triage_alert(ptx_stats: Dict[str, Any], yolo_detections: List[Dict[str, Any]], ctr_stats: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns unassigned baseline status. 
    Automated severity warnings disabled per physician specification. 
    The physician manually assigns triage priority on the PACS workstation.
    """
    return {
        "level": "UNASSIGNED",
        "badge_class": "badge-neutral",
        "summary": "Số liệu đo đạc khách quan đã sẵn sàng. Chờ bác sĩ đánh giá và tự gán mức cảnh báo."
    }

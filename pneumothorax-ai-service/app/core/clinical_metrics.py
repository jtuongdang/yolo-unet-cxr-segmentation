import numpy as np
from typing import Dict, List, Any, Optional

def calculate_ctr(heart_box: Optional[List[float]], img_w: int) -> Dict[str, Any]:
    """
    Computes Cardiothoracic Ratio (CTR = W_heart / W_thorax).
    W_thorax is approximated from standard PA chest radiograph rib cage boundaries (~85% of image width).
    """
    if heart_box is None:
        return {
            "ctr_value": None,
            "status": "Not Assessed / Normal",
            "is_enlarged": False
        }
        
    x1, y1, x2, y2 = heart_box
    heart_w = max(0, x2 - x1)
    estimated_thorax_w = img_w * 0.85
    
    ctr = min(1.0, max(0.0, heart_w / estimated_thorax_w))
    ctr_rounded = round(float(ctr), 3)
    
    if ctr >= 0.55:
        status = "Severe Cardiomegaly"
        is_enlarged = True
    elif ctr >= 0.50:
        status = "Borderline / Mild Cardiomegaly"
        is_enlarged = True
    else:
        status = "Normal Heart Size"
        is_enlarged = False
        
    return {
        "ctr_value": ctr_rounded,
        "status": status,
        "is_enlarged": is_enlarged
    }

def calculate_pleural_collapse(ptx_mask: np.ndarray) -> Dict[str, Any]:
    """
    Calculates the pleural collapse fraction Phi = (ptx_area / estimated_hemithorax_lung_area) * 100%
    In PA chest radiographs, the ipsilateral hemithorax lung area is ~15% of total image pixels.
    """
    h, w = ptx_mask.shape[:2]
    total_pixels = h * w
    ptx_pixels = int(np.sum(ptx_mask > 0))
    
    if ptx_pixels == 0:
        return {
            "collapse_percentage": 0.0,
            "ptx_pixels": 0,
            "severity": "None",
            "extent_description": "Normal pleural surface"
        }
        
    estimated_lung_pixels = total_pixels * 0.20
    fraction = min(100.0, (ptx_pixels / estimated_lung_pixels) * 100.0)
    fraction_rounded = round(float(fraction), 1)
    
    if fraction_rounded >= 20.0:
        severity = "Large Pneumothorax (>= 20%)"
        extent_description = "Extensive hemithoracic volume"
    elif fraction_rounded >= 10.0:
        severity = "Moderate Pneumothorax (10-20%)"
        extent_description = "Moderate hemithoracic volume"
    else:
        severity = "Small / Apical Pneumothorax (< 10%)"
        extent_description = "Minimal apical/lateral rim"
        
    return {
        "collapse_percentage": fraction_rounded,
        "ptx_pixels": ptx_pixels,
        "severity": severity,
        "extent_description": extent_description
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
    Synthesizes overall clinical triage alert level:
    - EMERGENCY (Red): Significant pneumothorax collapse >= 10%
    - PRIORITY (Yellow): Small pneumothorax, cardiomegaly, or parenchymal opacities
    - ROUTINE (Green): Clean radiograph
    """
    has_ptx = ptx_stats.get("detected", False)
    collapse_pct = ptx_stats.get("collapse_percentage", 0.0)
    has_cardiomegaly = ctr_stats.get("is_enlarged", False)
    has_lesions = len(yolo_detections) > 0

    if has_ptx and collapse_pct >= 10.0:
        return {
            "level": "EMERGENCY",
            "badge_class": "badge-danger",
            "summary": "Critical Finding: Significant Visceral Pleural Collapse Delineated."
        }
    elif has_ptx or has_cardiomegaly or has_lesions:
        findings_summary = []
        if has_ptx:
            findings_summary.append("Pneumothorax")
        if has_cardiomegaly:
            findings_summary.append("Cardiomegaly")
        for det in yolo_detections:
            cls_name = det.get("class_name")
            if cls_name not in findings_summary:
                findings_summary.append(cls_name)
                
        return {
            "level": "PRIORITY",
            "badge_class": "badge-warning",
            "summary": f"Focal Radiographic Abnormalities: {', '.join(findings_summary)}."
        }
    else:
        return {
            "level": "ROUTINE",
            "badge_class": "badge-success",
            "summary": "No Significant Focal Lesions or Pleural Detachment Detected."
        }

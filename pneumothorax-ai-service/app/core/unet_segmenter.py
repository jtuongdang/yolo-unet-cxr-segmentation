import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from pathlib import Path
from app.core.config import UNET_WEIGHTS_PATH, UNET_THRESHOLD, UNET_MIN_PIXELS

class UNetSegmenter:
    def __init__(self, weights_path: Path = UNET_WEIGHTS_PATH, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            self.device = torch.device(device)
            
        self.weights_path = weights_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.weights_path):
            print(f"[UNetSegmenter] Warning: Weights file {self.weights_path} not found.")
            return

        print(f"[UNetSegmenter] Loading U-Net++ weights from {self.weights_path} on {self.device}...")
        self.model = smp.UnetPlusPlus(
            encoder_name="se_resnext50_32x4d",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder_attention_type="scse"
        )
        ckpt = torch.load(self.weights_path, map_location=self.device)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else (ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("[UNetSegmenter] U-Net++ model loaded successfully.")

    def preprocess(self, img_bgr: np.ndarray, target_size: int = 512):
        """Apply CLAHE and standard normalization matching training."""
        if len(img_bgr.shape) == 2:
            gray = img_bgr
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3-channel RGB
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        tensor = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))  # (3, H, W)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, img_bgr: np.ndarray, use_tta: bool = True):
        """
        Runs U-Net++ inference with TTA, thresholding, and morphological filtering.
        Returns:
            mask_orig: binary mask (uint8, 0 or 255) at original image resolution
            stats: dict containing metrics (ptx_detected, ptx_area_px, ptx_percentage)
        """
        orig_h, orig_w = img_bgr.shape[:2]
        if self.model is None:
            return np.zeros((orig_h, orig_w), dtype=np.uint8), {
                "detected": False,
                "area_px": 0,
                "confidence_mean": 0.0
            }

        tensor = self.preprocess(img_bgr, target_size=512)
        
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output)
            
            if use_tta:
                # Horizontal flip TTA
                tensor_flip = torch.flip(tensor, dims=[3])
                output_flip = self.model(tensor_flip)
                prob_flip = torch.flip(torch.sigmoid(output_flip), dims=[3])
                prob = 0.5 * (prob + prob_flip)
                
            prob_map = prob.squeeze().cpu().numpy()

        # Thresholding at UNET_THRESHOLD
        binary_mask_512 = (prob_map >= UNET_THRESHOLD).astype(np.uint8)

        # Connected component filtering: remove small spurious clusters (< UNET_MIN_PIXELS)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_512, connectivity=8)
        filtered_mask_512 = np.zeros_like(binary_mask_512)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= UNET_MIN_PIXELS:
                filtered_mask_512[labels == i] = 1

        # Resize back to original image dimensions
        mask_orig = cv2.resize(filtered_mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) * 255

        # Quantitative summary
        positive_pixels = int(np.sum(mask_orig > 0))
        has_ptx = positive_pixels > 0
        mean_conf = float(np.mean(prob_map[filtered_mask_512 > 0])) if has_ptx else 0.0

        return mask_orig, {
            "detected": has_ptx,
            "area_px": positive_pixels,
            "confidence_mean": round(mean_conf, 4)
        }

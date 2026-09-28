from typing import Tuple
import numpy as np
import torch
from PIL import Image

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[SAMSegmentation] segment_anything not installed, SAM disabled.")

class SAMSegmentation:
    def __init__(self, checkpoint_path="checkpoints/sam_vit_b_01ec64.pth", model_type="vit_b"):
        self.mask_generator = None
        device = "cuda" if SAM_AVAILABLE and torch.cuda.is_available() else "cpu"
        if SAM_AVAILABLE:
            try:
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                sam.to(device=device)
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=sam, points_per_side=32, pred_iou_thresh=0.9, stability_score_thresh=0.95,
                )
                print(f"[SAMSegmentation] SAM {model_type} loaded on {device}")
            except Exception as e:
                print(f"[SAMSegmentation] Failed to load SAM: {e}")

    def segment_foreground(self, image: Image.Image) -> Image.Image:
        if self.mask_generator is None:
            return image
        max_side = 1024
        orig_size = image.size
        if max(orig_size) > max_side:
            image = image.copy()
            image.thumbnail((max_side, max_side), Image.BICUBIC)
            print(f"[SAM] Resized image from {orig_size} to {image.size} for segmentation")
        np_image = np.array(image.convert("RGB"))
        masks = self.mask_generator.generate(np_image)
        if not masks:
            return Image.fromarray(np_image.astype(np.uint8))
        largest = max(masks, key=lambda x: x["area"])
        mask = largest["segmentation"].astype(np.uint8)
        foreground = np_image * mask[:, :, None]
        return Image.fromarray(foreground.astype(np.uint8))

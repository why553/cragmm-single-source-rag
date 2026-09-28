from typing import Tuple
import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[GradCAMAttention] transformers not installed, attention disabled.")

class GradCAMAttention:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if GRADCAM_AVAILABLE:
            try:
                self.model = CLIPModel.from_pretrained(model_name, attn_implementation="eager").to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model.eval()
                print(f"[GradCAMAttention] Loaded {model_name} on {self.device}")
            except Exception as e:
                print(f"[GradCAMAttention] Failed to load CLIP: {e}")

    def _get_attention_map(self, image: Image.Image, query: str) -> np.ndarray:
        if self.model is None:
            return np.zeros((14, 14))
        inputs = self.processor(text=[query], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_attentions=True, return_dict=True)
        attentions = vision_outputs.attentions
        if attentions is None or len(attentions) == 0:
            return np.zeros((14, 14))
        last_attn = attentions[-1]
        cls_attn = last_attn[0, :, 0, 1:].mean(dim=0)
        n_patches = cls_attn.shape[0]
        grid_size = int(np.sqrt(n_patches))
        attn_map = cls_attn.cpu().numpy().reshape(grid_size, grid_size)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map

    def find_attention_bbox(self, attn_map: np.ndarray, image_size: Tuple[int, int], threshold: float = 0.6) -> Tuple[int, int, int, int]:
        h, w = attn_map.shape
        img_w, img_h = image_size
        mask = attn_map > threshold
        if not mask.any():
            return (0, 0, img_w, img_h)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        x = int(x_min / w * img_w)
        y = int(y_min / h * img_h)
        box_w = int((x_max - x_min + 1) / w * img_w)
        box_h = int((y_max - y_min + 1) / h * img_h)
        box_w = max(box_w, 64)
        box_h = max(box_h, 64)
        return (x, y, box_w, box_h)

    def crop_attention_region(self, image: Image.Image, query: str, threshold: float = 0.6) -> Image.Image:
        if self.model is None:
            return image
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.BICUBIC)
            print(f"[GradCAM] Image resized to {image.size} for speed")
        attn_map = self._get_attention_map(image, query)
        x, y, w, h = self.find_attention_bbox(attn_map, image.size, threshold)
        cropped = image.crop((x, y, x + w, y + h))
        return cropped.resize(image.size, Image.BICUBIC)

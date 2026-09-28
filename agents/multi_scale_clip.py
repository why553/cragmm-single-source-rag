from typing import List, Tuple
import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[MultiScaleCLIP] transformers not installed, multi-scale encoding disabled.")

class MultiScaleCLIPEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if CLIP_AVAILABLE:
            try:
                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model.eval()
                print(f"[MultiScaleCLIP] Loaded {model_name} on {self.device}")
            except Exception as e:
                print(f"[MultiScaleCLIP] Failed to load CLIP: {e}")

    def _encode_single(self, image: Image.Image) -> np.ndarray:
        if self.model is None:
            return np.zeros(512)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            else:
                return np.zeros(512)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    def encode_multi_scale(self, image: Image.Image, num_patches: int = 4) -> np.ndarray:
        if self.model is None:
            return np.zeros(512 * 3)
        feat_global = self._encode_single(image)
        patches = self._crop_patches(image, grid=(2, 2))
        patch_feats = [self._encode_single(p) for p in patches]
        feat_local = np.mean(patch_feats, axis=0)
        feat_local = feat_local / (np.linalg.norm(feat_local) + 1e-8)
        image_up = image.resize((image.width * 2, image.height * 2), Image.BICUBIC)
        feat_up = self._encode_single(image_up)
        return np.concatenate([feat_global, feat_local, feat_up])

    def _crop_patches(self, image: Image.Image, grid: Tuple[int, int] = (2, 2)) -> List[Image.Image]:
        w, h = image.size
        rows, cols = grid
        patch_w, patch_h = w // cols, h // rows
        patches = []
        for r in range(rows):
            for c in range(cols):
                left = c * patch_w
                upper = r * patch_h
                right = left + patch_w if c < cols - 1 else w
                lower = upper + patch_h if r < rows - 1 else h
                patches.append(image.crop((left, upper, right, lower)))
        return patches

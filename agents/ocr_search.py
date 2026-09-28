from typing import List, Dict
from thefuzz import fuzz

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("[OCRSearch] PaddleOCR not installed, OCR disabled.")

class OCRSearch:
    def __init__(self, lang="en"):
        self.ocr = None
        if PADDLE_AVAILABLE:
            try:
                self.ocr = PaddleOCR(lang=lang)
                print("[OCRSearch] PaddleOCR initialized")
            except Exception as e:
                print(f"[OCRSearch] PaddleOCR init failed: {e}")

    def extract_text(self, image_path: str) -> str:
        if self.ocr is None:
            return ""
        try:
            result = self.ocr.ocr(image_path)
            if result is None or result[0] is None:
                return ""
            texts = []
            for block in result:
                if block is None:
                    continue
                for line in block:
                    if line:
                        texts.append(line[1][0])
            return " ".join(texts)
        except Exception:
            return ""

    def match_candidates(self, image_text: str, kg_entities: List[Dict], threshold: int = 70) -> List[Dict]:
        if not image_text or not kg_entities:
            return []
        matched = []
        image_text_lower = image_text.lower()
        for ent in kg_entities:
            name = ent.get("entity_name", "")
            if not name:
                continue
            score = fuzz.partial_ratio(name.lower(), image_text_lower)
            if score >= threshold:
                matched.append({
                    "name": name,
                    "attrs": ent.get("entity_attributes", {}),
                    "ocr_score": score / 100.0,
                    "source": "ocr"
                })
        matched.sort(key=lambda x: x["ocr_score"], reverse=True)
        return matched[:5]

    def search(self, image_path: str, kg_entities: List[Dict]) -> List[Dict]:
        try:
            text = self.extract_text(image_path)
            if text:
                print(f"[OCRSearch] Extracted text: {text[:80]}...")
            else:
                print("[OCRSearch] No text extracted")
            return self.match_candidates(text, kg_entities)
        except Exception as e:
            print(f"[OCRSearch] Search skipped: {e}")
            return []

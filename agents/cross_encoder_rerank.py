from typing import List, Dict
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, max_length=512)
        print(f"[CrossEncoderReranker] Loaded {model_name}")

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        if not candidates:
            return []
        pairs = []
        for c in candidates:
            name = c.get("name", "")
            attrs = c.get("attrs", {})
            attr_text = " ".join([f"{k}={v}" for k, v in list(attrs.items())[:3]])
            pairs.append((query, f"{name} {attr_text}".strip()))
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
            raw = c.get("raw_score", 0.0)
            c["final_score"] = 0.5 * raw + 0.5 * (float(s) + 10) / 20
        ranked = sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)
        return ranked[:top_k]

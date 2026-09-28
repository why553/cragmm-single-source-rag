from typing import Dict
import re

class AdaptiveRetrieval:
    SIMPLE_KEYWORDS = {"what brand","what color","what is this","who is this","where is","what type","what kind","is this","does this","can this","what logo","what symbol","what car","what animal","what plant"}
    COMPLEX_KEYWORDS = {"when","founded","born","history","first","before","after","compared to","versus","difference","revenue","population","how many","how much","what year","what temperature","calories","production","manufacturer","founder","author","compare"}
    NUMERIC_PATTERNS = [r"\d+",r"how many",r"how much",r"what (year|time|date|age)"]

    def __init__(self):
        print("[AdaptiveRetrieval] Initialized with simple/complex routing")

    def classify_question(self, query: str) -> Dict:
        ql = query.lower()
        is_simple = any(kw in ql for kw in self.SIMPLE_KEYWORDS)
        is_complex = any(kw in ql for kw in self.COMPLEX_KEYWORDS)
        has_numeric = any(re.search(p, ql) for p in self.NUMERIC_PATTERNS)
        level = "complex" if (is_complex or has_numeric or not is_simple) else "simple"
        if level == "simple":
            config = {"level":"simple","use_clip":True,"use_ocr":False,"use_triple":False,"use_web":False,"num_candidates":3,"use_rerank":False}
        else:
            config = {"level":"complex","use_clip":True,"use_ocr":True,"use_triple":True,"use_web":True,"num_candidates":20,"use_rerank":True}
        print(f"[AdaptiveRetrieval] Query: '{query[:50]}...' -> {level} mode")
        return config

    def route(self, query: str) -> Dict:
        return self.classify_question(query)

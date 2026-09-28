from typing import List, Dict, Set
import jieba
import re

STOP_WORDS = {"a","an","the","is","this","that","it","what","how","when","where","why","which","does","can","do","are","was","were","be","been","have","has","had","will","would","could","should","may","might","to","of","in","on","at","for","with","by","from","as","and","or","but","not","no","any","some","more","than","these","those","there","here","about","into","through","over","i","you","he","she","we","they","me","him","her","us","them","my","your","his","its","our","their","also","any","both","each","few","most","other","such","only","own","same","so","too","very","just","now"}

class TripleFilter:
    def __init__(self):
        print("[TripleFilter] Initialized with jieba + stopword filtering")

    def extract_keywords(self, query: str) -> Set[str]:
        words = jieba.lcut(query.lower())
        keywords = set()
        for w in words:
            w = w.strip()
            if len(w) < 3 or w in STOP_WORDS:
                continue
            if re.match(r'^[a-z0-9\u4e00-\u9fff]+$', w):
                keywords.add(w)
        return keywords

    def filter_entities(self, query: str, entities: List[Dict]) -> List[Dict]:
        keywords = self.extract_keywords(query)
        if not keywords:
            return []
        matched = []
        for ent in entities:
            name = ent.get("entity_name", "").lower()
            attrs = ent.get("entity_attributes", {})
            hit_count = 0
            for kw in keywords:
                if kw in name:
                    hit_count += 1
            for k, v in attrs.items():
                if v is None:
                    continue
                val_str = str(v).lower()
                for kw in keywords:
                    if kw in val_str:
                        hit_count += 1
            if hit_count > 0:
                score = min(hit_count / len(keywords), 1.0)
                matched.append({"name": ent.get("entity_name", ""), "attrs": attrs, "triple_score": score, "source": "triple"})
        matched.sort(key=lambda x: x["triple_score"], reverse=True)
        return matched[:10]

    def pre_filter_pool(self, query: str, all_entities: List[Dict]) -> List[Dict]:
        try:
            return self.filter_entities(query, all_entities)
        except Exception as e:
            print(f"[TripleFilter] Filter skipped: {e}")
            return []

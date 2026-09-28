import os
from typing import List, Dict
from openai import OpenAI

class SelfRAGFilter:
    API_KEY = os.environ.get("ZHIPU_API_KEY", "")
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    MODEL_NAME = "glm-4v-plus"

    def __init__(self):
        self.client = OpenAI(api_key=self.API_KEY, base_url=self.BASE_URL)
        print("[SelfRAGFilter] Initialized")

    def score_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        scored = []
        for c in candidates:
            name = c.get("name", "")
            attrs = c.get("attrs", {})
            attr_text = " ".join([f"{k}={v}" for k, v in list(attrs.items())[:3]])
            judge_prompt = f"Question: {query}\nCandidate entity: {name}\nAttributes: {attr_text}\n\nRate how relevant this candidate is to answering the question. Reply with ONLY a number from 0 to 1, where 1 means perfectly relevant."
            score = self._call_judge(judge_prompt)
            c["self_rag_score"] = score
            scored.append(c)
        filtered = [c for c in scored if c["self_rag_score"] >= 0.5]
        filtered.sort(key=lambda x: x["self_rag_score"], reverse=True)
        print(f"[SelfRAG] {len(candidates)} candidates -> {len(filtered)} after filtering")
        return filtered[:5]

    def _call_judge(self, prompt: str) -> float:
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a relevance judge. Reply with only a number from 0 to 1."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10, temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            import re
            nums = re.findall(r"0\.\d+|1\.0|1|0", text)
            if nums:
                return float(nums[0])
            return 0.5
        except Exception as e:
            print(f"[SelfRAG] Judge API error: {e}")
            return 0.5

from typing import Dict, List, Any
from PIL import Image

class UnifiedSearchPipeline:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, query, k=3):
        return []

class BaseAgent:
    def __init__(self, search_pipeline=None):
        self.search_pipeline = search_pipeline or UnifiedSearchPipeline()
    def get_batch_size(self) -> int:
        raise NotImplementedError
    def batch_generate_response(self, queries, images, message_histories) -> List[str]:
        raise NotImplementedError

from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class ModelEvaluatorInterface(ABC):
    def __init__(
        self,
        bible_path: str,
        url: str,
        prompts: List[str],
        chapter_numbers: List[int],
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
        encoding: str = 'utf-8'
    ):
        self.bible_path = bible_path
        self.url = url
        self.prompts = prompts
        self.chapter_numbers = chapter_numbers
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        self.encoding = encoding

    @abstractmethod
    def evaluate(self, max_prompts: int = 1) -> Dict:
        """Run evaluation using model and return results."""
        pass

    @abstractmethod
    def extract_threshold_metrics(self, results: dict) -> dict:
        """Extract threshold metrics like latency or execution time per token size."""
        pass
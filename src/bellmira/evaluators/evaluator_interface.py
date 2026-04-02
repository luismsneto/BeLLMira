from typing import Dict
from abc import ABC, abstractmethod


class ModelEvaluatorInterface(ABC):

    @abstractmethod
    def evaluate(self, max_prompts: int = 1) -> Dict:
        """Run evaluation using the model and return results."""
        pass

    @abstractmethod
    def extract_threshold_metrics(self, results: dict) -> dict:
        """Collapse raw results into a flat metrics dict ready for logging."""
        pass

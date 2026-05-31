from collections import Counter
from typing import Dict, List


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Space-efficient LCS using two rows of DP."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


def rouge_n(hypothesis: str, reference: str, n: int) -> Dict[str, float]:
    """Compute ROUGE-N precision, recall, and F1."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    hyp_counts = Counter(_ngrams(hyp_tokens, n))
    ref_counts = Counter(_ngrams(ref_tokens, n))
    overlap = sum((hyp_counts & ref_counts).values())
    precision = overlap / sum(hyp_counts.values()) if hyp_counts else 0.0
    recall = overlap / sum(ref_counts.values()) if ref_counts else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def rouge_l(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 using longest common subsequence."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compression_ratio(document: str, summary: str) -> float:
    """Return summary word count divided by document word count."""
    doc_words = len(document.split())
    sum_words = len(summary.split())
    return round(sum_words / doc_words, 4) if doc_words > 0 else 0.0

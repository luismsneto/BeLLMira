from typing import List, Dict, Optional


def mean_of_key(records: List[Dict], key: str, ndigits: int = 4) -> Optional[float]:
    """Return the rounded mean of records[key], ignoring None values, or None if no valid values."""
    vals = [r[key] for r in records if r.get(key) is not None]
    return round(sum(vals) / len(vals), ndigits) if vals else None

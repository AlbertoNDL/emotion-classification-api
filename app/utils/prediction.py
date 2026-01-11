from typing import List, Tuple

def argmax_with_confidence(probs: List[float]) -> Tuple[int, float]:
    if not probs:
        raise ValueError("Empty probability list")

    idx = max(range(len(probs)), key=probs.__getitem__)
    return idx, float(probs[idx])

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def hr_ndcg_at_k(ranked_items: np.ndarray, gt: int, k: int) -> Tuple[float, float]:
    topk = ranked_items[:k]
    hit = 1.0 if gt in topk else 0.0
    if hit == 0.0:
        return 0.0, 0.0
    idx = int(np.where(topk == gt)[0][0])
    ndcg = 1.0 / np.log2(idx + 2.0)
    return hit, ndcg

def evaluate_full_ranking(scores: np.ndarray, gt: np.ndarray, seen_mask: np.ndarray, Ks: List[int]) -> Dict[str, float]:
    # scores: [U,I], gt: [U], seen_mask: [U,I] True if should be excluded
    U, I = scores.shape
    out = {f"HR@{k}": 0.0 for k in Ks}
    out.update({f"NDCG@{k}": 0.0 for k in Ks})
    valid_users = 0
    for u in range(U):
        if gt[u] < 0:
            continue
        s = scores[u].copy()
        s[seen_mask[u]] = -1e18
        ranked = np.argsort(-s)
        valid_users += 1
        for k in Ks:
            hr, nd = hr_ndcg_at_k(ranked, int(gt[u]), k)
            out[f"HR@{k}"] += hr
            out[f"NDCG@{k}"] += nd
    if valid_users == 0:
        return out
    for k in Ks:
        out[f"HR@{k}"] /= valid_users
        out[f"NDCG@{k}"] /= valid_users
    return out

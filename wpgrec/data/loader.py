from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class SplitData:
    train_seqs: List[List[int]]
    valid_targets: np.ndarray
    test_targets: np.ndarray
    train_edges: List[Tuple[int,int]]
    n_users: int
    n_items: int

def load_interactions(data_dir: str | Path) -> Tuple[np.ndarray, int, int]:
    data_dir = Path(data_dir)
    inter_path = data_dir / "interactions.txt"
    rows = []
    max_u = -1
    max_i = -1
    with open(inter_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            u, i, t = line.strip().split()
            u = int(u); i = int(i)
            rows.append((u, i, int(float(t))))
            max_u = max(max_u, u)
            max_i = max(max_i, i)
    arr = np.array(rows, dtype=np.int64)
    n_users = max_u + 1
    n_items = max_i + 1
    arr = arr[np.lexsort((arr[:,2], arr[:,0]))]
    return arr, n_users, n_items

def build_splits(interactions: np.ndarray, n_users: int, n_items: int, min_len: int=2) -> SplitData:
    user_items: List[List[int]] = [[] for _ in range(n_users)]
    for u, i, _ in interactions:
        user_items[u].append(int(i))

    train_seqs: List[List[int]] = []
    valid_targets = np.full((n_users,), -1, dtype=np.int64)
    test_targets = np.full((n_users,), -1, dtype=np.int64)
    train_edges: List[Tuple[int,int]] = []

    for u in range(n_users):
        seq = user_items[u]
        if len(seq) < min_len + 1:
            continue
        test_targets[u] = seq[-1]
        valid_targets[u] = seq[-2] if len(seq) >= min_len + 2 else seq[-1]
        train = seq[:-2] if len(seq) >= min_len + 2 else seq[:-1]
        train_seqs.append(train)
        for it in train:
            train_edges.append((u, int(it)))
    return SplitData(train_seqs=train_seqs, valid_targets=valid_targets, test_targets=test_targets,
                     train_edges=train_edges, n_users=n_users, n_items=n_items)

def truncate_pad(seq: List[int], max_len: int, pad: int=0) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq

def build_train_matrix(train_seqs: List[List[int]], n_users: int, max_len: int, pad_id: int=0) -> np.ndarray:
    X = np.full((n_users, max_len), pad_id, dtype=np.int64)
    for u in range(n_users):
        s = train_seqs[u] if u < len(train_seqs) else []
        X[u] = np.array(truncate_pad(s, max_len=max_len, pad=pad_id), dtype=np.int64)
    return X

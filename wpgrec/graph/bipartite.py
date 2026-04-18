from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch

@dataclass
class BipartiteGraph:
    n_users: int
    n_items: int
    n_nodes: int
    adj: torch.Tensor          # sparse COO (N,N), symmetric
    lap_tilde: torch.Tensor    # sparse COO (N,N)

def build_bipartite_graph(n_users: int, n_items: int, edges: List[Tuple[int,int]], device: torch.device) -> BipartiteGraph:
    N = n_users + n_items
    if len(edges) == 0:
        idx = torch.zeros((2,0), dtype=torch.long, device=device)
        val = torch.zeros((0,), dtype=torch.float32, device=device)
        A = torch.sparse_coo_tensor(idx, val, (N,N)).coalesce()
    else:
        ui = torch.tensor(edges, dtype=torch.long, device=device)
        u = ui[:,0]
        i = ui[:,1] + n_users
        src = torch.cat([u, i], dim=0)
        dst = torch.cat([i, u], dim=0)
        idx = torch.stack([src, dst], dim=0)
        val = torch.ones((idx.shape[1],), dtype=torch.float32, device=device)
        A = torch.sparse_coo_tensor(idx, val, (N,N)).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)
    d_inv_sqrt = deg.pow(-0.5)
    r, c = A.indices()
    norm_val = A.values() * d_inv_sqrt[r] * d_inv_sqrt[c]
    A_norm = torch.sparse_coo_tensor(A.indices(), norm_val, (N,N)).coalesce()

    I_idx = torch.arange(N, device=device)
    I = torch.sparse_coo_tensor(torch.stack([I_idx, I_idx]), torch.ones(N, device=device), (N,N)).coalesce()
    L = (I - A_norm).coalesce()
    lap_tilde = (L - I).coalesce()
    return BipartiteGraph(n_users=n_users, n_items=n_items, n_nodes=N, adj=A_norm, lap_tilde=lap_tilde)

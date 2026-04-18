from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn

class ChebyLayer(nn.Module):
    def __init__(self, d: int, K: int):
        super().__init__()
        self.K = K
        self.thetas = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(K+1)])

    def forward(self, lap_tilde: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        T0 = H
        out = self.thetas[0](T0)
        if self.K == 0:
            return out
        T1 = torch.sparse.mm(lap_tilde, H)
        out = out + self.thetas[1](T1)
        for k in range(2, self.K+1):
            Tk = 2.0 * torch.sparse.mm(lap_tilde, T1) - T0
            out = out + self.thetas[k](Tk)
            T0, T1 = T1, Tk
        return out

class ChebyStack(nn.Module):
    def __init__(self, d: int, K: int, Lg: int):
        super().__init__()
        self.layers = nn.ModuleList([ChebyLayer(d=d, K=K) for _ in range(Lg)])

    def forward(self, lap_tilde: torch.Tensor, H0: torch.Tensor) -> torch.Tensor:
        H = H0
        for layer in self.layers:
            H = layer(lap_tilde, H)
        return H

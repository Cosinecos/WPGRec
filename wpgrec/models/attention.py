from __future__ import annotations
import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(d, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, Z: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        # Z: [U, T, d]
        s = self.v(torch.tanh(self.proj(Z))).squeeze(-1)  # [U,T]
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)  # [U,T]
        p = torch.einsum("ut,utd->ud", a, Z)
        return p, a

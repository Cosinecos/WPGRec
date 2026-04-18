from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from wpgrec.wavelets.swpt import swpt_full_tree, band_level_ids
from wpgrec.models.attention import AdditiveAttention
from wpgrec.models.gate import GateMLP
from wpgrec.graph.cheby import ChebyStack

def spectral_flatness(Z: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    # Z: [U, T, d]
    U, T, d = Z.shape
    X = torch.fft.rfft(Z, dim=1)
    P = (X.real ** 2 + X.imag ** 2).mean(dim=2) + eps  # [U, F]
    g = torch.exp(torch.mean(torch.log(P), dim=1))
    a = torch.mean(P, dim=1)
    return g / a

@dataclass
class WPGRecOutput:
    user_z: torch.Tensor
    item_z: torch.Tensor
    gate: torch.Tensor

class WPGRec(nn.Module):
    def __init__(self, n_users: int, n_items: int, d: int, wavelet: str, depth_l: int, tau: int,
                 attn_hidden: int, K: int, Lg: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.wavelet = wavelet
        self.depth_l = depth_l
        self.B = 2 ** depth_l
        self.tau = tau

        self.item_emb = nn.Embedding(n_items, d)
        self.boundary = nn.Parameter(torch.zeros(tau, d))
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.boundary, std=0.02)

        self.attn = nn.ModuleList([AdditiveAttention(d=d, hidden=attn_hidden) for _ in range(self.B)])
        self.gnn = nn.ModuleList([ChebyStack(d=d, K=K, Lg=Lg) for _ in range(self.B)])

        self.gate_mlp = GateMLP(in_dim=3, hidden=attn_hidden, out_dim=self.B)
        self.Wf = nn.Linear(d, d, bias=False)
        self.ln_u = nn.LayerNorm(d)
        self.ln_i = nn.LayerNorm(d)

    def build_sequence_tensor(self, seqs: torch.Tensor) -> torch.Tensor:
        # seqs: [U, T] item ids, 0 can be pad but still an item; caller should ensure pad id exists.
        X = self.item_emb(seqs)  # [U,T,d]
        if self.tau <= 0:
            return X
        U = X.shape[0]
        left = self.boundary.unsqueeze(0).expand(U, -1, -1)
        right = self.boundary.unsqueeze(0).expand(U, -1, -1)
        return torch.cat([left, X, right], dim=1)

    def forward_full(self, seqs: torch.Tensor, lap_tilde: torch.Tensor, user_mask: torch.Tensor | None=None) -> WPGRecOutput:
        # seqs: [U, T]
        Xp = self.build_sequence_tensor(seqs)  # [U, T', d]
        bands = swpt_full_tree(Xp, wavelet=self.wavelet, depth_l=self.depth_l)  # list len B, each [U,T',d]

        U = seqs.shape[0]
        energy = []
        sfm = []
        Xu_b = []

        for b in range(self.B):
            Zb = bands[b]
            Eb = (Zb.pow(2).sum(dim=2).mean(dim=1))  # [U]
            SFMb = spectral_flatness(Zb)  # [U]
            pb, _ = self.attn[b](Zb, mask=user_mask)
            energy.append(Eb)
            sfm.append(SFMb)
            Xu_b.append(pb)

        E = torch.stack(energy, dim=1)  # [U,B]
        S = torch.stack(sfm, dim=1)     # [U,B]
        band_id = band_level_ids(self.depth_l, device=seqs.device, dtype=seqs.dtype).view(1, self.B).expand(U, -1)
        desc = torch.stack([E, S, band_id], dim=2)  # [U,B,3]
        gate_logits = self.gate_mlp(desc)  # [U,B,B]
        gate = torch.softmax(torch.diagonal(gate_logits, dim1=1, dim2=2), dim=1)  # [U,B]

        item0 = self.item_emb.weight  # [I,d]
        user_outs = []
        item_outs = []
        for b in range(self.B):
            Xu = Xu_b[b]  # [U,d]
            H0 = torch.cat([Xu, item0], dim=0)  # [N,d]
            Hb = self.gnn[b](lap_tilde, H0)
            user_outs.append(Hb[:U])
            item_outs.append(Hb[U:])

        Hu = torch.stack(user_outs, dim=1)  # [U,B,d]
        Hi = torch.stack(item_outs, dim=1)  # [I,B,d]
        zu = self.ln_u(self.Wf(torch.einsum("ub,ubd->ud", gate, Hu)))
        zi = self.ln_i(self.Wf(Hi.mean(dim=1)))
        return WPGRecOutput(user_z=zu, item_z=zi, gate=gate)

    @staticmethod
    def scores(user_z: torch.Tensor, item_z: torch.Tensor) -> torch.Tensor:
        return user_z @ item_z.t()

def gate_regularizer(gate: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    # encourage non-degenerate gates
    ent = -torch.sum(gate * torch.log(gate + eps), dim=1).mean()
    return -ent

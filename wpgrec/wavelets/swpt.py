from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F
import pywt

def _make_filters(wavelet: str, device: torch.device, dtype: torch.dtype):
    w = pywt.Wavelet(wavelet)
    lo = torch.tensor(w.dec_lo, device=device, dtype=dtype).flip(0).view(1,1,-1)
    hi = torch.tensor(w.dec_hi, device=device, dtype=dtype).flip(0).view(1,1,-1)
    return lo, hi

def _dilated_conv1d(x: torch.Tensor, filt: torch.Tensor, dilation: int) -> torch.Tensor:
    # x: [B, C, T], filt: [1,1,k]
    B, C, T = x.shape
    k = filt.shape[-1]
    w = filt.repeat(C, 1, 1)  # [C,1,k]
    pad = dilation * (k - 1) // 2
    x = F.pad(x, (pad, pad), mode="reflect")
    y = F.conv1d(x, w, bias=None, stride=1, padding=0, dilation=dilation, groups=C)
    return y

def swpt_full_tree(x: torch.Tensor, wavelet: str, depth_l: int) -> List[torch.Tensor]:
    # x: [B, T, d]
    B, T, d = x.shape
    device = x.device
    dtype = x.dtype
    lo, hi = _make_filters(wavelet, device, dtype)
    x_ch = x.transpose(1,2)  # [B, d, T]
    nodes = [x_ch]
    for level in range(1, depth_l+1):
        dilation = 2 ** (level-1)
        new_nodes = []
        for n in nodes:
            low = _dilated_conv1d(n, lo, dilation=dilation)
            high = _dilated_conv1d(n, hi, dilation=dilation)
            new_nodes.extend([low, high])
        nodes = new_nodes
    # back to [B,T,d]
    return [n.transpose(1,2) for n in nodes]

def band_level_ids(depth_l: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # leaves all at depth_l, but keep a normalized band index as descriptor
    B = 2 ** depth_l
    idx = torch.arange(B, device=device, dtype=dtype)
    return idx / max(1.0, float(B-1))

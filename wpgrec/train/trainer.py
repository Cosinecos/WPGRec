from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from wpgrec.models.wpgrec import WPGRec, gate_regularizer
from wpgrec.train.metrics import evaluate_full_ranking

@dataclass
class TrainState:
    best_metric: float = -1e9
    best_epoch: int = -1

def _chunked_logits(user_z: torch.Tensor, item_z: torch.Tensor, chunk: int) -> torch.Tensor:
    U = user_z.shape[0]
    I = item_z.shape[0]
    if chunk <= 0 or chunk >= I:
        return user_z @ item_z.t()
    outs = []
    for s in range(0, I, chunk):
        e = min(I, s + chunk)
        outs.append(user_z @ item_z[s:e].t())
    return torch.cat(outs, dim=1)

def train_loop(model: WPGRec,
               lap_tilde: torch.Tensor,
               seqs: torch.Tensor,
               train_targets: torch.Tensor,
               seen_mask: np.ndarray,
               valid_targets: np.ndarray,
               test_targets: np.ndarray,
               Ks: List[int],
               epochs: int,
               batch_size: int,
               lr: float,
               weight_decay: float,
               lambda_gate: float,
               full_softmax_chunk: int,
               run_dir: Path,
               patience: int,
               device: torch.device) -> None:

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    state = TrainState()
    run_dir.mkdir(parents=True, exist_ok=True)

    U = seqs.shape[0]
    idx_all = torch.arange(U, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        out = model.forward_full(seqs, lap_tilde)
        user_z = out.user_z
        item_z = out.item_z
        gate = out.gate

        perm = torch.randperm(U, device=device)
        loss_sum = 0.0
        n_batches = 0

        for s in range(0, U, batch_size):
            bidx = perm[s:s+batch_size]
            tgt = train_targets[bidx]
            m = tgt >= 0
            if m.sum().item() == 0:
                continue
            bidx = bidx[m]
            tgt = tgt[m]
            uz = user_z[bidx]

            logits = _chunked_logits(uz, item_z, full_softmax_chunk)
            loss = F.cross_entropy(logits, tgt)

            if lambda_gate > 0:
                loss = loss + lambda_gate * gate_regularizer(gate[bidx])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            out = model.forward_full(seqs, lap_tilde)
            scores = _chunked_logits(out.user_z, out.item_z, full_softmax_chunk).detach().cpu().numpy()
            val = evaluate_full_ranking(scores, valid_targets, seen_mask, Ks)
            test = evaluate_full_ranking(scores, test_targets, seen_mask, Ks)

        key = f"NDCG@{Ks[0]}"
        metric = val.get(key, 0.0)
        log = {"epoch": epoch, "loss": loss_sum / max(1, n_batches), "val": val, "test": test}
        with open(run_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json_dumps(log) + "\n")

        if metric > state.best_metric:
            state.best_metric = metric
            state.best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": {"Ks": Ks}}, run_dir / "best.pt")

        if epoch - state.best_epoch >= patience:
            break

def json_dumps(x) -> str:
    import json
    return json.dumps(x, ensure_ascii=False)

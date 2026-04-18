from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from wpgrec.utils.config import load_yaml, ensure_dir
from wpgrec.utils.seed import seed_all
from wpgrec.data.loader import load_interactions, build_splits, truncate_pad
from wpgrec.graph.bipartite import build_bipartite_graph
from wpgrec.models.wpgrec import WPGRec
from wpgrec.train.trainer import train_loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_dir", default="")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_all(int(cfg.get("seed", 42)))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    interactions, n_users, n_items = load_interactions(args.data_dir)
    sp = build_splits(interactions, n_users, n_items, min_len=int(cfg["dataset"]["min_len"]))

    max_len = int(cfg["dataset"]["max_len"])
    pad_id = 0
    seqs = []
    train_targets = np.full((n_users,), -1, dtype=np.int64)
    seen_mask = np.zeros((n_users, n_items), dtype=bool)

    user_items = [[] for _ in range(n_users)]
    for u, i, _ in interactions:
        user_items[int(u)].append(int(i))

    for u in range(n_users):
        seq = user_items[u]
        if len(seq) < 2:
            seqs.append([pad_id] * max_len)
            continue
        train_tgt = seq[-2] if len(seq) >= 3 else seq[-1]
        train_targets[u] = train_tgt
        train_hist = seq[:-2] if len(seq) >= 3 else seq[:-1]
        for it in train_hist:
            seen_mask[u, it] = True
        seqs.append(truncate_pad(train_hist, max_len=max_len, pad=pad_id))

    seqs = torch.tensor(np.array(seqs, dtype=np.int64), device=device)
    train_targets_t = torch.tensor(train_targets, device=device)

    g = build_bipartite_graph(n_users=n_users, n_items=n_items, edges=sp.train_edges, device=device)

    mcfg = cfg["model"]
    gcfg = cfg["graph"]
    model = WPGRec(n_users=n_users, n_items=n_items, d=int(mcfg["d"]), wavelet=str(mcfg["wavelet"]),
                   depth_l=int(mcfg["depth_l"]), tau=int(mcfg["tau"]),
                   attn_hidden=int(mcfg["attn_hidden"]), K=int(gcfg["K"]), Lg=int(gcfg["Lg"]))
    model.to(device)

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / Path(args.data_dir).name
    run_dir = ensure_dir(run_dir)

    train_loop(model=model, lap_tilde=g.lap_tilde, seqs=seqs, train_targets=train_targets_t,
               seen_mask=seen_mask, valid_targets=sp.valid_targets, test_targets=sp.test_targets,
               Ks=list(cfg["eval"]["Ks"]), epochs=int(cfg["train"]["epochs"]),
               batch_size=int(cfg["train"]["batch_size"]), lr=float(cfg["train"]["lr"]),
               weight_decay=float(cfg["train"]["weight_decay"]),
               lambda_gate=float(cfg["train"]["lambda_gate"]),
               full_softmax_chunk=int(cfg["train"]["full_softmax_chunk"]),
               run_dir=run_dir, patience=int(cfg["train"]["early_stop_patience"]),
               device=device)

if __name__ == "__main__":
    main()

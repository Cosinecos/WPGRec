from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from wpgrec.utils.config import load_yaml
from wpgrec.data.loader import load_interactions, build_splits, truncate_pad
from wpgrec.graph.bipartite import build_bipartite_graph
from wpgrec.models.wpgrec import WPGRec
from wpgrec.train.metrics import evaluate_full_ranking

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--config", default="configs/wpgrec.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    interactions, n_users, n_items = load_interactions(args.data_dir)
    sp = build_splits(interactions, n_users, n_items, min_len=int(cfg["dataset"]["min_len"]))

    max_len = int(cfg["dataset"]["max_len"])
    pad_id = 0
    seqs = []
    seen_mask = np.zeros((n_users, n_items), dtype=bool)
    user_items = [[] for _ in range(n_users)]
    for u, i, _ in interactions:
        user_items[int(u)].append(int(i))
    for u in range(n_users):
        seq = user_items[u]
        train_hist = seq[:-2] if len(seq) >= 3 else seq[:-1]
        for it in train_hist:
            seen_mask[u, it] = True
        seqs.append(truncate_pad(train_hist, max_len=max_len, pad=pad_id))
    seqs = torch.tensor(np.array(seqs, dtype=np.int64), device=device)

    g = build_bipartite_graph(n_users=n_users, n_items=n_items, edges=sp.train_edges, device=device)

    mcfg = cfg["model"]
    gcfg = cfg["graph"]
    model = WPGRec(n_users=n_users, n_items=n_items, d=int(mcfg["d"]), wavelet=str(mcfg["wavelet"]),
                   depth_l=int(mcfg["depth_l"]), tau=int(mcfg["tau"]),
                   attn_hidden=int(mcfg["attn_hidden"]), K=int(gcfg["K"]), Lg=int(gcfg["Lg"]))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        out = model.forward_full(seqs, g.lap_tilde)
        scores = (out.user_z @ out.item_z.t()).detach().cpu().numpy()

    Ks = list(cfg["eval"]["Ks"])
    val = evaluate_full_ranking(scores, sp.valid_targets, seen_mask, Ks)
    test = evaluate_full_ranking(scores, sp.test_targets, seen_mask, Ks)
    print("VALID:", val)
    print("TEST :", test)

if __name__ == "__main__":
    main()

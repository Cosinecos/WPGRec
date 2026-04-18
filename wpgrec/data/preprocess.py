from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import csv

@dataclass
class Interaction:
    u: int
    i: int
    t: int

def read_raw(in_path: str | Path, sep: str, u_col: int, i_col: int, t_col: int, has_header: bool) -> List[Interaction]:
    in_path = Path(in_path)
    rows: List[Interaction] = []
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=sep)
        if has_header:
            next(reader, None)
        for r in reader:
            if not r:
                continue
            rows.append(Interaction(int(r[u_col]), int(r[i_col]), int(float(r[t_col]))))
    return rows

def remap_ids(rows: List[Interaction]) -> Tuple[List[Interaction], int, int]:
    u_map = {}
    i_map = {}
    u_cnt = 0
    i_cnt = 0
    out = []
    for x in rows:
        if x.u not in u_map:
            u_map[x.u] = u_cnt
            u_cnt += 1
        if x.i not in i_map:
            i_map[x.i] = i_cnt
            i_cnt += 1
        out.append(Interaction(u_map[x.u], i_map[x.i], x.t))
    return out, u_cnt, i_cnt

def write_interactions(rows: List[Interaction], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "interactions.txt"
    with open(p, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(f"{x.u} {x.i} {x.t}\n")

def preprocess(in_path: str | Path, out_dir: str | Path, sep: str="\t", u_col: int=0, i_col: int=1, t_col: int=2,
               has_header: bool=False) -> None:
    rows = read_raw(in_path, sep=sep, u_col=u_col, i_col=i_col, t_col=t_col, has_header=has_header)
    rows, n_users, n_items = remap_ids(rows)
    rows.sort(key=lambda x: (x.u, x.t))
    write_interactions(rows, out_dir)
    meta = Path(out_dir) / "meta.yaml"
    with open(meta, "w", encoding="utf-8") as f:
        f.write(f"n_users: {n_users}\n")
        f.write(f"n_items: {n_items}\n")
        f.write(f"n_interactions: {len(rows)}\n")

from __future__ import annotations
import argparse
from wpgrec.data.preprocess import preprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sep", default="\t")
    p.add_argument("--u_col", type=int, default=0)
    p.add_argument("--i_col", type=int, default=1)
    p.add_argument("--t_col", type=int, default=2)
    p.add_argument("--has_header", action="store_true")
    args = p.parse_args()
    preprocess(args.in_path, args.out_dir, sep=args.sep, u_col=args.u_col, i_col=args.i_col, t_col=args.t_col, has_header=args.has_header)

if __name__ == "__main__":
    main()

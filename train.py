import sys
import os
import numpy as np
from time import sleep
from argparse import ArgumentParser


def load_from_ckpt(path):
    with open(path, "r") as src:
        lines = src.read()
        s, i = lines.split(":")
        print(lines)
    return s, i


if __name__ == "__main__":

    ap = ArgumentParser()

    ap.add_argument("--experiment_dir")
    ap.add_argument("--max_iter", type=int, required=True)
    ap.add_argument("--load_from_ckpt", action="store_true")
    ap.add_argument("--random_seed", type=float)

    args, unknown = ap.parse_known_args()
    ckpt_path = os.path.join(args.experiment_dir, "checkpoint.txt")
    print(f"experiment_dir {args.experiment_dir}, {ckpt_path}")

    if args.load_from_ckpt:
        start, increment = load_from_ckpt(ckpt_path)
        start = float(start)
        increment = float(increment)
    else:
        start = np.random.rand() * 1000
        increment = args.random_seed / 100

    i = 0

    print("starting experiment...", start, increment)

    while i < args.max_iter:
        sleep(0.5)
        start -= increment
        print(f"{i+1}: {start}")
        i += 1

    with open(os.path.join(args.experiment_dir, "checkpoint.txt"), "w") as dst:
        dst.write(f"{start}:{increment}")

"""Quick end-to-end demo: synthetic specimens -> nested CV -> metrics.

    python -m picg.run            # quick: physics, constant, ridge, decoder (2 profiles x 2 inner splits)
    python -m picg.run --full     # full decoder search (4 profiles x 3 inner splits), still no ablations

This is a thin preset over ``python -m picg.benchmark``; the benchmark command
runs every variant plus the ablations and writes the full artifact set. The
target is the number of cycles to crack nucleation (first crossing of the
short-to-long-crack transition threshold). On synthetic data the printed
numbers demonstrate that the pipeline runs; they are not a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import benchmark


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, help="specimen CSV (default: generate a synthetic table)")
    parser.add_argument("--n-specimens", type=int, default=56)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--full", action="store_true", help="run the full decoder search")
    parser.add_argument("--out", type=Path, default=Path("runs/quick"), help="run directory for the artifacts")
    args = parser.parse_args(argv)

    forwarded = ["--out", str(args.out), "--n-specimens", str(args.n_specimens), "--seed", str(args.seed),
                 "--variants", "physics", "constant", "ridge", "decoder", "--no-ablations"]
    if args.table:
        forwarded += ["--table", str(args.table)]
    if not args.full:
        forwarded += ["--n-inner", "2", "--n-loss-profiles", "2"]
    return benchmark.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())

"""
Single-run trainer wrapper for the hyperparameters sweep.

This forwards dotted CLI overrides to the local independent trainer
`train_sweep_experiment.py` to ensure the sweep is reproducible even if
the upstream experiment code changes later.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from typing import List, cast


SWEEP_DIR = pathlib.Path(__file__).resolve().parent


def resolve_base_trainer() -> pathlib.Path:
    """Use the local independent training script for sweep stability."""
    return (SWEEP_DIR / "train_sweep_experiment.py").resolve()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-run trainer wrapper")
    parser.add_argument("--", dest="overrides", nargs=argparse.REMAINDER, help="Dotted overrides after --")
    args, _ = parser.parse_known_args(argv)

    overrides = cast(List[str], args.overrides or [])
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    base_trainer = resolve_base_trainer()
    # Forward only the dotted overrides (no separator)
    cmd = [sys.executable, str(base_trainer)] + list(overrides)

    env = os.environ.copy()
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

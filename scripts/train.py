"""Train one forecaster from a YAML config.

Example:
    python scripts/train.py --config configs/patchtst42_etth1.yaml --horizon 96 --seed 2021
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaguetst.experiment import RunSpec, run_experiment  # noqa: E402
from leaguetst.utils import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--horizon", type=int, required=True, help="forecast length T")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch budget")
    parser.add_argument("--patience", type=int, default=None, help="override early stopping")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results/preliminary")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="empty string disables saving")
    parser.add_argument("--tag", default="", help="suffix for the result file name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    loaded = load_yaml(args.config)
    training = dict(loaded.get("training", {}))
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.patience is not None:
        training["patience"] = args.patience
    loaded["training"] = training

    spec = RunSpec.from_yaml_dict(
        loaded,
        horizon=args.horizon,
        seed=args.seed,
        device=args.device,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        checkpoint_dir=args.checkpoint_dir,
        tag=args.tag,
    )
    record = run_experiment(spec)
    print(f"test mse {record['test']['mse']:.4f}  mae {record['test']['mae']:.4f}")


if __name__ == "__main__":
    main()

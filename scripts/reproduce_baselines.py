"""Run the PatchTST/42 and DLinear baselines over horizons and seeds.

Example, the preliminary check at the shortest horizon with three seeds:
    python scripts/reproduce_baselines.py --configs configs/patchtst42_etth1.yaml \
        configs/dlinear_etth1.yaml --horizons 96 --seeds 2021 2022 2023
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
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results/preliminary")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="empty string disables saving")
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-existing", action="store_true", help="keep finished runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    for config_path in args.configs:
        loaded = load_yaml(config_path)
        training = dict(loaded.get("training", {}))
        if args.epochs is not None:
            training["epochs"] = args.epochs
        if args.patience is not None:
            training["patience"] = args.patience
        loaded["training"] = training

        for horizon in args.horizons:
            for seed in args.seeds:
                spec = RunSpec.from_yaml_dict(
                    loaded,
                    horizon=horizon,
                    seed=seed,
                    device=args.device,
                    data_dir=args.data_dir,
                    out_dir=args.out_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    tag=args.tag,
                )
                target = Path(spec.out_dir) / f"{spec.run_name}.json"
                if args.skip_existing and target.exists():
                    logging.info("skip %s", target)
                    continue
                run_experiment(spec)


if __name__ == "__main__":
    main()

"""Write the train and validation rows of both benchmarks to data/trainval/.

Run once, by the project owner, from the folder that holds the full files
(the vault). The output folder gets the leading rows of each file and
MANIFEST.json with the split borders of the full file and both digests.

Example:
    python scripts/make_trainval.py --src ~/leaguetst_vault --out data/trainval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaguetst.data.trainval import make_trainval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="folder with ETT-small/ETTh1.csv and illness/national_illness.csv")
    parser.add_argument("--out", default="data/trainval")
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "MANIFEST.json").exists():
        print(f"{out}/MANIFEST.json exists; the trainval copy is written once. Remove it by hand to rebuild.")
        return 1
    manifest = make_trainval(Path(args.src).expanduser(), out)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The AB4 queue (scripts/ablations/run_ab4_queue.sh) runs exactly the registered items, in order, with the
registered flags, through the AB2 launcher at one job per GPU."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "ablations" / "run_ab4_queue.sh"


def run_lines():
    text = QUEUE.read_text()
    return [line.strip() for line in text.splitlines() if re.match(r"\s*run \S", line)]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_syntax():
    subprocess.run(["bash", "-n", str(QUEUE)], check=True)


def test_items_order_and_flags():
    lines = run_lines()
    assert len(lines) == 4  # two prompted lines inside the ILI/ETTh1 loop, then two transfer lines
    anon, repair, s30, s120 = lines
    assert "--anonymous" in anon and "--no-constrained" not in anon and "--item ABL_P_anon_local" in anon
    assert "--no-constrained" in repair and "--anonymous" not in repair and "--item ABL_P_repair_local" in repair
    for line in (anon, repair):
        assert "prompted" in line and "--seeds 30" in line and "--model $Q15" in line
        assert "--steps" not in line and "--beta" not in line
    for line, steps in ((s30, 30), (s120, 120)):
        assert f"--steps {steps} " in line and f"--item ABL_G_steps{steps}_local" in line
        assert "transfer --source ETTh1 --target ILI --arms league:10" in line
        assert "--beta" not in line and "--model" not in line  # E4 recipe otherwise
    assert re.search(r'for d in ILI ETTh1; do', QUEUE.read_text())  # ILI (tested) before ETTh1 (descriptive)


def test_one_job_per_gpu_lock_and_guards():
    text = QUEUE.read_text()
    assert '"$PY" scripts/ablate_llm.py "$@" --per-gpu 1' in text
    assert 'exec 9> "$LOGS/l4_queue.lock"' in text and "flock -n 9" in text  # shared with the AB2 queue
    assert "git status --porcelain -- leaguetst scripts" in text
    assert "shasum -a 256 -c docs/e4_rerun.sha256" in text
    assert "ab4_queue_status.txt" in text and "l4_queue_status.txt" not in text
    assert "bin/modal" not in text and "MODAL_PROFILE" not in text and "modal_" not in text  # nothing runs on Modal


def test_q15_is_the_pinned_reference_model():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import ablate_llm

    assert "Q15=Qwen/Qwen2.5-1.5B-Instruct" in QUEUE.read_text()
    assert ablate_llm.PINNED["Qwen/Qwen2.5-1.5B-Instruct"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

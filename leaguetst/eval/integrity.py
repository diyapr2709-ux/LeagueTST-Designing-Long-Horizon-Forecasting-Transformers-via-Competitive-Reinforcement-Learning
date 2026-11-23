"""Checks behind gate G3: the test log, the ledger and the E0 lock.

This module is itself in LOCKED_FILES, so the list of locked files and the
checks cannot be weakened without breaking the lock. A lock rewritten by
amending its own commit cannot be seen from inside the repository; the
lock digest printed here is also given to the owner, whose push anchors it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = "prereg/e0_lock.json"
APPEND_ONLY = ("results/ledger.jsonl", "results/test_access.log")
LOCKED_FILES = (
    "leaguetst/data/datasets.py",
    "leaguetst/data/trainval.py",
    "leaguetst/data/test_split.py",
    "data/trainval/MANIFEST.json",
    "leaguetst/eval/integrity.py",
    "leaguetst/eval/metrics.py",
    "leaguetst/eval/numpy_model.py",
    "leaguetst/eval/rescore.py",
    "leaguetst/eval/stats.py",
    "scripts/eval_test.py",
    "scripts/prereg.py",
)


def _git(*args: str, root: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True)


def committed_clean(rel: str, root: Path = ROOT) -> bool:
    return _git("ls-files", "--error-unmatch", rel, root=root).returncode == 0 and \
        _git("diff", "--quiet", "HEAD", "--", rel, root=root).returncode == 0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_only_problems(rel: str, root: Path = ROOT) -> list[str]:
    """Every committed version of ``rel`` must start with the one before it,
    and the working copy must start with the last committed version."""
    commits = _git("log", "--format=%H", "--reverse", "--", rel, root=root).stdout.decode().split()
    previous = b""
    problems = []
    for commit in commits:
        shown = _git("show", f"{commit}:{rel}", root=root)
        if shown.returncode != 0:
            problems.append(f"{rel} was deleted in {commit[:8]}")
            previous = b""
            continue
        if not shown.stdout.startswith(previous):
            problems.append(f"{rel} lost or changed lines in {commit[:8]}")
        previous = shown.stdout
    path = root / rel
    if commits and previous:
        if not path.exists():
            problems.append(f"{rel} is missing from the working tree")
        elif not path.read_bytes().startswith(previous):
            problems.append(f"{rel} lost or changed lines since the last commit")
    return problems


def g3_problems(root: Path = ROOT) -> tuple[list[str], str]:
    """Problems found, and a note with the lock digest and whether the remote holds it."""
    problems: list[str] = []
    for rel in APPEND_ONLY:
        problems += append_only_problems(rel, root)
    log = root / "results" / "test_access.log"
    if log.exists():
        if not committed_clean("results/test_access.log", root):
            problems.append("results/test_access.log has uncommitted changes")
        for line in log.read_text().splitlines():
            if line.strip():
                prereg = json.loads(line).get("prereg", "")
                if not committed_clean(f"prereg/{prereg}.json", root):
                    problems.append(f"test access for {prereg!r} without a committed prereg file")
    lock_path = root / LOCK
    note = ""
    if not lock_path.exists() or not committed_clean(LOCK, root):
        problems.append(f"{LOCK} is missing or not committed unchanged")
        return problems, note
    history = _git("log", "--format=%H", "--", LOCK, root=root).stdout.decode().split()
    if len(history) != 1:
        problems.append(f"{LOCK} was changed after it was written ({len(history)} commits)")
    files = json.loads(lock_path.read_text())["files"]
    missing = [f for f in LOCKED_FILES if f not in files]
    changed = [f for f in LOCKED_FILES if f in files and sha(root / f) != files[f]]
    if missing:
        problems.append("not covered by the lock: " + ", ".join(missing))
    if changed:
        problems.append("changed since the lock: " + ", ".join(changed))
    anchored = bool(history) and _git("merge-base", "--is-ancestor", history[-1], "origin/main", root=root).returncode == 0
    note = f"lock {sha(lock_path)[:16]} in commit {history[-1][:8] if history else '?'}, " + \
        ("anchored on origin/main" if anchored else "not yet on origin/main (the owner's push anchors it)")
    return problems, note


def write_lock(root: Path = ROOT) -> Path:
    path = root / LOCK
    if path.exists():
        raise FileExistsError(f"{path} exists; the lock is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"files": {f: sha(root / f) for f in LOCKED_FILES}}, indent=2, sort_keys=True) + "\n")
    return path

"""AB-P and AB-G on Soumitra's RTX items (AB2 and AB4): Anirudh's independent computation (AB5). CPU.

Moved on 26 Nov from scripts/ablations/llm_tests.py, where AB5 put it, so that llm_tests.py stays byte-identical
to the version Diya's AB4 on main pins by sha256 (08c72a4c...) for scripts/ablations/llm_registered.py, the
computation of record. No function changed in the move; tests/test_ablations.py compares every definition of this
file as committed in the move (943d893) with its text at 523ae42. The statistics (spearman_trend, aucs,
policy_level) are imported unchanged from llm_tests.py.

Added on 26 Nov, after the AB2 and AB4 results were seen, for the area-chair review: the run-path check (B19; every
current arm passes it), and descriptive blocks for the report (each arm's queue DONE time, run commits, the
timeline, Diya's disclosure sentence, whether the result of record exists, common.summarise per arm, per-policy
AUCs). No test, family, permutation, completeness or finality rule changed.

The RTX items of AB2 and AB4 (Soumitra's local GPU) are committed under results/ablations/rtx/ on
origin/soumitra-leg2. They are read from the working tree if a merge has put them there, otherwise from the
pinned commit RTX_REF through git (AB5; each file's source, commit and sha256 recorded; AB2's move manifest must
match or it stops). Every reference is local (AB1, AB4).

AB-P on ILI held-out AUC: anon against ref, repair against ref, scale trend; Holm over 3. AB-G on ILI held-out
AUC: the Spearman correlation between the level rank and AUC, two-sided permutation test, for the step trend
(0, 30, 60, 120) and the KL trend (0, 0.04, 0.1); Holm over 2. A test whose arms are missing counts as not
rejected and keeps the family size.

    .venv/bin/python scripts/ablations/llm_rtx.py      # writes results/ablations/LLM-rtx.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, apply_holm, by_seed, compare, load_table, mean_ci, provenance, read_runs, summarise, \
    write_json  # noqa: E402
from llm_tests import aucs, diagnostics, policy_level, revin_off_share, runs_of, spearman_trend  # noqa: E402


def training_summary(records: list[dict], errors: list[dict]) -> dict:
    return {"policies": len(records), "errors": [{"run_names": e.get("run_names"), "error": e.get("error", "")[:160]} for e in errors],
            "onpolicy_mean_max": max((row["onpolicy_mean"] for r in records for row in r["log"]), default=None),
            "unique_rate_mean": float(np.mean([row["unique_rate"] for r in records for row in r["log"]])) if records else None,
            "minutes_per_policy": mean_ci([r["seconds"] / 60 for r in records]) if len(records) > 1 else None}

RTX_DIR = "results/ablations/rtx"

RTX_REF = "e6926d8664b8493442f4e372a4cf0349a34d06e0"  # origin/soumitra-leg2, pinned; moved forward by a commit of this file when AB4 lands

DEADLINE = "2026-11-28T23:59:00-05:00"  # AB1: an arm unfinished then counts as not run

TRANSFER_METHOD = "grpo_league_from_ETTh1"

# AB2 and AB4: arm -> (item, method of its searches)
PROMPTED = {"ref": ("ABL_P_ref_local", "qwen_prompted_1.5b_local"),
            "anon": ("ABL_P_anon_local", "qwen_prompted_anon_1.5b_local"),
            "repair": ("ABL_P_repair_local", "qwen_prompted_repair_1.5b_local"),
            "0.5B": ("ABL_P_05b_local", "qwen_prompted_0.5b_local"),
            "3B": ("ABL_P_3b_local", "qwen_prompted_3b_local")}

TRANSFER = {"steps30": "ABL_G_steps30_local", "ref": "ABL_G_ref_local", "steps120": "ABL_G_steps120_local",
            "beta0": "ABL_G_beta0_local", "beta01": "ABL_G_beta01_local"}

SEARCHES, POLICIES = 30, 10

REGISTERED_AT = {"AB2": "c2dda8dfa49c6f090816f516285792a00585a79d", "AB4": "15db16314f1303854d4540894d7aaed6fa1a3270"}  # runs must descend from these

LAUNCHER_SHA256 = "3919d68337302590eda42f53ca10158ca27766bb488026b5a4fc8373c477b786"  # scripts/ablate_llm.py (AB2, AB4)

REVISIONS = {"0.5B": "7ae557604adf67be50417f59c2c2f167def9a775", "1.5B": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
             "3B": "aa8e72537993ba99e69dfaafa59ed015b17504d1"}  # AB2

ITEM_AMENDMENT = {"ABL_P_ref_local": "AB2", "ABL_P_05b_local": "AB2", "ABL_P_3b_local": "AB2", "ABL_G_ref_local": "AB2",
                  "ABL_G_beta0_local": "AB2", "ABL_G_beta01_local": "AB2", "ABL_P_anon_local": "AB4", "ABL_P_repair_local": "AB4",
                  "ABL_G_steps30_local": "AB4", "ABL_G_steps120_local": "AB4"}

PROMPTED_MODEL = {"ref": "1.5B", "anon": "1.5B", "repair": "1.5B", "0.5B": "0.5B", "3B": "3B"}

TRANSFER_JOB = {"steps30": (30, 0.04), "ref": (60, 0.04), "steps120": (120, 0.04), "beta0": (60, 0.0), "beta01": (60, 0.1)}  # steps, beta

E4_MANIFEST_COMMIT = "a1751ae2a9dd4323bab484a42625fc835e76f989"  # main's docs/e4_rerun.sha256, the fixed reference (AB1)

STATUS_FILES = ("results/logs/l4_queue_status.txt", "results/logs/ab4_queue_status.txt")

# Added 26 Nov after the AB2 and AB4 results were seen (area-chair review B1, B2, B7, B19). None of these feeds a
# test, a Holm family, completeness or finality; the run-path check is an extra check that every current arm passes.

RUN_PATH = ("leaguetst", "scripts/llm_local.py", "requirements-llm.txt")  # Diya's AB4 (2ab2996): equal to AB2's c2dda8d

MAIN_AB4 = "2ab29963e129a85e294520be49bf53164c002e9c"  # Diya's AB4 on origin/main

DEADLINE_MAIN_AB4 = "2026-11-28T18:00:00-05:00"  # Diya's AB4 deadline for the four AB4 arms; descriptive here (AB1's DEADLINE decides)

RECORD_FILE = "results/ablations/LLM-registered.json"  # the result of record named by Diya's AB4

TIMELINE_COMMITS = {
    REGISTERED_AT["AB2"]: "AB2 registered (origin/soumitra-leg2): the model-scale and KL arms",
    REGISTERED_AT["AB4"]: "Soumitra's AB4 registered (origin/soumitra-leg2): the anon, repair and step arms, local references, "
                          "the tests, the 28 Nov 23:59 ET deadline, policy-level analyses exploratory",
    "7fd5085f9ab9b3d106db37eec917ac3ab566e4fc": "AB4 move of AB2's run files to results/ablations/rtx/ (origin/soumitra-leg2)",
    "510633d53bed6bfbea793bc4e0bbba4d2e875995": "AB5 (this branch): this computation and the scale trend's statistic",
    MAIN_AB4: "Diya's AB4 (origin/main): llm_registered.py as the computation of record, analyses (i) to (iii) registered "
              "secondary, the reading rule, 28 Nov 18:00 ET for the four AB4 arms",
    "185a0b1b4d40283beb65587dcea7f6bd45b0be33": "AB4 prompted arms' run files committed: anon and repair, ILI and ETTh1 (origin/soumitra-leg2)",
    "c7e4bc9cba0de411705842b55349624ba31f6431": "AB4 ABL_G_steps30_local run and training files committed (origin/soumitra-leg2)",
    "e880b0a9c34385d7d46d5f44e2038e7697a28894": "AB4 ABL_G_steps120_local run and training files and the AB4 queue status file committed (origin/soumitra-leg2)",
    RTX_REF: "'Results' commit, the pinned tip of origin/soumitra-leg2: duplicate copies of the run files under results/runs/llm_local/, "
             "scripts/ablations/rtx_tests.py and its tests, copies of common.py and llm_tests.py, and the lock file results/logs/l4_queue.lock; "
             "no AB4 run file is new in it",
}

LLM_TESTS = "scripts/ablations/llm_tests.py"

QUEUE_EVENTS = ("QUEUE_START", "DONE", "INTERRUPTED", "STALE_LOCK", "QUEUE_END")

def fetch_time_of(head: Path, tip: str | None, branch: str = "soumitra-leg2") -> str | None:
    """FETCH_HEAD's time if it lists ``branch`` at ``tip``, else None."""
    from datetime import datetime

    if tip is None or not head.exists():
        return None
    listed = any(line.split()[:1] == [tip] and f"branch '{branch}'" in line for line in head.read_text().splitlines())
    return datetime.fromtimestamp(head.stat().st_mtime).astimezone().isoformat() if listed else None

class RtxSource:
    """Run files of the RTX items: the working tree when it has them, else the teammate branch through git."""

    def __init__(self, ref: str = RTX_REF):
        import subprocess

        self._run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, check=True).stdout
        self.ref = ref
        try:
            self.commit = self._run("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
        except subprocess.CalledProcessError:
            self.commit = None
        self.files: list[dict] = []
        self._cache: dict[str, bytes | None] = {}
        try:  # the local tip of the teammate branch (no fetch), to tell whether the pin is stale
            self.tip = self._run("rev-parse", "--verify", "origin/soumitra-leg2^{commit}").decode().strip()
            self.tip_time = self._run("log", "-1", "--format=%cI", self.tip).decode().strip()
        except subprocess.CalledProcessError:
            self.tip, self.tip_time = None, None
        self.fetch_time = self._fetch_time()
        self._frozen: dict[str, str] | None = None

    def _fetch_time(self) -> str | None:
        """When this clone last fetched origin/soumitra-leg2: FETCH_HEAD's modification time, counted only if FETCH_HEAD
        holds that branch at the local tip (a fetch of another branch alone leaves the tip stale)."""
        import subprocess
        from datetime import datetime

        try:
            common = Path(self._run("rev-parse", "--git-common-dir").decode().strip())
        except subprocess.CalledProcessError:
            return None
        head = (common if common.is_absolute() else ROOT / common) / "FETCH_HEAD"
        return fetch_time_of(head, self.tip)

    def _read(self, rel: str) -> bytes | None:
        if rel not in self._cache:
            self._cache[rel] = self._read_once(rel)
        return self._cache[rel]

    def _read_once(self, rel: str) -> bytes | None:
        import hashlib
        import subprocess

        path, blob = ROOT / rel, None
        if self.commit:
            try:
                blob = self._run("show", f"{self.commit}:{rel}")
            except subprocess.CalledProcessError:
                blob = None
        if path.exists():
            data, source = path.read_bytes(), "working tree"
        elif blob is not None:
            data, source = blob, "git"
        else:
            return None
        entry = {"path": rel, "source": source, "commit": self.commit if source == "git" else None,
                 "sha256": hashlib.sha256(data).hexdigest(), "last_commit_time": self.commit_time(rel)}
        if source == "working tree" and blob is not None and blob != data:
            raise SystemExit(f"{rel}: the working-tree copy differs from the pinned commit {self.commit}")
        self.files.append(entry)
        return data

    def commit_time(self, rel: str) -> str | None:
        """Commit time of the last change to a file on the branch (an upper bound on when its runs finished)."""
        if not self.commit:
            return None
        out = self._run("log", "-1", "--format=%cI", self.commit, "--", rel).decode().strip()
        return out or None

    def descends(self, run_hash: str, registered: str) -> bool:
        import subprocess

        try:
            self._run("merge-base", "--is-ancestor", registered, run_hash)
            return True
        except subprocess.CalledProcessError:
            return False

    def frozen_digests(self) -> dict[str, str]:
        """The ten registered digests, from main's manifest at a fixed commit, never from the run's own commit."""
        if self._frozen is None:
            listed = self._run("show", f"{E4_MANIFEST_COMMIT}:docs/e4_rerun.sha256").decode().splitlines()
            self._frozen = {path: digest for digest, path in (line.split() for line in listed if line.strip())}
        return self._frozen

    def frozen_ok(self, run_hash: str) -> bool:
        """The ten files of docs/e4_rerun.sha256 have their registered digests at this commit (AB1)."""
        import hashlib
        import subprocess

        try:
            expected = self.frozen_digests()
            return len(expected) == 10 and all(hashlib.sha256(self._run("show", f"{run_hash}:{path}")).hexdigest() == digest
                                               for path, digest in expected.items())
        except subprocess.CalledProcessError:
            return False

    def launcher_sha256(self, run_hash: str) -> str | None:
        import hashlib
        import subprocess

        try:
            return hashlib.sha256(self._run("show", f"{run_hash}:scripts/ablate_llm.py")).hexdigest()
        except subprocess.CalledProcessError:
            return None

    def run_path_ok(self, run_hash: str) -> bool:
        """leaguetst/, scripts/llm_local.py and requirements-llm.txt at this commit equal AB2's c2dda8d (git object ids)."""
        import subprocess

        def ids(commit: str) -> list[str]:
            return [self._run("rev-parse", f"{commit}:{path}").decode().strip() for path in RUN_PATH]

        try:
            if getattr(self, "_run_path_ref", None) is None:
                self._run_path_ref = ids(REGISTERED_AT["AB2"])
            return ids(run_hash) == self._run_path_ref
        except subprocess.CalledProcessError:
            return False

    def commit_time_of(self, commit: str) -> str | None:
        """Commit time of a commit, or None if this clone lacks it."""
        import subprocess

        try:
            return self._run("log", "-1", "--format=%cI", commit).decode().strip() or None
        except subprocess.CalledProcessError:
            return None

    def show_at(self, commit: str, rel: str) -> bytes | None:
        """A file at any commit (not a run file: not added to ``files``)."""
        import subprocess

        try:
            return self._run("show", f"{commit}:{rel}")
        except subprocess.CalledProcessError:
            return None

    def resolve(self, ref: str) -> str | None:
        import subprocess

        try:
            return self._run("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
        except subprocess.CalledProcessError:
            return None

    def _list(self, rel_dir: str) -> list[str]:
        """File names in a folder: the pinned commit is the authority; local run files it lacks stop the script."""
        import subprocess

        folder = ROOT / rel_dir
        local = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
        if not self.commit:
            return local
        try:
            pinned = sorted(self._run("ls-tree", "--name-only", f"{self.commit}:{rel_dir}").decode().split())
        except subprocess.CalledProcessError:
            pinned = []
        extra = [n for n in local if n.endswith(".jsonl") and n not in pinned]
        if extra:
            raise SystemExit(f"{rel_dir}: local run files not in the pinned commit {self.commit}: {extra}")
        return pinned

    def rows(self, rel: str) -> list[dict]:
        data = self._read(rel)
        return [json.loads(line) for line in data.decode().splitlines() if line.strip()] if data else []

    def runs(self, item: str, setting_dir: str) -> list[dict]:
        rel = f"{RTX_DIR}/{item}/{setting_dir}"
        return [r for name in self._list(rel) if name.endswith("_runs.jsonl") for r in self.rows(f"{rel}/{name}")]

def duplicates(runs: list[dict]) -> list[tuple[str, int]]:
    """(method, seed) pairs that occur more than once; by_seed would silently keep only the last."""
    from collections import Counter

    return sorted(k for k, n in Counter((r["method"], int(r["seed"])) for r in runs).items() if n > 1)

def prompted_complete(runs: list[dict], method: str) -> bool:
    """AB2: a prompted item is complete on a table when its runs file holds 30 distinct seeds (and no duplicate)."""
    return {int(r["seed"]) for r in runs if r["method"] == method} == set(range(SEARCHES)) and not duplicates(runs)

def stopped_seeds(errors: list[dict]) -> set[int]:
    """Deployment seeds of policies the on-policy check stopped (run name grpo_league_ETTh1_seed{100 + s})."""
    from leaguetst.architect.runner import ON_POLICY_STOP

    return {int(name.rsplit("seed", 1)[1]) - 100 for e in errors if ON_POLICY_STOP in e.get("error", "") for name in e.get("run_names", [])}

def transfer_complete(runs: list[dict], stopped: set[int]) -> bool:
    """AB2: every policy has an outcome and every non-stopped policy is deployed."""
    deployed = {int(r["seed"]) for r in runs if r["method"] == TRANSFER_METHOD}
    return deployed | stopped == set(range(POLICIES)) and not deployed & stopped and not duplicates(runs)

def check_hashes(src: "RtxSource", item: str, hashes: set[str], frozen: bool) -> list[str]:
    """Registration order, clean tree and the registered launcher at every commit that ran or trained an arm."""
    problems = []
    for run_hash in sorted(hashes):
        if run_hash.endswith("-dirty"):
            problems.append(f"run at a dirty tree {run_hash}")
        elif not src.descends(run_hash, REGISTERED_AT[ITEM_AMENDMENT[item]]):
            problems.append(f"commit {run_hash[:12]} does not descend from {ITEM_AMENDMENT[item]}")
        elif src.launcher_sha256(run_hash) != LAUNCHER_SHA256:
            problems.append(f"launcher digest differs at {run_hash[:12]}")
        elif not src.run_path_ok(run_hash):  # added 26 Nov (review B19); every current arm passes
            problems.append(f"run-path code ({', '.join(RUN_PATH)}) differs from AB2's c2dda8d at {run_hash[:12]}")
        elif frozen and not src.frozen_ok(run_hash):
            problems.append(f"docs/e4_rerun.sha256 files differ at {run_hash[:12]} (AB1: such an AB-G arm is exploratory)")
    return problems

def check_runs(src: "RtxSource", item: str, runs: list[dict], revision: str, method: str, frozen: bool) -> list[str]:
    """Problems that make an arm not the registered one (AB5): method, backend, revision, order, launcher, duplicates."""
    problems = []
    methods = {r["method"] for r in runs}
    if methods != {method}:
        problems.append(f"methods {sorted(methods)} instead of {method}")
    envs = [r.get("env", {}) for r in runs]
    if {e.get("backend") for e in envs} != {"local"} or {e.get("per_gpu") for e in envs} != {1}:
        problems.append("backend or jobs per GPU differ from AB2 (local, 1)")
    revisions = {e.get("model_revision") for e in envs}
    if revisions != {revision}:
        problems.append(f"model revisions {sorted(map(str, revisions))} instead of {revision}")
    problems += check_hashes(src, item, {str(e.get("git_hash")) for e in envs}, frozen)
    if duplicates(runs):
        problems.append(f"duplicate (method, seed): {duplicates(runs)[:5]}")
    return problems

def gpu_names(runs: list[dict]) -> list[str]:
    return sorted({str(r.get("env", {}).get("gpu_name")) for r in runs})

def run_commits(records: list[dict]) -> list[str]:
    """The commits stamped in records (env.git_hash), for the report's checks table."""
    return sorted({str(r.get("env", {}).get("git_hash")) for r in records})

def item_rows(src: "RtxSource", item: str, name: str) -> list[dict]:
    """A transfer item's training or error log: AB2 left them in results/runs/<item>/ILI_T24/, AB4's queue copies them
    to results/ablations/rtx/<item>/ILI_T24/. Either place counts; both present and different stops the script.
    (Fixed 26 Nov after the first AB4 run, which read only the AB2 place and so found no AB4 training records.)"""
    found = [rows for rows in (src.rows(f"{RTX_DIR}/{item}/ILI_T24/{name}"), src.rows(f"results/runs/{item}/ILI_T24/{name}")) if rows]
    if len(found) == 2 and found[0] != found[1]:
        raise SystemExit(f"{item}: {name} differs between {RTX_DIR}/ and results/runs/")
    return found[0] if found else []

def check_training(records: list[dict], steps: int, beta: float, stopped: set[int], src: "RtxSource | None" = None, item: str = "") -> list[str]:
    """AB2 and AB4: league rivals, ETTh1 source, the 1.5B revision, lr 1e-4, the item's steps and beta, seeds 100 to 109,
    and (with ``src``) the commits that trained them, as for the searches."""
    problems = []
    if src is not None:
        problems += check_hashes(src, item, {str(r.get("env", {}).get("git_hash")) for r in records}, frozen=True)
        if {r.get("env", {}).get("backend") for r in records} - {"local"} or {r.get("env", {}).get("per_gpu") for r in records} - {1}:
            problems.append("training backend or jobs per GPU differ from AB2 (local, 1)")
    seeds = sorted(int(r["job"]["seed"]) for r in records)
    if seeds != sorted(100 + s for s in range(POLICIES) if s not in stopped):
        problems.append(f"training seeds {seeds}")
    for r in records:
        j = r["job"]
        want = {"steps": steps, "beta": beta, "lr": 1e-4, "rivals": "league", "dataset": "ETTh1", "revision": REVISIONS["1.5B"]}
        bad = {k: j.get(k) for k, v in want.items() if (abs(float(j.get(k, "nan")) - v) > 1e-12 if isinstance(v, float) else j.get(k) != v)}
        if bad:
            problems.append(f"{j.get('run_name')}: {bad}")
    return problems

def done_times(src: "RtxSource") -> dict[str, str]:
    """Last 'DONE <name> rc=0' time per queue name in the AB2 and AB4 status files."""
    out = {}
    for rel in STATUS_FILES:
        data = src._read(rel)
        for line in (data.decode().splitlines() if data else []):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "DONE" and parts[3] == "rc=0":
                out[parts[2]] = parts[0]
    return out

def finished_in_time(name: str, files: list[dict], done: dict[str, str], deadline) -> bool | None:
    """AB1: finished by the deadline, by the queue's DONE time, else by the last commit of its files; None if unknown."""
    from datetime import datetime

    if name in done:
        return datetime.fromisoformat(done[name]) <= deadline
    times = [f["last_commit_time"] for f in files]
    if not times or any(t is None for t in times):
        return None
    return max(datetime.fromisoformat(t) for t in times) <= deadline

def registered_families(prompted: dict[str, list[float] | None], transfer: dict[str, list[float] | None], aligned: bool) -> dict:
    """AB-P (Holm over 3) and AB-G (Holm over 2) from per-seed ILI AUCs in seed order; None marks an arm not complete.

    ``aligned`` says that the three trained steps levels hold the same ten deployment seeds, so position i is policy i.

    A test whose arm is missing is not run and counts as not rejected; a trend with any level missing is not tested."""
    ref = prompted.get("ref")
    p_tests, p_missing = {}, []
    for arm in ("anon", "repair"):
        if ref is not None and prompted.get(arm) is not None:
            p_tests[f"{arm} vs ref_local"] = compare(prompted[arm], ref, 3)
        else:
            p_missing.append(f"{arm} vs ref_local")
    scale = {0.5: prompted.get("0.5B"), 1.5: ref, 3.0: prompted.get("3B")}
    if all(v is not None for v in scale.values()):
        p_tests["scale trend"] = spearman_trend(scale)
    else:
        p_missing.append("scale trend")
    steps = {0: ref, 30: transfer.get("steps30"), 60: transfer.get("ref"), 120: transfer.get("steps120")}
    kl = {0.0: transfer.get("beta0"), 0.04: transfer.get("ref"), 0.1: transfer.get("beta01")}
    g_tests, g_missing = {}, []
    for name, levels in (("steps trend", steps), ("KL trend", kl)):
        if all(v is not None for v in levels.values()):
            g_tests[name] = spearman_trend(levels)
        else:
            g_missing.append(name)
    out = {"AB-P": {"tests": apply_holm(p_tests, 3), "missing": p_missing},
           "AB-G": {"tests": apply_holm(g_tests, 2), "missing": g_missing}}
    if aligned and all(v is not None for v in steps.values()):
        # AB4: addendum 3's two analyses on the RTX levels, exploratory; level i of every arm is policy i.
        out["AB-G"]["exploratory_policy_level"] = policy_level(steps)
    return out

TEST_ARMS = {"anon vs ref_local": [("prompted", "ref"), ("prompted", "anon")],
             "repair vs ref_local": [("prompted", "ref"), ("prompted", "repair")],
             "scale trend": [("prompted", "0.5B"), ("prompted", "ref"), ("prompted", "3B")],
             "steps trend": [("prompted", "ref"), ("transfer", "steps30"), ("transfer", "ref"), ("transfer", "steps120")],
             "KL trend": [("transfer", "beta0"), ("transfer", "ref"), ("transfer", "beta01")]}

def check_manifest(src: "RtxSource") -> dict[str, bool]:
    """AB2's nine run files must match the move manifest (AB4 addendum 1); the script stops otherwise (AB5)."""
    import hashlib

    manifest = src._read(f"{RTX_DIR}/AB2_move.sha256")
    listed = [line.split() for line in manifest.decode().splitlines() if line.strip()] if manifest else []
    result = {}
    for digest, path in listed:
        data = src._read(path)
        result[path] = data is not None and hashlib.sha256(data).hexdigest() == digest
    if len(listed) != 9 or not all(result.values()):
        raise SystemExit(f"AB2 run files do not match results/ablations/rtx/AB2_move.sha256: {result}")
    return result

def arm_reason(st: dict, deadline_passed: bool) -> str:
    if st["problems"]:
        return "failed checks, so not the registered arm (AB5): " + "; ".join(st["problems"])
    if not st["counts_complete"]:
        return "not run (AB1: unfinished at 28 Nov 23:59 ET)" if deadline_passed else "pending (AB4 running)"
    if not st["complete"]:
        return "not run (AB1: not shown to have finished by 28 Nov 23:59 ET)"
    return "complete"

def registered_source(table, etth1, now=None, src: "RtxSource | None" = None) -> dict:
    """The AB2 and AB4 families on the RTX items (registered), with completeness, checks, provenance and descriptive blocks."""
    import hashlib
    import inspect
    from datetime import datetime

    src = src or RtxSource()
    now = now or datetime.now().astimezone()
    deadline = datetime.fromisoformat(DEADLINE)
    deadline_passed = now > deadline
    manifest = check_manifest(src)
    done = done_times(src)
    status = {"prompted": {}, "transfer": {}}
    prompted, transfer, prompted_runs, transfer_runs = {}, {}, {}, {}

    def arm_files(item: str) -> list[dict]:
        prefixes = (f"{RTX_DIR}/{item}/ILI_T24/", f"results/runs/{item}/ILI_T24/")
        return [f for f in src.files if f["path"].startswith(prefixes)]

    def settle(st: dict, name: str, counts_ok: bool, problems: list[str], files: list[dict]) -> bool:
        in_time = finished_in_time(name, files, done, deadline) if counts_ok else None
        st.update({"counts_complete": counts_ok, "problems": problems, "finished_in_time": in_time,
                   "complete": counts_ok and not problems and not (deadline_passed and in_time is not True),
                   "done_time": done.get(name)})  # the queue's DONE time (descriptive, for the report's timeline)
        return st["complete"]

    for arm, (item, method) in PROMPTED.items():
        runs = src.runs(item, "ILI_T24")
        st = status["prompted"][arm] = {"item": item, "searches": len({int(r["seed"]) for r in runs}), "gpu": gpu_names(runs),
                                        "amendment": ITEM_AMENDMENT[item], "commits": run_commits(runs),
                                        "done_time_ETTh1": done.get(f"{item.lower()}_ETTh1")}
        problems = check_runs(src, item, runs, REVISIONS[PROMPTED_MODEL[arm]], method, frozen=False) if runs else []
        ok = settle(st, f"{item.lower()}_ILI", prompted_complete(runs, method), problems, arm_files(item))
        prompted[arm] = aucs(runs, table, method) if ok else None
        prompted_runs[arm] = runs if ok else []
    for arm, item in TRANSFER.items():
        runs = src.runs(item, "ILI_T24")
        errors = item_rows(src, item, "errors.jsonl")
        records = item_rows(src, item, "training.jsonl")
        stopped = stopped_seeds(errors)
        st = status["transfer"][arm] = {"item": item, "deployed": len({int(r["seed"]) for r in runs}), "stopped": sorted(stopped),
                                        "gpu": gpu_names(runs), "training": training_summary(records, errors),
                                        "amendment": ITEM_AMENDMENT[item], "commits": run_commits(runs),
                                        "training_commits": run_commits(records), "training_gpu": gpu_names(records)}
        problems = (check_runs(src, item, runs, REVISIONS["1.5B"], TRANSFER_METHOD, frozen=True)
                    + check_training(records, *TRANSFER_JOB[arm], stopped, src, item)) if runs else []
        ok = settle(st, item.lower(), transfer_complete(runs, stopped), problems, arm_files(item))
        transfer[arm] = aucs(runs, table, TRANSFER_METHOD) if ok else None
        transfer_runs[arm] = runs if ok else []
    aligned = all(transfer[a] is not None and {int(r["seed"]) for r in transfer_runs[a]} == set(range(POLICIES))
                  for a in ("steps30", "ref", "steps120"))
    families = registered_families(prompted, transfer, aligned)
    for fam in families.values():
        fam["missing_reasons"] = {name: "; ".join(f"{status[kind][arm]['item']}: {arm_reason(status[kind][arm], deadline_passed)}"
                                                  for kind, arm in TEST_ARMS[name] if not status[kind][arm]["complete"])
                                  for name in fam["missing"]}
    pinned_is_tip = src.commit is not None and src.commit == src.tip
    missing = bool(families["AB-P"]["missing"] or families["AB-G"]["missing"])
    fetched_after_deadline = src.fetch_time is not None and datetime.fromisoformat(src.fetch_time) > deadline
    # Before the deadline only complete data finalise; after it, only a tip fetched after the deadline (AB5).
    final = pinned_is_tip and (fetched_after_deadline if deadline_passed else not missing)
    provisional_reasons = ([] if final else
                           (["some tests are missing (reasons in the family tables)"] if missing and not deadline_passed else [])
                           + ([] if pinned_is_tip else ["the pinned commit is not the local tip of origin/soumitra-leg2"])
                           + (["origin/soumitra-leg2 not fetched after the deadline"] if deadline_passed and not fetched_after_deadline else []))
    etth1_block = {}
    for arm, (item, method) in PROMPTED.items():
        runs = src.runs(item, "ETTh1_T336")
        if runs and prompted_complete(runs, method) and not check_runs(src, item, runs, REVISIONS[PROMPTED_MODEL[arm]], method, frozen=False):
            etth1_block[arm] = {"summary": summarise(runs, etth1), "diagnostics": diagnostics(runs)}
    return {"source": {"ref": "origin/soumitra-leg2", "commit": src.commit, "tip": src.tip, "tip_time": src.tip_time,
                       "fetch_time": src.fetch_time, "pinned_is_tip": pinned_is_tip,
                       "files": src.files, "ab2_manifest_matches": manifest},
            "spearman_trend_source_sha256": hashlib.sha256(inspect.getsource(spearman_trend).encode()).hexdigest(),
            "computed_at": now.isoformat(), "deadline": DEADLINE, "deadline_passed": deadline_passed, "final": final,
            "provisional_reasons": provisional_reasons,
            "queue_done_times": done, "status": status, "families": families,
            "diagnostics_ILI": {**{f"prompted {a}": diagnostics(r) for a, r in prompted_runs.items() if r},
                                **{f"transfer {a}": diagnostics(r) for a, r in transfer_runs.items() if r}},
            "ETTh1_descriptive": etth1_block,
            "descriptive": rtx_descriptive(table, prompted_runs, transfer_runs)}

def paired_by_seed(a: dict[int, float], b: dict[int, float]) -> dict | None:
    both = sorted(set(a) & set(b))
    if len(both) < 2:
        return None
    return {"seeds": len(both), "a": mean_ci([a[s] for s in both]), "b": mean_ci([b[s] for s in both]),
            "diff": mean_ci([a[s] - b[s] for s in both])}

def rtx_descriptive(table, prompted_runs: dict[str, list[dict]], transfer_runs: dict[str, list[dict]]) -> dict:
    """AB2's hardware check and AB5's descriptive blocks; no tests."""
    out: dict = {}
    j2 = by_seed(read_runs(ROOT / "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl"), TRANSFER_METHOD, table)
    if transfer_runs.get("ref"):
        out["hardware_G_ref_local_vs_J2"] = paired_by_seed(by_seed(transfer_runs["ref"], TRANSFER_METHOD, table), j2)
    if prompted_runs.get("ref"):
        ref = by_seed(prompted_runs["ref"], PROMPTED["ref"][1], table)
        mac = by_seed(runs_of(OUT / "mps" / "ABL_P_ref_mps" / "ILI_T24"), "qwen_prompted_mps", table)
        e3 = by_seed(read_runs(ROOT / "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl"), "qwen_prompted", table)
        out["prompted_ref_local_vs_mac"] = paired_by_seed(ref, mac)
        out["prompted_ref_local_vs_E3_A10"] = paired_by_seed(ref, e3)
    out["revin_off_share_ILI"] = {**{f"prompted {a}": revin_off_share(r) for a, r in prompted_runs.items() if r},
                                  **{f"transfer {a}": revin_off_share(r) for a, r in transfer_runs.items() if r}}
    # Added 26 Nov after the results were seen (review B6, results review R7): common.summarise's AUC in seed-SD units
    # and raw validation MSE per complete arm, and each policy's AUC (by_seed) per transfer arm; descriptive.
    out["summary_ILI"] = {**{f"prompted {a}": summarise(r, table)[PROMPTED[a][1]] for a, r in prompted_runs.items() if r},
                          **{f"transfer {a}": summarise(r, table)[TRANSFER_METHOD] for a, r in transfer_runs.items() if r}}
    out["per_seed_ILI"] = {a: {str(s): v for s, v in sorted(by_seed(r, TRANSFER_METHOD, table).items())} for a, r in transfer_runs.items() if r}
    return out


def queue_events(src: "RtxSource") -> list[dict]:
    """The queue lines of the AB2 and AB4 status files other than START, in file order (descriptive)."""
    out = []
    for rel in STATUS_FILES:
        data = src._read(rel)
        for line in (data.decode().splitlines() if data else []):
            parts = line.split(maxsplit=2)
            if len(parts) >= 2 and parts[1] in QUEUE_EVENTS:
                rest = parts[2] if len(parts) > 2 else ""
                name, detail = ("", rest) if parts[1] not in ("DONE", "INTERRUPTED") else (rest.partition(" ")[0], rest.partition(" ")[2])
                out.append({"time": parts[0], "event": parts[1], "name": name.rstrip(":"), "detail": detail, "file": rel})
    return out


def disclosure(text: str | None) -> str | None:
    """The report disclosure that Diya's AB4 fixes ('Report disclosure, fixed now: "..."'), joined to one line."""
    if not text:
        return None
    flat = " ".join(text.split())
    marker = 'Report disclosure, fixed now: "'
    start = flat.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = flat.find('"', start)
    return flat[start:end] if end > start else None


def record_context(src: "RtxSource") -> dict:
    """Timeline, Diya's disclosure, whether the result of record exists and the digest of llm_tests.py (all descriptive)."""
    import hashlib

    main_tip = src.resolve("origin/main")
    prereg = src.show_at(MAIN_AB4, "docs/prereg_ablations.md")
    places = {"origin/main": main_tip, "Diya's AB4": MAIN_AB4, "origin/soumitra-leg2 (pinned)": src.commit}
    return {"timeline": {"commits": [{"commit": c, "time": src.commit_time_of(c), "label": label} for c, label in TIMELINE_COMMITS.items()],
                         "queue": queue_events(src)},
            "deadline_main_AB4": DEADLINE_MAIN_AB4,
            "disclosure_main_AB4": {"commit": MAIN_AB4, "text": disclosure(prereg.decode() if prereg else None)},
            "record_file": {"path": RECORD_FILE, "working_tree": (ROOT / RECORD_FILE).exists(),
                            "at": {name: {"commit": c, "exists": c is not None and src.show_at(c, RECORD_FILE) is not None}
                                   for name, c in places.items()}},
            "llm_tests_sha256": hashlib.sha256((Path(__file__).resolve().parent / "llm_tests.py").read_bytes()).hexdigest(),
            "llm_tests_history": llm_tests_history(src), "main_AB4_fetched": first_fetch_holding(src, MAIN_AB4)}


def llm_tests_history(src: "RtxSource") -> list[dict]:
    """Commits of this branch that changed llm_tests.py since AB5 (510633d), oldest first, with commit times (review B8)."""
    import subprocess

    try:
        out = src._run("log", "--reverse", "--format=%H %cI", "510633d53bed6bfbea793bc4e0bbba4d2e875995^..HEAD", "--", LLM_TESTS).decode().split("\n")
    except subprocess.CalledProcessError:
        return []
    return [{"commit": c, "time": t} for c, t in (line.split() for line in out if line.strip())]


def first_fetch_holding(src: "RtxSource", commit: str, ref: str = "origin/main") -> str | None:
    """Time of the earliest reflog entry of ``ref`` in this clone at which ``ref`` contained ``commit`` (None if unknown)."""
    import subprocess

    try:
        lines = src._run("reflog", "show", "--date=iso-strict", "--format=%H %gd", ref).decode().splitlines()
    except subprocess.CalledProcessError:
        return None
    times = []
    for line in lines:
        head, _, selector = line.partition(" ")
        if "@{" in selector and src.descends(head, commit):
            times.append(selector.split("@{", 1)[1].rstrip("}"))
    from datetime import datetime

    return min(times, key=datetime.fromisoformat) if times else None


def main() -> int:
    from common import code_hash

    if code_hash().endswith("-dirty"):
        raise SystemExit("scripts/ or leaguetst/ has uncommitted changes; commit before writing results")
    table, _ = load_table("ILI")
    etth1, _ = load_table("ETTh1")
    inputs = ["results/ledger.jsonl", *(str(p.relative_to(ROOT)) for p in sorted((OUT / "mps").glob("*/*/*.jsonl"))),
              "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl", "results/search/E3/ILI_T24/cpu_runs.jsonl",
              "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl", "results/search/E4/ILI_T24/grpo_none_from_ETTh1_runs.jsonl"]
    src = RtxSource()
    registered = registered_source(table, etth1, src=src)
    registered.update(record_context(src))
    result = {"note": "Anirudh's independent computation of the registered AB-P and AB-G families (AB5); the result of record is "
                      "results/ablations/LLM-registered.json from scripts/ablations/llm_registered.py (Diya's AB4 on main)",
              "provenance": provenance(__file__, inputs), "registered": registered}
    import leaguetst

    result["provenance"]["leaguetst_path"] = str(Path(leaguetst.__file__).resolve().parent)
    write_json(OUT / "LLM-rtx.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

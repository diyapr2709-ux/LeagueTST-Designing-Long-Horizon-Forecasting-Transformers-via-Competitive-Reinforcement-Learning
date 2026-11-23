"""Search methods replayed on a tabular benchmark at a fixed epoch budget.

The table holds, per config and seed, the validation curve of a training
run with early stopping. A search run asks an :class:`Oracle` for full
trainings or for the best value up to epoch k. The oracle

* draws, the first time a config is looked up in a run, one of its seeds
  at random, and answers from that seed's run, so the search sees the seed
  noise a live search would see;
* charges each distinct config the epochs it actually ran (or k for a
  screening), once, and stops the run when the next charge would exceed
  the budget;
* records the incumbent (best fully trained config so far) after every
  full lookup, so regret can be read at any smaller budget.

Regret is scored against the table's multi-seed mean, never the observed
draw.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from leaguetst.search.feasible import config_id, feasible_all
from leaguetst.search.space import resolve_choice

FULL_RUN_EPOCHS = 30
MUTABLE_FIELDS = ("patch_len", "stride", "n_layers", "n_heads", "d_model", "ff_ratio", "dropout", "instance_norm")


class BudgetExhausted(Exception):
    """The next lookup would exceed the run's epoch budget."""


@dataclass(frozen=True)
class Observation:
    val_curve: tuple[float, ...]
    train_curve: tuple[float, ...]
    best: float
    epochs_run: int


class Table:
    """Per-config, per-seed observations from ledger rows of one setting.

    A diverged run (non-finite best value or curve entries) keeps its charge
    but its values are replaced by a penalty, ten times the largest finite
    best value, so it ranks last without breaking any method. The regret
    reference is the best multi-seed mean among configs with the most seeds.
    """

    def __init__(self, rows: Iterable[dict[str, Any]], min_seeds: int = 1):
        rows = list(rows)
        finite = [float(r["best_val_mse"]) for r in rows if math.isfinite(float(r["best_val_mse"]))]
        self.penalty = 10.0 * max(finite) if finite else 1e6
        self.n_diverged = 0
        observations: dict[str, dict[int, Observation]] = {}
        self.model_configs: dict[str, dict[str, Any]] = {}
        for row in rows:
            best = float(row["best_val_mse"])
            curve = tuple(v if math.isfinite(v) else self.penalty for v in map(float, row["val_curve"])) or (self.penalty,)
            train = tuple(v if math.isfinite(v) else self.penalty for v in map(float, row.get("train_curve", ())))
            if not math.isfinite(best):
                best = self.penalty
                self.n_diverged += 1
            obs = Observation(curve, train, best, max(1, int(row["epochs_run"])))
            observations.setdefault(row["config_id"], {})[int(row["seed"])] = obs
            self.model_configs[row["config_id"]] = row["model_config"]
        self.obs = {cid: seeds for cid, seeds in observations.items() if len(seeds) >= min_seeds}
        self.ids = sorted(self.obs)
        self._true = {cid: float(np.mean([o.best for o in seeds.values()])) for cid, seeds in self.obs.items()}
        most = max(len(seeds) for seeds in self.obs.values())
        reference = {cid: v for cid, v in self._true.items() if len(self.obs[cid]) == most}
        self.best_id = min(reference, key=reference.get)
        self.best_true = reference[self.best_id]
        self.reference_seeds = most

    def true_value(self, cid: str) -> float:
        return self._true[cid]

    def seed_sd(self) -> float:
        """Median over configs with 3 or more seeds of the seed SD of best val MSE."""
        sds = [np.std([o.best for o in seeds.values()], ddof=1) for seeds in self.obs.values() if len(seeds) >= 3]
        return float(np.median(sds)) if sds else float("nan")


@dataclass
class Oracle:
    table: Table
    budget: int
    seed: int
    rng: np.random.Generator = field(init=False)
    spent: int = 0
    charged: dict[str, int] = field(default_factory=dict)
    obs_seed: dict[str, int] = field(default_factory=dict)
    full_values: dict[str, float] = field(default_factory=dict)
    trace: list[tuple[int, str]] = field(default_factory=list)
    charges: list[tuple[str, int, int, str, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _observation(self, cid: str) -> Observation:
        """The seed this search run sees for ``cid``.

        The draw depends only on (search seed, config id), so every method run
        with the same search seed sees the same draw for the same config
        (common random numbers for paired comparisons).
        """
        seeds = self.table.obs[cid]
        if cid not in self.obs_seed:
            ordered = sorted(seeds)
            digest = hashlib.sha256(f"{self.seed}:{cid}".encode()).hexdigest()
            self.obs_seed[cid] = ordered[int(digest[:12], 16) % len(ordered)]
        return seeds[self.obs_seed[cid]]

    def _charge(self, cid: str, epochs: int, kind: str, read_epoch: int) -> None:
        extra = max(0, epochs - self.charged.get(cid, 0))
        if self.spent + extra > self.budget:
            raise BudgetExhausted
        if extra:
            self.spent += extra
            self.charged[cid] = self.charged.get(cid, 0) + extra
            self.charges.append((cid, self.obs_seed[cid], extra, kind, read_epoch))

    def _record_full(self, cid: str, value: float) -> None:
        if cid not in self.full_values:
            self.full_values[cid] = value
            incumbent = min(self.full_values, key=self.full_values.get)
            self.trace.append((self.spent, incumbent))

    def full(self, cid: str) -> float:
        obs = self._observation(cid)
        self._charge(cid, obs.epochs_run, "full", obs.epochs_run)
        self._record_full(cid, obs.best)
        return obs.best

    def partial(self, cid: str, epoch: int) -> float:
        """Best validation MSE up to ``epoch``; a run that stopped earlier counts as complete."""
        obs = self._observation(cid)
        epochs = min(epoch, obs.epochs_run)
        self._charge(cid, epochs, "partial", epoch)
        if epoch >= obs.epochs_run:
            self._record_full(cid, obs.best)
            return obs.best
        return min(obs.val_curve[:epochs])

    def curves(self, cid: str, epoch: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Validation and training curves up to ``epoch`` (call after ``partial``)."""
        obs = self._observation(cid)
        return obs.val_curve[:epoch], obs.train_curve[:epoch]


class Space:
    """Field domains of the feasible set and the map from choices to config ids."""

    def __init__(self, ids: Iterable[str]):
        feasible = feasible_all()
        allowed = set(ids)
        self.ids = sorted(config_id(c) for c in feasible if config_id(c) in allowed)
        self.id_set = set(self.ids)
        self.choices: dict[str, dict[str, Any]] = {}
        for config in feasible:
            cid = config_id(config)
            if cid in self.id_set:
                choice = config.to_dict()
                choice["stride"] = "half" if config.stride * 2 == config.patch_len else "full"
                self.choices[cid] = choice
        self.domains = {f: sorted({c[f] for c in self.choices.values()}, key=lambda v: (str(type(v)), v)) for f in MUTABLE_FIELDS}

    def to_id(self, choice: dict[str, Any]) -> str | None:
        values = {**choice, "channel_mode": "independent"}
        try:
            cid = config_id(resolve_choice(values))
        except ValueError:
            return None
        return cid if cid in self.id_set else None

    def architecture(self, cid: str) -> tuple:
        c = self.choices[cid]
        return (c["patch_len"], c["stride"], c["n_layers"], c["d_model"], c["ff_ratio"])


# Methods. Each takes (oracle, space, rng) and runs until the budget is spent.

def random_search(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
    for index in rng.permutation(len(space.ids)):
        oracle.full(space.ids[index])


def portfolio(order: list[str]) -> Callable[[Oracle, Space, np.random.Generator], None]:
    """Evaluate a fixed list of configs in order, then fall back to random search."""

    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        seen = set()
        for cid in order:
            if cid in space.id_set and cid not in seen:
                seen.add(cid)
                oracle.full(cid)
        for index in rng.permutation(len(space.ids)):
            if space.ids[index] not in seen:
                oracle.full(space.ids[index])

    return run


def mutate(cid: str, space: Space, rng: np.random.Generator, seen: set[str] | None = None, tries: int = 200) -> str:
    """A feasible one-field mutation, preferring configs not evaluated yet.

    When no unseen feasible mutation turns up in ``tries`` attempts, the
    nearest unseen config to the parent is returned instead.
    """
    parent = space.choices[cid]
    fallback = None
    for _ in range(tries):
        name = MUTABLE_FIELDS[rng.integers(len(MUTABLE_FIELDS))]
        options = [v for v in space.domains[name] if v != parent[name]]
        if not options:
            continue
        child = space.to_id({**parent, name: options[rng.integers(len(options))]})
        if child is None:
            continue
        if seen is None or child not in seen:
            return child
        fallback = fallback or child
    if seen is not None:
        return nearest_unseen(parent, space, seen, rng)
    return fallback or space.ids[rng.integers(len(space.ids))]


def regularized_evolution(population_size: int = 20, sample_size: int = 5):
    """Aging evolution (Real et al., 2019) with one-field mutations inside the feasible set."""

    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        population: deque[tuple[str, float]] = deque()
        seen: set[str] = set()
        for index in rng.permutation(len(space.ids))[:population_size]:
            cid = space.ids[index]
            population.append((cid, oracle.full(cid)))
            seen.add(cid)
        while True:
            picks = rng.choice(len(population), size=min(sample_size, len(population)), replace=False)
            parent = min((population[i] for i in picks), key=lambda item: item[1])[0]
            child = mutate(parent, space, rng, seen)
            population.append((child, oracle.full(child)))
            seen.add(child)
            population.popleft()

    return run


def nearest_unseen(params: dict[str, Any], space: Space, seen: set[str], rng: np.random.Generator) -> str:
    """The unseen feasible config closest to ``params`` in fields changed (ties at random)."""
    best, best_distance = [], None
    for cid in space.ids:
        if cid in seen:
            continue
        choice = space.choices[cid]
        distance = sum(choice[f] != params[f] for f in MUTABLE_FIELDS)
        if best_distance is None or distance < best_distance:
            best, best_distance = [cid], distance
        elif distance == best_distance:
            best.append(cid)
    if not best:
        raise BudgetExhausted
    return best[rng.integers(len(best))]


def tpe(n_startup: int = 10, warm_start: list[str] | None = None, max_trials: int = 5_000):
    """Optuna's multivariate TPE over the fields.

    A proposal that is infeasible or already evaluated is replaced by the
    nearest unseen feasible config; the raw proposal is discarded as failed
    and the evaluated config is added as a completed trial, so the model
    learns only from configs that were actually charged.
    """

    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        distributions = {f: optuna.distributions.CategoricalDistribution(space.domains[f]) for f in MUTABLE_FIELDS}
        sampler = optuna.samplers.TPESampler(seed=int(rng.integers(2**31)), n_startup_trials=n_startup, multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        seen: set[str] = set()

        def evaluate(cid: str) -> None:
            value = oracle.full(cid)
            seen.add(cid)
            study.add_trial(optuna.trial.create_trial(
                params={f: space.choices[cid][f] for f in MUTABLE_FIELDS}, distributions=distributions, value=math.log(value)))

        for cid in warm_start or []:
            if cid in space.id_set and cid not in seen:
                evaluate(cid)
        for _ in range(max_trials):
            trial = study.ask(distributions)
            cid = space.to_id(trial.params)
            if cid is None or cid in seen:
                cid = nearest_unseen(trial.params, space, seen, rng)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            evaluate(cid)

    return run


def asha(rungs: tuple[int, ...] = (1, 3, 9, 30), eta: int = 3):
    """Successive halving with asynchronous promotion, run one job at a time."""

    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        members: list[dict[str, float]] = [dict() for _ in rungs]
        promoted: list[set[str]] = [set() for _ in rungs]
        order = iter(rng.permutation(len(space.ids)))
        while True:
            for level in reversed(range(len(rungs) - 1)):
                ranked = sorted(members[level], key=members[level].get)[: len(members[level]) // eta]
                candidates = [c for c in ranked if c not in promoted[level]]
                if candidates:
                    cid = candidates[0]
                    promoted[level].add(cid)
                    members[level + 1][cid] = oracle.partial(cid, rungs[level + 1])
                    break
            else:
                cid = space.ids[next(order)]
                members[0][cid] = oracle.partial(cid, rungs[0])

    return run


def _features(space: Space, cid: str, val: tuple[float, ...], train: tuple[float, ...], k: int) -> list[float]:
    choice = space.choices[cid]
    row = []
    for name in MUTABLE_FIELDS:
        row += [1.0 if choice[name] == v else 0.0 for v in space.domains[name]]
    val = list(val) + [val[-1]] * (k - len(val))
    train = list(train) + [train[-1] if train else val[-1]] * (k - len(train))
    return row + [math.log(v) for v in val[:k]] + [math.log(max(t, 1e-12)) for t in train[:k]]


def screened_search(k: int, group: int = 8, promote: int = 2, use_referee: bool = True, min_train: int = 10):
    """Random groups screened at epoch k; the best ``promote`` per group are trained fully.

    With ``use_referee`` a gradient-boosted regressor, refitted on every fully
    trained config so far, ranks the screened configs from their fields and
    curves up to epoch k; otherwise they are ranked by their epoch-k value.
    """

    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        order = iter(rng.permutation(len(space.ids)))
        features: list[list[float]] = []
        targets: list[float] = []
        while True:
            batch = []
            for _ in range(group):
                cid = space.ids[next(order)]
                value = oracle.partial(cid, k)
                val, train = oracle.curves(cid, k)
                batch.append((cid, value, _features(space, cid, val, train, k)))
            if use_referee and len(targets) >= min_train:
                model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
                model.fit(np.array(features), np.array(targets))
                scores = model.predict(np.array([b[2] for b in batch]))
            else:
                scores = np.log([b[1] for b in batch])
            for index in np.argsort(scores)[:promote]:
                cid, _, feats = batch[index]
                value = oracle.full(cid)
                features.append(feats)
                targets.append(math.log(value))

    return run


def factorized_controller(group: int = 8, rivals: str = "none", n_rivals: int = 4, lr: float = 0.1,
                          snapshot_every: int = 2, entropy: float = 0.01):
    """One softmax per field, trained with group-normalised policy gradient.

    ``rivals`` is "none", "league" (samples from frozen past snapshots of the
    policy) or "random" (uniform feasible configs). Rivals are charged their
    lookups and enter the group baseline only. An infeasible sample scores
    the group minimum minus one and costs nothing.
    """

    def run(oracle: Oracle, space: Space, rng: np.random.Generator) -> None:
        logits = {f: np.zeros(len(space.domains[f])) for f in MUTABLE_FIELDS}
        moment = {f: np.zeros_like(v) for f, v in logits.items()}
        second = {f: np.zeros_like(v) for f, v in logits.items()}
        snapshots = [{f: v.copy() for f, v in logits.items()}]
        step = 0
        updates = 0
        idle_steps = 0

        def sample(table_logits):
            picks = {}
            for f in MUTABLE_FIELDS:
                p = np.exp(table_logits[f] - table_logits[f].max())
                p /= p.sum()
                picks[f] = int(rng.choice(len(p), p=p))
            return picks

        while True:
            step += 1
            spent_before = oracle.spent
            own = [sample(logits) for _ in range(group)]
            rewards: list[float | None] = []
            for picks in own:
                cid = space.to_id({f: space.domains[f][i] for f, i in picks.items()})
                rewards.append(None if cid is None else -math.log(oracle.full(cid)))
            baseline_rewards = [r for r in rewards if r is not None]
            if rivals != "none":
                for _ in range(n_rivals):
                    if rivals == "league":
                        snap = snapshots[rng.integers(len(snapshots))]
                        cid = space.to_id({f: space.domains[f][i] for f, i in sample(snap).items()})
                    else:
                        cid = space.ids[rng.integers(len(space.ids))]
                    if cid is not None:
                        baseline_rewards.append(-math.log(oracle.full(cid)))
            if not baseline_rewards:
                idle_steps += 1
                if idle_steps >= 50:
                    return
                continue
            updates += 1
            floor = min(baseline_rewards) - 1.0
            filled = [floor if r is None else r for r in rewards]
            pool = np.array(baseline_rewards + [floor] * sum(r is None for r in rewards))
            advantages = (np.array(filled) - pool.mean()) / (pool.std() + 1e-8)
            for f in MUTABLE_FIELDS:
                p = np.exp(logits[f] - logits[f].max())
                p /= p.sum()
                grad = np.zeros_like(p)
                for picks, adv in zip(own, advantages):
                    onehot = np.zeros_like(p)
                    onehot[picks[f]] = 1.0
                    grad += adv * (onehot - p)
                grad /= len(own)
                grad += entropy * (-(p * (np.log(p + 1e-12) + 1)) + p * np.sum(p * (np.log(p + 1e-12) + 1)))
                moment[f] = 0.9 * moment[f] + 0.1 * grad
                second[f] = 0.999 * second[f] + 0.001 * grad**2
                m_hat = moment[f] / (1 - 0.9**updates)
                v_hat = second[f] / (1 - 0.999**updates)
                logits[f] = logits[f] + lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            if updates % snapshot_every == 0:
                snapshots.append({f: v.copy() for f, v in logits.items()})
            # A policy that only proposes configs it has already paid for
            # spends nothing; stop it rather than loop forever.
            idle_steps = idle_steps + 1 if oracle.spent == spent_before else 0
            if idle_steps >= 50:
                return

    return run


# Each factory takes the screening epoch k chosen in E2 (used by the
# screened methods only).
METHODS: dict[str, Callable[..., Callable[[Oracle, Space, np.random.Generator], None]]] = {
    "random": lambda k=1: random_search,
    "re": lambda k=1: regularized_evolution(),
    "tpe": lambda k=1: tpe(),
    "asha": lambda k=1: asha(),
    "epochk": lambda k=1: screened_search(k=k, use_referee=False),
    "referee": lambda k=1: screened_search(k=k, use_referee=True),
    "pg": lambda k=1: factorized_controller(rivals="none"),
    "pg_league": lambda k=1: factorized_controller(rivals="league"),
    "pg_random_rivals": lambda k=1: factorized_controller(rivals="random"),
}


def run_method(method: Callable, table: Table, space: Space, budget: int, seed: int) -> Oracle:
    """Run one search until its budget is spent; return the oracle with its trace."""
    oracle = Oracle(table, budget, seed)
    rng = np.random.default_rng(seed + 1_000_003)
    try:
        method(oracle, space, rng)
    except (BudgetExhausted, StopIteration):
        pass
    return oracle


def regret_at(oracle: Oracle, budget: int, fallback: float) -> float:
    """True value of the incumbent at ``budget`` minus the table's best."""
    incumbent = None
    for spent, cid in oracle.trace:
        if spent <= budget:
            incumbent = cid
        else:
            break
    value = oracle.table.true_value(incumbent) if incumbent else fallback
    return value - oracle.table.best_true


def regret_curve(oracle: Oracle, budgets: Iterable[int], fallback: float) -> list[float]:
    return [regret_at(oracle, b, fallback) for b in budgets]


def regret_from_trace(trace: Iterable, table: Table, budget: int, fallback: float) -> tuple[float, str | None]:
    """Regret of the incumbent at ``budget`` and its config id, from a (spent, id) trace."""
    incumbent = None
    for spent, cid in trace:
        if spent <= budget:
            incumbent = cid
        else:
            break
    value = table.true_value(incumbent) if incumbent else fallback
    return value - table.best_true, incumbent


def no_incumbent_value(table: Table) -> float:
    """Value used for the regret of a run with no fully trained config yet: the table mean."""
    return float(np.mean([table.true_value(c) for c in table.ids]))


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(Path(runs_dir).glob("*_runs.jsonl")):
        runs += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return runs


def claim_values(table: Table, runs_dir: Path, method: str, baseline: str, budget_eq: int) -> tuple[list[float], list[float]]:
    """Per-search-seed regrets in seed-SD units of ``method`` and ``baseline`` at one budget."""
    sigma = table.seed_sd()
    fallback = no_incumbent_value(table)
    runs = load_runs(runs_dir)

    def values(name: str) -> list[float]:
        chosen = sorted((r for r in runs if r["method"] == name), key=lambda r: r["seed"])
        if not chosen:
            raise ValueError(f"no runs for {name}")
        return [regret_from_trace(r["trace"], table, budget_eq * FULL_RUN_EPOCHS, fallback)[0] / sigma for r in chosen]

    return values(method), values(baseline)

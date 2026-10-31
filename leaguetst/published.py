"""Published test errors of the two baselines and the protocol behind them.

The values are those of Table 3 in Nie et al. (ICLR 2023), which lists
PatchTST/42 next to DLinear. Both official code bases ran one seed, 2021,
and both test loaders dropped the last partial batch when these numbers
were produced: PatchTST still does, and LTSF-Linear switched to keeping
every window in March 2023, after publication. The drop-last means of a
run are therefore the like-for-like comparison.
"""

from __future__ import annotations

PUBLISHED_SEED = 2021

# (model, dataset, horizon) -> (test MSE, test MAE)
PUBLISHED: dict[tuple[str, str, int], tuple[float, float]] = {
    ("patchtst", "ETTh1", 96): (0.375, 0.399),
    ("patchtst", "ETTh1", 192): (0.414, 0.421),
    ("patchtst", "ETTh1", 336): (0.431, 0.436),
    ("patchtst", "ETTh1", 720): (0.449, 0.466),
    ("dlinear", "ETTh1", 96): (0.375, 0.399),
    ("dlinear", "ETTh1", 192): (0.405, 0.416),
    ("dlinear", "ETTh1", 336): (0.439, 0.443),
    ("dlinear", "ETTh1", 720): (0.472, 0.490),
    ("patchtst", "ILI", 24): (1.522, 0.814),
    ("patchtst", "ILI", 36): (1.430, 0.834),
    ("patchtst", "ILI", 48): (1.673, 0.854),
    ("patchtst", "ILI", 60): (1.529, 0.862),
    ("dlinear", "ILI", 24): (2.215, 1.081),
    ("dlinear", "ILI", 36): (1.963, 0.963),
    ("dlinear", "ILI", 48): (2.130, 1.024),
    ("dlinear", "ILI", 60): (2.368, 1.096),
}

"""Independent evaluation: NumPy re-scorer, metrics and paired statistics.

Nothing here imports the training loop, the torch model or the torch
datasets, so a bug there cannot hide in the re-score.
"""

from leaguetst.eval.metrics import exact_and_drop_last
from leaguetst.eval.numpy_model import numpy_patchtst_forward
from leaguetst.eval.rescore import rescore_checkpoint, validation_windows
from leaguetst.eval.stats import endpoint_bootstrap, holm, paired_two_level_bootstrap, permutation_test

__all__ = [
    "endpoint_bootstrap",
    "exact_and_drop_last",
    "holm",
    "numpy_patchtst_forward",
    "paired_two_level_bootstrap",
    "permutation_test",
    "rescore_checkpoint",
    "validation_windows",
]

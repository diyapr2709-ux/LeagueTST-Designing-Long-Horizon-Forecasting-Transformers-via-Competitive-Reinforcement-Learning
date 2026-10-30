"""Error metrics accumulated over batches of any size."""

from __future__ import annotations

import torch


class ErrorAccumulator:
    """Per-sample error sums over every window seen.

    Summing per window and dividing once at the end gives the exact mean over
    the split, whatever the batch size. The same sums also give the mean over
    the first ``floor(n / batch_size) * batch_size`` windows, which is what
    the official PatchTST evaluation reports because its test loader drops
    the last partial batch. Both are returned by :meth:`summary` so the two
    protocols can be compared.
    """

    def __init__(self) -> None:
        self.squared: list[torch.Tensor] = []
        self.absolute: list[torch.Tensor] = []
        self.elements_per_sample = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        # Per-sample sums are taken on the device in its own precision and
        # moved to the CPU as float64; MPS has no float64 of its own.
        error = (prediction.detach() - target.detach()).flatten(start_dim=1)
        self.squared.append(error.pow(2).sum(dim=1).cpu().double())
        self.absolute.append(error.abs().sum(dim=1).cpu().double())
        self.elements_per_sample = error.shape[1]

    @property
    def n_samples(self) -> int:
        return sum(len(chunk) for chunk in self.squared)

    def _means(self, n_keep: int) -> tuple[float, float]:
        if n_keep <= 0:
            return float("nan"), float("nan")
        squared = torch.cat(self.squared)[:n_keep]
        absolute = torch.cat(self.absolute)[:n_keep]
        denominator = n_keep * self.elements_per_sample
        return float(squared.sum() / denominator), float(absolute.sum() / denominator)

    @property
    def mse(self) -> float:
        return self._means(self.n_samples)[0]

    @property
    def mae(self) -> float:
        return self._means(self.n_samples)[1]

    def summary(self, batch_size: int | None = None) -> dict[str, float | int]:
        """Exact means, plus the drop-last means when ``batch_size`` is given."""
        n = self.n_samples
        mse, mae = self._means(n)
        result: dict[str, float | int] = {"mse": mse, "mae": mae, "n_samples": n}
        if batch_size is not None:
            n_keep = (n // batch_size) * batch_size
            mse_drop, mae_drop = self._means(n_keep)
            result.update(
                {"mse_drop_last": mse_drop, "mae_drop_last": mae_drop, "n_samples_drop_last": n_keep}
            )
        return result

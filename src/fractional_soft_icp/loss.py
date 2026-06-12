from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor, nn

Reduction = Literal["none", "mean", "sum"]


def _validate_inputs(
    source: Tensor,
    target: Tensor,
    fraction: float,
    sigma: float,
    reduction: Reduction,
    chunk_size: Optional[int],
) -> None:
    if source.ndim not in (2, 3) or source.shape[-1] != 3:
        raise ValueError("source must have shape (N, 3) or (B, N, 3)")
    if target.ndim != source.ndim or target.shape[-1] != 3:
        raise ValueError("target must have the same rank as source and end in 3")
    if source.ndim == 3 and source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same batch size")
    if source.shape[-2] == 0 or target.shape[-2] == 0:
        raise ValueError("source and target must contain at least one point")
    if not source.is_floating_point() or not target.is_floating_point():
        raise TypeError("source and target must be floating-point tensors")
    if source.device != target.device:
        raise ValueError("source and target must be on the same device")
    if source.dtype != target.dtype:
        raise ValueError("source and target must have the same dtype")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the interval (0, 1]")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer or None")


def _target_chunks(target: Tensor, chunk_size: Optional[int]):
    size = target.shape[1]
    step = size if chunk_size is None else chunk_size
    for start in range(0, size, step):
        yield target[:, start : start + step]


def _nearest_squared_distances(
    source: Tensor, target: Tensor, chunk_size: Optional[int]
) -> Tensor:
    nearest = torch.full(
        source.shape[:2],
        torch.inf,
        device=source.device,
        dtype=source.dtype,
    )
    for target_chunk in _target_chunks(target, chunk_size):
        distances = torch.cdist(source, target_chunk).square()
        nearest = torch.minimum(nearest, distances.min(dim=-1).values)
    return nearest


def _soft_expected_squared_distance(
    source: Tensor,
    target: Tensor,
    sigma: float,
    chunk_size: Optional[int],
) -> Tensor:
    batch_size, point_count = source.shape[:2]
    max_logit = torch.full(
        (batch_size, point_count),
        -torch.inf,
        device=source.device,
        dtype=source.dtype,
    )
    weight_sum = torch.zeros_like(max_logit)
    weighted_distance_sum = torch.zeros_like(max_logit)
    temperature = 2.0 * sigma**2

    for target_chunk in _target_chunks(target, chunk_size):
        squared_distances = torch.cdist(source, target_chunk).square()
        logits = -squared_distances / temperature
        chunk_max = logits.max(dim=-1).values
        new_max = torch.maximum(max_logit, chunk_max)

        old_scale = torch.exp(max_logit - new_max)
        chunk_scale = torch.exp(chunk_max - new_max)
        chunk_weights = torch.exp(logits - chunk_max.unsqueeze(-1))

        weight_sum = (
            weight_sum * old_scale
            + chunk_weights.sum(dim=-1) * chunk_scale
        )
        weighted_distance_sum = (
            weighted_distance_sum * old_scale
            + (chunk_weights * squared_distances).sum(dim=-1) * chunk_scale
        )
        max_logit = new_max

    return weighted_distance_sum / weight_sum


def fractional_soft_icp_loss(
    source: Tensor,
    target: Tensor,
    *,
    fraction: float = 0.25,
    sigma: float = 0.1,
    reduction: Reduction = "mean",
    chunk_size: Optional[int] = None,
) -> Tensor:
    """Compute the differentiable fractional soft-ICP loss.

    The closest ``fraction`` of source points is selected according to its
    nearest-target distance. Each selected point is then softly matched to all
    target points with a Gaussian kernel, and the expected squared distances
    are averaged.

    Args:
        source: Moving points with shape ``(N, 3)`` or ``(B, N, 3)``.
        target: Fixed points with shape ``(M, 3)`` or ``(B, M, 3)``.
        fraction: Fraction of source points used by the loss, in ``(0, 1]``.
        sigma: Gaussian correspondence bandwidth, in the points' units.
        reduction: Reduction across batches: ``"none"``, ``"mean"``, or
            ``"sum"``.
        chunk_size: Optional number of target points processed at once. Use it
            to reduce peak memory for large point clouds.

    Returns:
        A scalar tensor, or one loss per batch item when ``reduction="none"``.
    """
    _validate_inputs(source, target, fraction, sigma, reduction, chunk_size)

    unbatched = source.ndim == 2
    if unbatched:
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)

    nearest = _nearest_squared_distances(source, target, chunk_size)
    selected_count = max(1, int(fraction * source.shape[1]))
    selected_indices = nearest.topk(
        selected_count, dim=-1, largest=False
    ).indices
    selected_source = source.gather(
        1, selected_indices.unsqueeze(-1).expand(-1, -1, 3)
    )

    point_losses = _soft_expected_squared_distance(
        selected_source, target, sigma, chunk_size
    )
    batch_losses = point_losses.mean(dim=-1)

    if unbatched:
        batch_losses = batch_losses.squeeze(0)
    if reduction == "none":
        return batch_losses
    if reduction == "sum":
        return batch_losses.sum()
    return batch_losses.mean()


class FractionalSoftICPLoss(nn.Module):
    """``torch.nn.Module`` wrapper around :func:`fractional_soft_icp_loss`."""

    def __init__(
        self,
        fraction: float = 0.25,
        sigma: float = 0.1,
        reduction: Reduction = "mean",
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.fraction = fraction
        self.sigma = sigma
        self.reduction = reduction
        self.chunk_size = chunk_size

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        return fractional_soft_icp_loss(
            source,
            target,
            fraction=self.fraction,
            sigma=self.sigma,
            reduction=self.reduction,
            chunk_size=self.chunk_size,
        )

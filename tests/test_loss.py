import pytest
import torch

from fractional_soft_icp import FractionalSoftICPLoss, fractional_soft_icp_loss


def reference_loss(source, target, fraction=0.25, sigma=0.1):
    distances = torch.cdist(source, target)
    nearest = distances.min(dim=1).values
    selected_count = max(1, int(fraction * source.shape[0]))
    indices = nearest.topk(selected_count, largest=False).indices
    squared_distances = torch.cdist(source[indices], target).square()
    weights = torch.softmax(-squared_distances / (2 * sigma**2), dim=1)
    return (weights * squared_distances).sum(dim=1).mean()


def test_matches_original_implementation():
    torch.manual_seed(7)
    source = torch.randn(12, 3)
    target = torch.randn(9, 3)

    actual = fractional_soft_icp_loss(
        source, target, fraction=0.5, sigma=0.2
    )
    expected = reference_loss(source, target, fraction=0.5, sigma=0.2)

    torch.testing.assert_close(actual, expected)


def test_chunked_computation_matches_full_computation():
    torch.manual_seed(11)
    source = torch.randn(2, 13, 3)
    target = torch.randn(2, 17, 3)

    full = fractional_soft_icp_loss(
        source, target, fraction=0.4, sigma=0.3, reduction="none"
    )
    chunked = fractional_soft_icp_loss(
        source,
        target,
        fraction=0.4,
        sigma=0.3,
        reduction="none",
        chunk_size=5,
    )

    torch.testing.assert_close(chunked, full)


def test_supports_gradients():
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        requires_grad=True,
    )
    target = torch.tensor([[0.1, 0.0, 0.0], [0.7, 0.0, 0.0]])

    loss = fractional_soft_icp_loss(source, target, fraction=2 / 3)
    loss.backward()

    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert source.grad.abs().sum() > 0


def test_module_and_batch_reductions():
    source = torch.zeros(2, 4, 3)
    target = torch.ones(2, 3, 3)
    module = FractionalSoftICPLoss(fraction=0.5, reduction="none")

    losses = module(source, target)

    assert losses.shape == (2,)
    torch.testing.assert_close(losses[0], losses[1])


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_rejects_invalid_fraction(fraction):
    points = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="fraction"):
        fractional_soft_icp_loss(points, points, fraction=fraction)


def test_rejects_non_floating_points():
    points = torch.zeros(2, 3, dtype=torch.long)
    with pytest.raises(TypeError, match="floating-point"):
        fractional_soft_icp_loss(points, points)

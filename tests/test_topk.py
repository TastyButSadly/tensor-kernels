import pytest
import torch

from moe_ops import top8_experts, topk_experts


def reference(scores, k):
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]


def test_output_shape():
    for shape in [(17,), (4, 17), (2, 3, 4, 17), (0, 17), (2, 0)]:
        for k in [0, 1, 5, 17, 22]:
            scores = torch.zeros(shape, dtype=torch.float32)
            assert topk_experts(scores, k).shape == (*shape[:-1], min(k, shape[-1]))


def test_top8_output_shape():
    for shape in [(12,), (4, 12), (2, 3), (2, 0)]:
        assert top8_experts(torch.zeros(shape)).shape == (*shape[:-1], min(8, shape[-1]))


def test_output_dtype():
    scores = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    actual = topk_experts(scores, 2)
    assert actual.dtype == torch.int64
    assert actual.device.type == "cpu"
    assert not actual.requires_grad


def test_top8_output_dtype():
    assert top8_experts(torch.zeros(2, 12)).dtype == torch.int64


def test_exact_input_matches_reference():
    scores = torch.tensor([0.2, 0.9, 0.9, -1.0])
    for x in [scores, scores.repeat(4, 1)]:
        assert torch.equal(topk_experts(x, 2), reference(x, 2))


def test_top8_exact_input_matches_reference():
    scores = torch.tensor([2.0, 5.0, 5.0, -1.0, 5.0, 2.0, 0.0, 7.0, -3.0, 6.0])
    for x in [scores, scores.repeat(4, 1), scores[:3]]:
        assert torch.equal(top8_experts(x), reference(x, 8))


def test_varied_inputs_match_reference():
    generator = torch.Generator().manual_seed(2026)
    inputs = [
        torch.tensor([[2.0, 5.0, 5.0, -1.0, 5.0, 2.0], [7.0, 0.0, 3.0, 9.0, 4.0, 2.0]]),
        torch.tensor([0.0, float("nan"), -0.0, float("inf"), float("nan"), float("-inf")]),
        torch._neg_view(torch.tensor([1.0, 2.0, 3.0])),
        torch.randn((2, 3, 4, 17), generator=generator),
    ]
    for scores in inputs:
        for k in [0, 1, 2, 8, scores.shape[-1] + 2]:
            assert torch.equal(topk_experts(scores, k), reference(scores, k))


def test_noncontiguous_input_matches_reference_without_mutation():
    scores = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
    before = scores.clone()
    assert not scores.is_contiguous()
    assert torch.equal(topk_experts(scores, 2), reference(scores, 2))
    assert torch.equal(scores, before)


def test_invalid_dtype_raises_runtime_error():
    for dtype in [torch.int64, torch.float64]:
        with pytest.raises(RuntimeError, match="expected torch.float32"):
            topk_experts(torch.ones(3, dtype=dtype), 1)


def test_invalid_arguments_raise_runtime_error():
    sparse = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2))
    for scores, k, message in [
        (torch.tensor(1.0), 1, "expected at least one dimension"),
        (sparse, 1, "expected a strided tensor"),
        (torch.ones(3), -1, "k must be nonnegative"),
    ]:
        with pytest.raises(RuntimeError, match=message):
            topk_experts(scores, k)

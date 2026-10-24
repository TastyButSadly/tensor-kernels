import torch
from moe_ops import top8_experts

def test_top8_readme_example():
    x = torch.tensor([0.2, 0.9, 0.9, -1.0])
    expected = torch.tensor([1, 2, 0, 3])
    result = top8_experts(x)
    assert torch.equal(result, expected)
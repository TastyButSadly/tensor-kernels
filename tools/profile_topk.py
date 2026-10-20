"""Run repeated CPU top-k calls for python -m cProfile."""

import argparse

import torch

from moe_ops import topk_experts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.tokens, args.experts, args.iterations, args.threads) < 1 or args.k < 0:
        parser.error("sizes, iterations, and threads must be positive; k must be nonnegative")

    torch.set_num_threads(args.threads)
    scores = torch.randn(args.tokens, args.experts, generator=torch.Generator().manual_seed(17))
    expected = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., : args.k]
    assert torch.equal(topk_experts(scores, args.k), expected)
    for _ in range(args.iterations):
        result = topk_experts(scores, args.k)
    print(f"shape={tuple(scores.shape)}, k={args.k}, iterations={args.iterations}, "
          f"output_shape={tuple(result.shape)}, threads={args.threads}")


if __name__ == "__main__":
    main()

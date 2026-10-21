# Tensor kernels: CPU Top-K expert selection

A C++ extension for PyTorch that returns the indices of the highest-scoring
experts for each token. It implements one routing component of a Mixture of
Experts model; it does not execute experts or train a model.

## Operation

```python
import torch
from moe_ops import topk_experts, top8_experts

x = torch.tensor([0.2, 0.9, 0.9, -1.0])
print(topk_experts(x, k=2))  # tensor([1, 2])
```

`moe_ops.topk_experts(x, k)` matches, for nonnegative integer k:

```python
torch.argsort(x, dim=-1, descending=True, stable=True)[..., :k]
```

`moe_ops.top8_experts(x)` is the fixed-k operation with reference:

```python
torch.argsort(x, dim=-1, descending=True, stable=True)[..., :8]
```

The input must be a strided CPU float32 tensor with at least one dimension.
The last dimension contains expert scores; leading dimensions are preserved.
The output is int64, with last dimension `min(k, x.shape[-1])`. Equal scores
retain increasing input index order; NaNs precede numeric scores. Empty and
noncontiguous inputs are supported. Inputs are not modified.

C++ uses `std::partial_sort` on expert indices. Scalar inputs, other dtypes,
non-CPU devices, unsupported layouts, and negative k raise `RuntimeError`
through `TORCH_CHECK`. Selection returns discrete indices and has no gradient.

## Build and install

Use Linux, Python 3.10-3.13, a C++17 compiler, and Python development headers.
Verified with Ubuntu 24.04, GCC 13, Python 3.12, and CPU PyTorch 2.8.0.
From the repository root:

```bash
python3 -m venv /tmp/moe-build
/tmp/moe-build/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu /tmp/moe-build/bin/pip wheel . --no-deps -w dist

python3 -m venv /tmp/moe-run
/tmp/moe-run/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
/tmp/moe-run/bin/pip install dist/*.whl
(cd /tmp && /tmp/moe-run/bin/python -c 'import torch; from moe_ops import top8_experts; print(top8_experts(torch.tensor([0.2, 0.9, 0.9, -1.0])))')
```

The CPU index supplies PyTorch; PyPI supplies build tools. The wheel contains
our compiled extension, not PyTorch, and is specific to its Python ABI and platform.
The build and runtime both require PyTorch 2.8.0.

## Validation status

Wheel build/install, exact reference comparisons, C++ error paths, and concurrent
calls have been checked. The dedicated pytest suite, Docker/smoke setup, Makefile,
and push CI are assigned to the other contributor and are not present yet.
Once tests are supplied, run them against the installed wheel outside the checkout:

```bash
/tmp/moe-run/bin/pip install pytest
test_dir=$(mktemp -d)
cp -r tests "$test_dir/tests"
(cd /tmp && /tmp/moe-run/bin/python -m pytest -q "$test_dir/tests")
```

## Project note

- The implementation in commit 0bc76aa failed a reference comparison on a lazy
  negative view of `[1., 2., 3.]`: logical scores were `[-1., -2., -3.]`, but
  top-2 returned `[2, 1]` instead of `[0, 1]`. Fix commit 038e796 materializes
  negation with `resolve_neg()` before reading storage. The reproducer uses
  `torch._neg_view(torch.tensor([1.,2.,3.]))`.
- In commit 432bad1, a long native call still held Python's GIL, preventing another
  Python thread from progressing. Fix commit 04b6a16 releases the GIL during C++
  execution. A heartbeat
  check observed progress during the call after the fix; concurrent results and
  error propagation were also verified.
- Float64 is deliberately unsupported: it raises `RuntimeError` with
  `topk_experts: expected torch.float32`. Rows are processed sequentially on CPU.

cProfile recorded **101 C++ calls totaling 1.212 seconds** for
`(8192, 128)` float32 input and k=8 on an Intel Core i7-12700H, CPU PyTorch 2.8.0,
one PyTorch thread. This includes 100 loop calls and one correctness-check call;
startup/import time is outside the reported C++ entry. Reproduce from the root:

```bash
/tmp/moe-run/bin/python -m cProfile -o /tmp/moe-topk.prof tools/profile_topk.py --tokens 8192 --experts 128 --k 8 --iterations 100 --threads 1
/tmp/moe-run/bin/python -c 'import pstats; pstats.Stats("/tmp/moe-topk.prof").strip_dirs().sort_stats("tottime").print_stats("topk_experts")'
```

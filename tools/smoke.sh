#!/bin/bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
torch_cpu_index="https://download.pytorch.org/whl/cpu"
temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

"$python_bin" -m venv "$temp_dir/build" &&
PIP_EXTRA_INDEX_URL="$torch_cpu_index" \
    "$temp_dir/build/bin/python" -m pip wheel "$repo_dir" --no-deps -w "$repo_dir/dist" &&
ls "$repo_dir/dist" | grep 'moe_topk-.*-linux_.*\.whl$' &&
"$python_bin" -m venv "$temp_dir/run" &&
"$temp_dir/run/bin/python" -m pip install torch==2.8.0 --index-url "$torch_cpu_index" &&
"$temp_dir/run/bin/python" -m pip install --no-deps "$repo_dir"/dist/moe_topk-*.whl &&
(
    cd "$temp_dir" &&
    env -u PYTHONPATH "$temp_dir/run/bin/python" - <<'PY'
import torch

from moe_ops import top8_experts, topk_experts

scores = torch.tensor([0.2, 0.9, 0.9, -1.0], dtype=torch.float32)
assert topk_experts(scores, 2).tolist() == [1, 2]
assert top8_experts(scores).tolist() == [1, 2, 0, 3]
PY
) &&
echo OK

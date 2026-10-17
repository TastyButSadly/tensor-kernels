#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

torch::Tensor topk_experts(const torch::Tensor& scores, int64_t k) {
    TORCH_CHECK(scores.device().is_cpu(), "topk_experts: expected a CPU tensor");
    TORCH_CHECK(scores.layout() == c10::kStrided,
                "topk_experts: expected a strided tensor");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32,
                "topk_experts: expected torch.float32");
    TORCH_CHECK(scores.dim() >= 1,
                "topk_experts: expected at least one dimension");
    TORCH_CHECK(k >= 0, "topk_experts: k must be nonnegative");

    const auto num_experts = scores.size(-1);
    k = std::min(k, num_experts);
    auto output_shape = scores.sizes().vec();
    output_shape.back() = k;
    auto indices = torch::empty(output_shape, scores.options().dtype(torch::kInt64));
    if (indices.numel() == 0) {
        return indices;
    }

    // Materialize lazy negation before reading raw storage; then flatten rows.
    const auto contiguous_scores = scores.resolve_neg().contiguous();
    const auto* input = contiguous_scores.data_ptr<float>();
    auto* output = indices.data_ptr<int64_t>();
    const auto num_rows = scores.numel() / num_experts;
    std::vector<int64_t> order(num_experts);

    for (int64_t row = 0; row < num_rows; ++row) {
        const auto* values = input + row * num_experts;
        std::iota(order.begin(), order.end(), int64_t{0});
        const auto comes_before = [values](int64_t a, int64_t b) {
            const float left = values[a], right = values[b];
            // Break ties by index, including two NaNs; descending NaNs come first
            if (left == right || (std::isnan(left) && std::isnan(right))) {
                return a < b;
            }
            return std::isnan(left) || left > right;
        };
        std::partial_sort(order.begin(), order.begin() + k, order.end(), comes_before);
        std::copy_n(order.begin(), k, output + row * k);
    }
    return indices;
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "topk_experts", &topk_experts, pybind11::arg("scores"), pybind11::arg("k"),
        "Stable descending top-k indices for CPU float32 tensors; k must be nonnegative.");
}

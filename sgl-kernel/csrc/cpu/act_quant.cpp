#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/all.h>

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "common.h"

namespace {

constexpr float kFP8Max = 448.0f;
constexpr float kFP8Min = -448.0f;
constexpr float kMinScaleAmax = 1.0e-4f;

template <typename scalar_t, bool round_scale>
void act_quant_cpu_impl(
    at::Tensor& y,
    at::Tensor& scale,
    const at::Tensor& x,
    int64_t block_size,
    int64_t M,
    int64_t N,
    int64_t num_groups) {
  const scalar_t* __restrict__ x_ptr = x.const_data_ptr<scalar_t>();
  auto* __restrict__ y_ptr = y.mutable_data_ptr<at::Float8_e4m3fn>();
  float* __restrict__ scale_ptr = scale.mutable_data_ptr<float>();

  const int64_t total_blocks = M * num_groups;
  at::parallel_for(0, total_blocks, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    int64_t m = 0;
    int64_t group = 0;
    data_index_init(begin, m, M, group, num_groups);

    for (int64_t offset = begin; offset < end; ++offset) {
      const int64_t base = m * N + group * block_size;

      float amax = 0.0f;
      for (int64_t d = 0; d < block_size; ++d) {
        amax = std::max(amax, std::abs(static_cast<float>(x_ptr[base + d])));
      }
      amax = std::max(amax, kMinScaleAmax);

      float scale_value;
      if constexpr (round_scale) {
        scale_value = std::exp2(std::ceil(std::log2(amax / kFP8Max)));
      } else {
        scale_value = amax / kFP8Max;
      }
      scale_ptr[offset] = scale_value;

      const float inv_scale = 1.0f / scale_value;
      for (int64_t d = 0; d < block_size; ++d) {
        float value = static_cast<float>(x_ptr[base + d]) * inv_scale;
        value = std::min(std::max(value, kFP8Min), kFP8Max);
        y_ptr[base + d] = at::Float8_e4m3fn(value);
      }

      data_index_step(m, M, group, num_groups);
    }
  });
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> act_quant_cpu(
    at::Tensor& x, int64_t block_size, const std::optional<std::string>& scale_fmt) {
  RECORD_FUNCTION("sgl-kernel::act_quant_cpu", std::vector<c10::IValue>({x}));

  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int64_t N = x.size(-1);
  TORCH_CHECK(N % block_size == 0, "Last dimension size must be divisible by block_size");

  const int64_t M = x.numel() / N;
  const int64_t num_groups = N / block_size;

  at::Tensor y = at::empty_like(x, x.options().dtype(at::kFloat8_e4m3fn));

  std::vector<int64_t> scale_sizes = x.sizes().vec();
  scale_sizes.back() = num_groups;
  at::Tensor scale = at::empty(scale_sizes, x.options().dtype(at::kFloat));

  const bool round_scale = scale_fmt.has_value();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "act_quant_cpu", [&] {
    if (round_scale) {
      act_quant_cpu_impl<scalar_t, true>(y, scale, x, block_size, M, N, num_groups);
    } else {
      act_quant_cpu_impl<scalar_t, false>(y, scale, x, block_size, M, N, num_groups);
    }
  });

  return {y, scale};
}

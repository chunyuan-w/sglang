#include "common.h"
#include "vec.h"

#include <vector>

namespace {

template <typename scalar_t>
inline float load_as_float(const scalar_t* ptr, int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <typename scalar_t>
inline void store_from_float(scalar_t* ptr, int64_t idx, float value) {
  ptr[idx] = static_cast<scalar_t>(value);
}

inline bool is_power_of_two(int64_t x) {
  return x > 0 && (x & (x - 1)) == 0;
}

inline void fwht_float_inplace(float* __restrict__ row, int64_t n) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t fVecSize = fVec::size();

  for (int64_t h = 1; h < n; h <<= 1) {
    const int64_t block = h << 1;
    for (int64_t base = 0; base < n; base += block) {
      int64_t j = 0;
      for (; j + fVecSize <= h; j += fVecSize) {
        fVec a = fVec::loadu(row + base + j);
        fVec b = fVec::loadu(row + base + h + j);
        (a + b).store(row + base + j);
        (a - b).store(row + base + h + j);
      }
      for (; j < h; ++j) {
        const float a = row[base + j];
        const float b = row[base + h + j];
        row[base + j] = a + b;
        row[base + h + j] = a - b;
      }
    }
  }
}

template <typename scalar_t>
void hadamard_transform_cpu_impl(const at::Tensor& x, at::Tensor& out, double scale) {
  const int64_t n = x.size(-1);
  const int64_t rows = x.numel() / n;
  const scalar_t* __restrict__ x_ptr = x.const_data_ptr<scalar_t>();
  scalar_t* __restrict__ out_ptr = out.data_ptr<scalar_t>();

  at::parallel_for(0, rows, GRAIN_SIZE / n, [&](int64_t begin, int64_t end) {
    std::vector<float> buffer(n);
    for (int64_t row = begin; row < end; ++row) {
      const int64_t offset = row * n;
      for (int64_t i = 0; i < n; ++i) {
        buffer[i] = load_as_float(x_ptr, offset + i);
      }

      fwht_float_inplace(buffer.data(), n);

      for (int64_t i = 0; i < n; ++i) {
        store_from_float(out_ptr, offset + i, buffer[i] * static_cast<float>(scale));
      }
    }
  });
}

}  // namespace

at::Tensor hadamard_transform_cpu(at::Tensor& x, double scale) {
  RECORD_FUNCTION("sgl-kernel::hadamard_transform_cpu", std::vector<c10::IValue>({x}));
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  const int64_t n = x.size(-1);
  TORCH_CHECK(is_power_of_two(n), "Hidden size must be a power of 2 for Hadamard transform.");

  auto out = at::empty_like(x);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "hadamard_transform_cpu", [&] {
        hadamard_transform_cpu_impl<scalar_t>(x, out, scale);
      });
  return out;
}

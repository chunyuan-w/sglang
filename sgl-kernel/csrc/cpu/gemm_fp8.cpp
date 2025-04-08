#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {

}

at::Tensor fp8_scaled_mm_cpu(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2,
    std::vector<int64_t> block_size, std::optional<at::Tensor>& bias, 
    at::ScalarType out_dtype, bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::fp8_scaled_mm_cpu", std::vector<c10::IValue>({mat1, mat2, scales2, block_size, bias}));
  
  // TODO: fix me
  return mat1;

}
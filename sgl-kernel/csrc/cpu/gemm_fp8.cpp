#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {

} // anonymous namespace

at::Tensor fp8_scaled_mm_cpu(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2,
    std::vector<int64_t> block_size, std::optional<at::Tensor>& bias, 
    at::ScalarType out_dtype, bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::fp8_scaled_mm_cpu", std::vector<c10::IValue>({mat1, mat2, scales2, block_size, bias}));

  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);
  
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);
  TORCH_CHECK(scales2.scalar_type() == at::kFloat,
      "fp8_scaled_mm_cpu: expect scales2 to be float32.");
  
  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat2.size(1);
  
  
  CHECK_EQ(mat1.size(1), K);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  // TODO: check len of block_size == ndim of mat2

  // TODO: check out_dtype == mat1.dtype
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));
  
  // strides
  int64_t mat1_strideM = mat1.stride(0);
  int64_t out_strideM = out.stride(0);


  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  
  
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }  
    
  std::cout << "my w: " << mat2 << "\n";
  std::cout << "my packed_w: " << packed_w << "\n";

  // TODO: fix me
  return mat1;

}
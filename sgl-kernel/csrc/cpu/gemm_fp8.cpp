#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_add_stub(scalar_t* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) + fVec::loadu(bias + d);
    fVec data1 = fVec::loadu(input + d + fVec::size()) + fVec::loadu(bias + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + bias[d]);
  }
}

inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int N,
    int K,
    int ldb,
    int ldb_tmp,
    float scale) {
  // [K/2, N, 2]
  const int K2 = K >> 1;
  const int ldb2 = ldb; // ldb * 2 >> 1;
  const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(packed_B);
  const __m512 vd = _mm512_set1_ps(scale);

  for (int k = 0; k < K2; ++k) {
    for (int n = 0; n < N * 2; n += 32) {
        // Convert FP8 to BF16
        // TODO: should we add an API to convert to FP32 here for better perf?
        __m256i v_fp8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + k * ldb2 * 2 + n));
        __m512bh v_bf16 = cvt_e4m3_bf16_intrinsic_without_denorm(v_fp8);

        // Apply scale
        __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)v_bf16, 0));
        __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)v_bf16, 1));
        va0 = _mm512_mul_ps(va0, vd);
        va1 = _mm512_mul_ps(va1, vd);
        v_bf16 = _mm512_cvtne2ps_pbh(va1, va0);

        // Store to Btmp (no need for bit manipulation now)
        _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n, (__m512i)v_bf16);
    }
  }
}

template <typename scalar_t, typename packed_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A, const packed_t* __restrict__ B, scalar_t* __restrict__ C,
      const float* __restrict__ scale, int K, int lda, int ldb, int ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A, const at::BFloat16* __restrict__ B, at::BFloat16* __restrict__ C,
      const float* __restrict__ scale, int K, int lda, int ldb, int ldc) {

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    auto loadc = [&](auto i) {
      vc[i] = _mm512_set1_ps(0.f);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int K2 = K >> 1;
    const int lda2 = lda >> 1;
    const int ldb2 = ldb; // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b_ptr = reinterpret_cast<const float*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb[col] = (__m512bh)(_mm512_loadu_si512(b_ptr + k * ldb2 + col * 16));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      // for COLS = 1, 3 use 256bit store
      if constexpr (COLS % 2 == 0) {
        if constexpr (col % 2 == 0) {
          _mm512_storeu_si512(
              reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
              (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
        }
      } else {
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(C + row * ldc + col * 16),
            (__m256i)(_mm512_cvtneps_pbh(vc[i])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};

void print_16x16(const __m256i x) {
  at::BFloat16 a[16];
  _mm256_storeu_si256((__m256i *)a, x);

  for (int i = 0; i < 16; i++){
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}

void print_32x16(const __m512i x) {
  at::BFloat16 a[32];
  _mm512_storeu_si512((__m512i *)a, x);

  for (int i = 0; i < 32; i++){
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}

template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::Float8_e4m3fn, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A, const at::Float8_e4m3fn* __restrict__ B, at::BFloat16* __restrict__ C,
      const float* __restrict__ scale, int K, int lda, int ldb, int ldc) {

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    
    const __m512 vscale = _mm512_set1_ps(scale);
    const __m512i mask = _mm512_set1_epi32(0xFFFF);

    auto loadc = [&](auto i) {
      vc[i] = _mm512_set1_ps(0.f);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int K2 = K >> 1;
    const int lda2 = lda >> 1;
    const int ldb2 = ldb; // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + col * 16);
          if constexpr (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          __m512i idx0 = _mm512_cvtepu8_epi32(   _mm512_castsi512_si128(b8));
          __m512i idx1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 1));
          __m512i idx2 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 2));
          __m512i idx3 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 3));

          __m512i b16_0 = _mm512_i32gather_epi32(idx0, e4m3_to_16bit, 2);
          __m512i b16_1 = _mm512_i32gather_epi32(idx1, e4m3_to_16bit, 2);
          __m512i b16_2 = _mm512_i32gather_epi32(idx2, e4m3_to_16bit, 2);
          __m512i b16_3 = _mm512_i32gather_epi32(idx3, e4m3_to_16bit, 2);

          vb[col + 0] = (__m512bh)(_mm512_or_epi32(_mm512_slli_epi32(b16_2, 16), _mm512_and_epi32(b16_0, mask)));
          vb[col + 1] = (__m512bh)(_mm512_or_epi32(_mm512_slli_epi32(b16_3, 16), _mm512_and_epi32(b16_1, mask)));
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        __m512 vc0 = _mm512_mul_ps(vc[row * COLS + col + 0], vscale);
        __m512 vc1 = _mm512_mul_ps(vc[row * COLS + col + 1], vscale);
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(vc1, vc0)));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, packed_t, MB_SIZE, NB_SIZE>::apply(         \
        A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start, \
        scale, K, lda, ldb, ldc);

template <typename scalar_t, typename packed_t, bool has_bias>
struct brgemm {};

template <typename scalar_t, bool has_bias>
struct brgemm<scalar_t, scalar_t, has_bias> {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scales2,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      int64_t blocks_k_per_group) {
    UNUSED(scales2);

    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(
        M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};

template <bool has_bias>
struct brgemm<at::BFloat16, at::Float8_e4m3fn, has_bias> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scales2,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      int64_t blocks_k_per_group) {
    constexpr int BLOCK_N = block_size_n();

    // [BLOCK_K, BLOCK_N] -> [BLOCK_K / 2, BLOCK_N * 2]
    const int ldb_tmp = block_size_n();

    // accumulate across K per BLOCK_K
    for (int k = 0; k < K; k += BLOCK_K) {
      int kb_size = std::min(BLOCK_K, K - k);
      
      // TODO: check the index compute here
      int idx = (k / BLOCK_K) / blocks_k_per_group;
    //   std::cout << "scale idx before unpack_B: " << idx << " k:" << k << " block_size_K:" << block_size_K << " BLOCK_K:" << BLOCK_K << "\n";
      unpack_B(Btmp, B + k * ldb, N, kb_size, ldb, ldb_tmp, scales2[idx]);

      const bool add_C = (k != 0);
      at::native::cpublas::brgemm(
          M, N, kb_size, lda, ldb_tmp, BLOCK_N, add_C, A + k, Btmp, Ctmp);
    }

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }      
    }
  }
};

template <typename scalar_t, typename packed_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const packed_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t blocks_k_per_group) {

  if (brg) {
    brgemm<scalar_t, packed_t, has_bias>::apply(
        A, B, C, Btmp, Ctmp, bias, scale, M, N, K, lda, ldb, ldc, blocks_k_per_group);
    return;
  }

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch(mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 0x14: LAUNCH_TINYGEMM_KERNEL_NN(1, 64); break;
        // mb_size = 2
        case 0x22: LAUNCH_TINYGEMM_KERNEL_NN(2, 32); break;
        case 0x24: LAUNCH_TINYGEMM_KERNEL_NN(2, 64); break;
        // mb_size = 3
        case 0x32: LAUNCH_TINYGEMM_KERNEL_NN(3, 32); break;
        case 0x34: LAUNCH_TINYGEMM_KERNEL_NN(3, 64); break;
        // mb_size = 4
        case 0x42: LAUNCH_TINYGEMM_KERNEL_NN(4, 32); break;
        case 0x44: LAUNCH_TINYGEMM_KERNEL_NN(4, 64); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }

}

template <typename scalar_t, typename packed_t>
void fp8_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const packed_t* __restrict__ mat2,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t block_size_N,
    int64_t block_size_K) {

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

    // std::cout << "BLOCK_M: " << BLOCK_M << " M: " << M << "\n";
    // std::cout << "BLOCK_N: " << BLOCK_N << " N: " << N << "\n";
    // std::cout << "MB: " << MB << " NB: " << NB << "\n";

  // TODO: should we use div_up?
  const int64_t scale_size_N = div_up(N, block_size_N);
  const int64_t scale_size_K = div_up(K, block_size_K);

  const int64_t blocks_n_per_group = block_size_N / BLOCK_N;
  const int64_t blocks_k_per_group = block_size_K / BLOCK_K;

  // use avx512-bf16 when a) M is small; b) dtype is bfloat16, otherwise use amx
  // TODO: support use_brgemm = false;
  const bool use_brgemm = (M > 4) || (!std::is_same_v<scalar_t, at::BFloat16>);
  // const bool use_brgemm = true;

  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t mb{0}, nb{0};
      data_index_init(begin, mb, MB, nb, NB);

      // for brgemm, use float32 for accumulate
      alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
      // for brgemm when mat2 is float8_e4m3
      alignas(64) scalar_t Btmp[BLOCK_N * BLOCK_K];
      
      for (int64_t i = begin; i < end; ++i) {
        UNUSED(i);
        // TODO: check the index compute here
        const float* scale_ptr = scales2 + (nb / blocks_n_per_group) * scale_size_K;
        // std::cout << "nb: " << nb << " block_size_N: " << block_size_N << " scale_size_K:" << scale_size_K << "\n";
        
        // printf("scale ptr idx: %d\n", nb / (block_size_N / BLOCK_N) * scale_size_K);

        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, packed_t, has_bias>(
            /*   A */ mat1 + mb_start * K,
            /*   B */ mat2 + nb_start * K /* nb * BLOCK_N * K */,
            /*   C */ out + mb_start * N + nb_start,
            /* Btmp*/ Btmp,
            /* Ctmp*/ Ctmp,
            /*scale*/ scale_ptr,
            /* bias*/ bias + nb_start,
            /*   M */ mb_size,
            /*   N */ nb_size,
            /*   K */ K,
            /* lda */ K,
            /* ldb */ nb_size,
            /* ldc */ N,
            /* brg */ use_brgemm,
            /* blocks_k_per_group */ blocks_k_per_group);

        // move to the next index
        data_index_step(mb, MB, nb, NB);
      }

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });

}

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

  TORCH_CHECK(block_size.size() == 2,
      "fp8_scaled_mm_cpu: expect block_size.size() to be 2.");
  
  // TODO: K is [0] or [1]??
  int64_t block_size_N = block_size[0];
  int64_t block_size_K = block_size[1];

  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(block_size_N >= BLOCK_N, "expect block_size_N >= BLOCK_N");
  TORCH_CHECK(block_size_K >= BLOCK_K, "expect block_size_K >= BLOCK_K");
  // TODO: check numel of scales
  // CHECK_EQ(scales2.numel(), N);

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf,
      "fp8_scaled_mm_cpu: expect A to be bfloat16 or half.");
  TORCH_CHECK(st == out_dtype,
      "fp8_scaled_mm_cpu: expect A has same dtype with out_dtype.");
  TORCH_CHECK(mat2.scalar_type() == at::kFloat8_e4m3fn,
      "fp8_scaled_mm_cpu: expect mat2 to be fp8_e4m3.");
  TORCH_CHECK(scales2.scalar_type() == at::kFloat,
      "fp8_scaled_mm_cpu: expect scales to be float32.");
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));
  
  // TODO: do we need to support strides?
  // strides
//   int64_t mat1_strideM = mat1.stride(0);
//   int64_t out_strideM = out.stride(0);

  // TODO: seems the current code already supports it?
//   TORCH_CHECK(N % block_size_N == 0, "unsupported block_size_N");
//   TORCH_CHECK(K % block_size_K == 0, "unsupported block_size_K");

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }  
    
  CPU_DISPATCH_PACKED_FLOAT_TYPES(out_dtype, packed_w.scalar_type(), "fp8_scaled_mm_kernel_impl", [&] {
    fp8_scaled_mm_kernel_impl<scalar_t, packed_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<packed_t>(),
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K,
        block_size_N,
        block_size_K);
  });

  return out;

}
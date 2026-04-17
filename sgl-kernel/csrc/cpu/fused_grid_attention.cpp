/*****************************************************************************************
 * Fused grid self-attention kernel for sglang CPU backend.
 *
 * Mirrors the fusion strategy used by af3_kernels' TPP `_attention_impl`:
 *   - QKV and gating linear projections are computed inside the attention loop,
 *     so Q / K / V / gating activations are never fully materialized.
 *   - Attention uses the flash-attention softmax primitive from flash_attn.h
 *     (streaming softmax over BLOCK_N key chunks).
 *   - Per-head attention output is multiplied by sigmoid(gate) and stored
 *     interleaved across heads into a [B, N, H*K] buffer.
 *   - A second pass performs the final output projection as a standard
 *     weight_packed_linear gemm, matching how TPP splits its two loops.
 *
 * Weight layout assumption:
 *   Q/K/V/gating weights are packed via convert_weight_packed on the full
 *   [D, D] weight, where D = H * K.  With BLOCK_N (packed block size) == K,
 *   block h of the packed weight corresponds exactly to head h's slice.
 *   This holds for the AF3 grid-self-attention configuration (D=128, H=4,
 *   K=32).  Other K values trip a runtime check.
 ****************************************************************************************/
#include "common.h"
#include "flash_attn.h"  // also transitively includes vec_pack.h (no header guard)
#include "gemm.h"

namespace {

template <typename scalar_t>
inline void cast_copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec a0 = fVec::loadu(input + d);
    fVec a1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(a0, a1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void sigmoid_mul_scale_bf16gate_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ gate_bf,
    const float* __restrict__ v_prime,
    float inv_s,
    int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec one = fVec(1.f);
  const fVec vscale = fVec(inv_s);
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec g0, g1;
    std::tie(g0, g1) = load_float_vec2<scalar_t>(gate_bf + d);
    fVec s0 = one / (one + g0.neg().exp_u20());
    fVec s1 = one / (one + g1.neg().exp_u20());
    fVec v0 = fVec::loadu(v_prime + d) * vscale;
    fVec v1 = fVec::loadu(v_prime + d + fVec::size()) * vscale;
    bVec out_vec = convert_from_float_ext<scalar_t>(s0 * v0, s1 * v1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    float g = static_cast<float>(gate_bf[d]);
    float sg = 1.f / (1.f + std::exp(-g));
    out[d] = static_cast<scalar_t>(sg * v_prime[d] * inv_s);
  }
}

template <typename scalar_t>
inline void sigmoid_mul_scale_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ gate_f,
    const float* __restrict__ v_prime,
    float inv_s,
    int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec one = fVec(1.f);
  const fVec vscale = fVec(inv_s);
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec g0 = fVec::loadu(gate_f + d);
    fVec g1 = fVec::loadu(gate_f + d + fVec::size());
    fVec s0 = one / (one + g0.neg().exp_u20());
    fVec s1 = one / (one + g1.neg().exp_u20());
    fVec v0 = fVec::loadu(v_prime + d) * vscale;
    fVec v1 = fVec::loadu(v_prime + d + fVec::size()) * vscale;
    bVec out_vec = convert_from_float_ext<scalar_t>(s0 * v0, s1 * v1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    float g = gate_f[d];
    float sg = 1.f / (1.f + std::exp(-g));
    out[d] = static_cast<scalar_t>(sg * v_prime[d] * inv_s);
  }
}

// Core fused kernel.  Produces the gated attention output in `attn_out`
// ([B, N, H*K]); the caller runs the final output projection separately.
template <typename scalar_t, int BLOCK_M, int BLOCK_N>
void fused_attention_stage_impl(
    scalar_t* __restrict__ attn_out,     // [B, N, H*K]
    const scalar_t* __restrict__ pair,   // [B, N, H*K]
    const scalar_t* __restrict__ bias,   // [H, N, N]
    const scalar_t* __restrict__ q_w,    // packed: [D/2, D, 2]  (viewed as [H, D/2, K, 2])
    const scalar_t* __restrict__ k_w,    // packed: [D/2, D, 2]
    const scalar_t* __restrict__ v_w,    // packed: [D/2, D, 2]
    const scalar_t* __restrict__ g_w,    // packed: [D/2, D, 2]
    int B,
    int N,
    int H,
    int K,
    int bias_strideH,
    int bias_strideM,
    float sm_scale,
    void* __restrict__ buffer,
    int buffer_size_per_thread) {
  const int D = H * K;
  // Head h's packed weight block lives at element offset h * BLOCK_N * D in
  // the [NB, K_in/2, BLOCK_N, 2] layout when BLOCK_N == K.
  TORCH_CHECK(
      K == block_size_n(),
      "fused_grid_attention requires per-head dim K == BLOCK_N (=",
      block_size_n(),
      "), got K=",
      K);

  parallel_for(B * H, [&](int begin, int end) {
    int bs{0}, head_id{0};
    data_index_init(begin, bs, B, head_id, H);

    int tid = get_thread_num();
    char* base = reinterpret_cast<char*>(buffer) + static_cast<size_t>(tid) * buffer_size_per_thread;

    auto bump = [&](size_t nbytes) {
      void* p = base;
      base += (nbytes + 63) & ~size_t{63};
      return p;
    };

    // Per-head full projections (kept across MB iterations).
    scalar_t* K_full = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));
    scalar_t* V_full = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));

    // Flash-attention scratch: s_i is also reinterpreted as s_delta for the
    // second gemm of the streaming softmax, matching flash_attn.cpp.
    float* s_i = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * BLOCK_N));
    float* v_prime = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * K));

    // Projection scratch.
    scalar_t* Btmp = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_N * K));
    float* Ctmp = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * K));
    scalar_t* Q_block = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * K));

    scalar_t* s_delta = reinterpret_cast<scalar_t*>(s_i);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int idx = begin; idx < end; ++idx) {
      const scalar_t* pair_bs = pair + static_cast<size_t>(bs) * N * D;
      scalar_t* attn_bs = attn_out + static_cast<size_t>(bs) * N * D;

      // Per-head slice of the packed weight: block `head_id` in the NB dim.
      const scalar_t* q_w_h = q_w + static_cast<size_t>(head_id) * K * D;
      const scalar_t* k_w_h = k_w + static_cast<size_t>(head_id) * K * D;
      const scalar_t* v_w_h = v_w + static_cast<size_t>(head_id) * K * D;
      const scalar_t* g_w_h = g_w + static_cast<size_t>(head_id) * K * D;

      // ----- K projection for the entire sequence -----
      for (int m = 0; m < N; m += BLOCK_M) {
        int m_sz = std::min(BLOCK_M, N - m);
        at::native::cpublas::brgemm(
            /* M */ m_sz, /* N */ K, /* K */ D,
            /* lda */ D, /* ldb */ K, /* ldc */ K,
            /* add_C */ false,
            pair_bs + static_cast<size_t>(m) * D,
            k_w_h,
            Ctmp);
        for (int r = 0; r < m_sz; ++r) {
          cast_copy_stub<scalar_t>(K_full + static_cast<size_t>(m + r) * K, Ctmp + r * K, K);
        }
      }

      // ----- V projection for the entire sequence -----
      for (int m = 0; m < N; m += BLOCK_M) {
        int m_sz = std::min(BLOCK_M, N - m);
        at::native::cpublas::brgemm(
            /* M */ m_sz, /* N */ K, /* K */ D,
            /* lda */ D, /* ldb */ K, /* ldc */ K,
            /* add_C */ false,
            pair_bs + static_cast<size_t>(m) * D,
            v_w_h,
            Ctmp);
        for (int r = 0; r < m_sz; ++r) {
          cast_copy_stub<scalar_t>(V_full + static_cast<size_t>(m + r) * K, Ctmp + r * K, K);
        }
      }

      const scalar_t* bias_h = bias + static_cast<size_t>(head_id) * bias_strideH;

      // ----- Iterate over query blocks -----
      for (int m = 0; m < N; m += BLOCK_M) {
        int m_size = std::min(BLOCK_M, N - m);

        // Q projection for this block.
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ K, /* K */ D,
            /* lda */ D, /* ldb */ K, /* ldc */ K,
            /* add_C */ false,
            pair_bs + static_cast<size_t>(m) * D,
            q_w_h,
            Ctmp);
        for (int r = 0; r < m_size; ++r) {
          cast_copy_stub<scalar_t>(Q_block + r * K, Ctmp + r * K, K);
        }

        // Init flash-attention running statistics.
        fill_stub(v_prime, 0.f, m_size * K);
        fill_stub(s_prime, 0.f, m_size);
        fill_stub(m_prime, -std::numeric_limits<float>::infinity(), m_size);

        // Streaming softmax across K/V chunks.
        for (int n = 0; n < N; n += BLOCK_N) {
          int n_size = std::min(BLOCK_N, N - n);
          const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

          // Pack K chunk for Q @ K^T (expects layout [K/2, n_size, 2]).
          pack_vnni<scalar_t>(
              /* dst */ Btmp,
              /* src */ K_full + static_cast<size_t>(n) * K,
              /* N */ n_size,
              /* K */ K,
              /* ld_src */ K,
              /* ld_dst */ BLOCK_N);

          at::native::cpublas::brgemm(
              /* M */ m_size, /* N */ n_size, /* K */ K,
              /* lda */ K, /* ldb */ BLOCK_N, /* ldc */ BLOCK_N,
              /* add_C */ false,
              Q_block,
              Btmp,
              s_i);

          const scalar_t* bias_ptr = bias_h + static_cast<size_t>(m) * bias_strideM + n;
          flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
              s_i,
              s_delta,
              v_prime,
              s_prime,
              m_prime,
              m_size,
              n_size,
              padded_n_size,
              K,
              sm_scale,
              bias_ptr,
              bias_strideM);

          // Pack V chunk for s_delta @ V (expects layout [K/2, K_v, 2]).
          pack_vnni2<scalar_t>(
              /* dst */ Btmp,
              /* src */ V_full + static_cast<size_t>(n) * K,
              /* K */ n_size,
              /* N */ K,
              /* ld_src */ K,
              /* ld_dst */ K);

          at::native::cpublas::brgemm(
              /* M */ m_size, /* N */ K, /* K */ padded_n_size,
              /* lda */ BLOCK_N, /* ldb */ K, /* ldc */ K,
              /* add_C */ true,
              s_delta,
              Btmp,
              v_prime);
        }

        // Gating projection for this block.
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ K, /* K */ D,
            /* lda */ D, /* ldb */ K, /* ldc */ K,
            /* add_C */ false,
            pair_bs + static_cast<size_t>(m) * D,
            g_w_h,
            Ctmp);

        // attn_bs[m:m+m_size, head_id*K:(head_id+1)*K] = sigmoid(gate) * (v_prime / s_prime)
        for (int r = 0; r < m_size; ++r) {
          float inv_s = 1.f / s_prime[r];
          scalar_t* dst = attn_bs + static_cast<size_t>(m + r) * D + static_cast<size_t>(head_id) * K;
          sigmoid_mul_scale_stub<scalar_t>(dst, Ctmp + r * K, v_prime + r * K, inv_s, K);
        }
      }

      data_index_step(bs, B, head_id, H);
    }
    at::native::cpublas::brgemm_release();
  });
}

// v2 core: consumes an already-projected QKVG buffer [B, N, 4*D] produced by a
// single concat weight_packed_linear (stage A), and emits the gated attention
// output [B, N, D].  The final out_proj is run outside as stage C.  This keeps
// QKV+gating in a large-M GEMM (pair streamed once per projection-group) and
// only fuses the attention + sigmoid-gate tail that actually benefits.
template <typename scalar_t, int BLOCK_M, int BLOCK_N>
void fused_attention_core_impl(
    scalar_t* __restrict__ gated_attn,   // [B, N, H*K]
    const scalar_t* __restrict__ qkvg,   // [B, N, 4*H*K] (Q|K|V|G concat on last dim)
    const scalar_t* __restrict__ bias,   // [H, N, N]
    int B,
    int N,
    int H,
    int K,
    int bias_strideH,
    int bias_strideM,
    float sm_scale,
    void* __restrict__ buffer,
    int buffer_size_per_thread) {
  const int D = H * K;
  const int QKVG_STRIDE = 4 * D;  // row stride within qkvg (element count)

  parallel_for(B * H, [&](int begin, int end) {
    int bs{0}, head_id{0};
    data_index_init(begin, bs, B, head_id, H);

    int tid = get_thread_num();
    char* base = reinterpret_cast<char*>(buffer) + static_cast<size_t>(tid) * buffer_size_per_thread;

    auto bump = [&](size_t nbytes) {
      void* p = base;
      base += (nbytes + 63) & ~size_t{63};
      return p;
    };

    // Contiguous per-head K and V scratch (streamed through all Q blocks).
    scalar_t* K_contig = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));
    scalar_t* V_contig = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));

    // Flash-attention scratch (s_i aliased as s_delta for the second gemm).
    float* s_i = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * BLOCK_N));
    float* v_prime = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * K));
    scalar_t* Btmp = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_N * K));
    scalar_t* Q_block = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * K));

    scalar_t* s_delta = reinterpret_cast<scalar_t*>(s_i);
    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int idx = begin; idx < end; ++idx) {
      const scalar_t* qkvg_bs = qkvg + static_cast<size_t>(bs) * N * QKVG_STRIDE;
      scalar_t* attn_bs = gated_attn + static_cast<size_t>(bs) * N * D;

      // In qkvg, each row is [Q(D) | K(D) | V(D) | G(D)].
      // Per-head slice offsets within a row:
      const size_t Q_off = static_cast<size_t>(head_id) * K;
      const size_t K_off = static_cast<size_t>(D) + static_cast<size_t>(head_id) * K;
      const size_t V_off = static_cast<size_t>(2 * D) + static_cast<size_t>(head_id) * K;
      const size_t G_off = static_cast<size_t>(3 * D) + static_cast<size_t>(head_id) * K;

      // Gather this head's K/V into contiguous scratch once per (bs, head).
      // Streaming from the strided QKVG buffer inside the flash-attn loop was
      // tried but hurt perf badly because each 32-elem K row leaves 480 elems
      // of the cache line unused.
      for (int n = 0; n < N; ++n) {
        const scalar_t* src_k = qkvg_bs + static_cast<size_t>(n) * QKVG_STRIDE + K_off;
        const scalar_t* src_v = qkvg_bs + static_cast<size_t>(n) * QKVG_STRIDE + V_off;
        std::memcpy(K_contig + static_cast<size_t>(n) * K, src_k, sizeof(scalar_t) * K);
        std::memcpy(V_contig + static_cast<size_t>(n) * K, src_v, sizeof(scalar_t) * K);
      }

      const scalar_t* bias_h = bias + static_cast<size_t>(head_id) * bias_strideH;

      for (int m = 0; m < N; m += BLOCK_M) {
        int m_size = std::min(BLOCK_M, N - m);

        // Gather the current Q block to a contig buffer for brgemm.
        for (int r = 0; r < m_size; ++r) {
          const scalar_t* src_q = qkvg_bs + static_cast<size_t>(m + r) * QKVG_STRIDE + Q_off;
          std::memcpy(Q_block + static_cast<size_t>(r) * K, src_q, sizeof(scalar_t) * K);
        }

        fill_stub(v_prime, 0.f, m_size * K);
        fill_stub(s_prime, 0.f, m_size);
        fill_stub(m_prime, -std::numeric_limits<float>::infinity(), m_size);

        for (int n = 0; n < N; n += BLOCK_N) {
          int n_size = std::min(BLOCK_N, N - n);
          const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

          pack_vnni<scalar_t>(
              /* dst */ Btmp,
              /* src */ K_contig + static_cast<size_t>(n) * K,
              /* N */ n_size,
              /* K */ K,
              /* ld_src */ K,
              /* ld_dst */ BLOCK_N);

          at::native::cpublas::brgemm(
              /* M */ m_size, /* N */ n_size, /* K */ K,
              /* lda */ K, /* ldb */ BLOCK_N, /* ldc */ BLOCK_N,
              /* add_C */ false,
              Q_block,
              Btmp,
              s_i);

          const scalar_t* bias_ptr = bias_h + static_cast<size_t>(m) * bias_strideM + n;
          flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
              s_i,
              s_delta,
              v_prime,
              s_prime,
              m_prime,
              m_size,
              n_size,
              padded_n_size,
              K,
              sm_scale,
              bias_ptr,
              bias_strideM);

          pack_vnni2<scalar_t>(
              /* dst */ Btmp,
              /* src */ V_contig + static_cast<size_t>(n) * K,
              /* K */ n_size,
              /* N */ K,
              /* ld_src */ K,
              /* ld_dst */ K);

          at::native::cpublas::brgemm(
              /* M */ m_size, /* N */ K, /* K */ padded_n_size,
              /* lda */ BLOCK_N, /* ldb */ K, /* ldc */ K,
              /* add_C */ true,
              s_delta,
              Btmp,
              v_prime);
        }

        // gated_attn[bs, m:m+m_size, head*K:(head+1)*K] = sigmoid(G) * (v_prime / s_prime)
        for (int r = 0; r < m_size; ++r) {
          float inv_s = 1.f / s_prime[r];
          scalar_t* dst = attn_bs + static_cast<size_t>(m + r) * D + static_cast<size_t>(head_id) * K;
          const scalar_t* g_row = qkvg_bs + static_cast<size_t>(m + r) * QKVG_STRIDE + G_off;
          sigmoid_mul_scale_bf16gate_stub<scalar_t>(dst, g_row, v_prime + r * K, inv_s, K);
        }
      }

      data_index_step(bs, B, head_id, H);
    }
    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

// Declared in gemm.cpp; reused for the final output projection.
at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_vnni);

// Full fused op: Q/K/V + flash-attention + gating + sigmoid-mul + output
// projection.  All Q/K/V/gating/output weights must already be packed via
// convert_weight_packed.  Per-head K must equal BLOCK_N (=32) because we
// slice the packed weight by its OC block index.
//
//   pair          : [B, N, H*K]
//   bias          : [H, N, N]
//   q/k/v/g/o_w   : packed [D/2, D, 2]  (D = H * K, o_w same shape)
//   returns       : [B, N, H*K]
at::Tensor fused_grid_attention(
    at::Tensor& pair,
    at::Tensor& bias,
    at::Tensor& q_weight,
    at::Tensor& k_weight,
    at::Tensor& v_weight,
    at::Tensor& gating_weight,
    at::Tensor& output_weight,
    int64_t num_heads,
    bool is_vnni) {
  RECORD_FUNCTION(
      "sgl_kernel::fused_grid_attention",
      std::vector<c10::IValue>({pair, bias, q_weight, k_weight, v_weight, gating_weight, output_weight, num_heads}));

  CHECK_INPUT(pair);
  CHECK_DIM(3, pair);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(bias);
  CHECK_DIM(3, bias);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_INPUT(v_weight);
  CHECK_INPUT(gating_weight);
  CHECK_INPUT(output_weight);
  TORCH_CHECK(is_vnni, "fused_grid_attention currently requires pre-packed weights (is_vnni=True).");

  const int B = pair.size(0);
  const int N = pair.size(1);
  const int D = pair.size(2);
  const int H = static_cast<int>(num_heads);
  TORCH_CHECK(D % H == 0, "pair feature dim ", D, " not divisible by num_heads ", H);
  const int K = D / H;

  CHECK_EQ(bias.size(0), H);
  CHECK_EQ(bias.size(1), N);
  CHECK_EQ(bias.size(2), N);

  const int bias_strideH = bias.stride(0);
  const int bias_strideM = bias.stride(1);
  const double sm_scale = 1.0 / std::sqrt(static_cast<double>(K));

  auto attn_out = at::empty({B, N, D}, pair.options());

  // Block sizes: chosen small enough that per-thread scratch fits in L2 for
  // typical AF3 shapes (N up to ~4k, K=32).
  constexpr int BLOCK_M = 32;
  constexpr int BLOCK_N = 128;
  static_assert(BLOCK_M <= BLOCK_N, "flash_attn_softmax assumes BLOCK_M <= BLOCK_N.");

  const int num_threads = at::get_num_threads();
  const int per_thread_bytes =
      /* K_full  */ sizeof(uint16_t) * N * K +
      /* V_full  */ sizeof(uint16_t) * N * K +
      /* s_i     */ sizeof(float) * BLOCK_M * BLOCK_N +
      /* v_prime */ sizeof(float) * BLOCK_M * K +
      /* Btmp    */ sizeof(uint16_t) * BLOCK_N * K +
      /* Ctmp    */ sizeof(float) * BLOCK_M * K +
      /* Q_block */ sizeof(uint16_t) * BLOCK_M * K +
      /* alignment padding */ 64 * 8;
  auto buffer = at::empty({num_threads, per_thread_bytes}, pair.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(pair.scalar_type(), "fused_grid_attention", [&] {
    fused_attention_stage_impl<scalar_t, BLOCK_M, BLOCK_N>(
        attn_out.data_ptr<scalar_t>(),
        pair.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        q_weight.data_ptr<scalar_t>(),
        k_weight.data_ptr<scalar_t>(),
        v_weight.data_ptr<scalar_t>(),
        gating_weight.data_ptr<scalar_t>(),
        B,
        N,
        H,
        K,
        bias_strideH,
        bias_strideM,
        static_cast<float>(sm_scale),
        buffer.data_ptr(),
        per_thread_bytes);
  });

  // ----- Final output projection (2D view to match weight_packed_linear) -----
  auto attn_out_2d = attn_out.view({static_cast<int64_t>(B) * N, D});
  auto out_2d = weight_packed_linear(attn_out_2d, output_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  return out_2d.view({B, N, D});
}

// v2: split stage-A QKVG concat GEMM + fused attention-tail core + stage-C
// out_proj.  This trades the v1 monolithic fusion for better pair-read BW.
//
//   pair          : [B, N, D]              (D = H * K)
//   bias          : [H, N, N]
//   qkvg_weight   : packed [4*D, D] (pack of torch.cat([Q, K, V, G], dim=0))
//   output_weight : packed [D, D]
//   returns       : [B, N, D]
at::Tensor fused_grid_attention_v2(
    at::Tensor& pair,
    at::Tensor& bias,
    at::Tensor& qkvg_weight,
    at::Tensor& output_weight,
    int64_t num_heads,
    bool is_vnni) {
  RECORD_FUNCTION(
      "sgl_kernel::fused_grid_attention_v2",
      std::vector<c10::IValue>({pair, bias, qkvg_weight, output_weight, num_heads}));

  CHECK_INPUT(pair);
  CHECK_DIM(3, pair);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(bias);
  CHECK_DIM(3, bias);
  CHECK_INPUT(qkvg_weight);
  CHECK_INPUT(output_weight);
  TORCH_CHECK(is_vnni, "fused_grid_attention_v2 currently requires pre-packed weights (is_vnni=True).");

  const int B = pair.size(0);
  const int N = pair.size(1);
  const int D = pair.size(2);
  const int H = static_cast<int>(num_heads);
  TORCH_CHECK(D % H == 0, "pair feature dim ", D, " not divisible by num_heads ", H);
  const int K = D / H;

  CHECK_EQ(bias.size(0), H);
  CHECK_EQ(bias.size(1), N);
  CHECK_EQ(bias.size(2), N);

  const int bias_strideH = bias.stride(0);
  const int bias_strideM = bias.stride(1);
  const double sm_scale = 1.0 / std::sqrt(static_cast<double>(K));

  // ----- Stage A: one concat QKVG GEMM, pair streamed once -----
  auto pair_2d = pair.view({static_cast<int64_t>(B) * N, D});
  auto qkvg_2d = weight_packed_linear(pair_2d, qkvg_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  TORCH_CHECK(
      qkvg_2d.size(1) == 4 * D,
      "qkvg_weight must map D -> 4*D; got output dim ",
      qkvg_2d.size(1),
      ", expected ",
      4 * D);
  auto qkvg = qkvg_2d.view({B, N, 4 * D});

  // ----- Stage B: fused flash-attn + sigmoid(G) * attn -----
  auto gated_attn = at::empty({B, N, D}, pair.options());

  constexpr int BLOCK_M = 32;
  constexpr int BLOCK_N = 128;
  static_assert(BLOCK_M <= BLOCK_N, "flash_attn_softmax assumes BLOCK_M <= BLOCK_N.");

  const int num_threads = at::get_num_threads();
  const int per_thread_bytes =
      /* K_contig */ sizeof(uint16_t) * N * K +
      /* V_contig */ sizeof(uint16_t) * N * K +
      /* s_i      */ sizeof(float) * BLOCK_M * BLOCK_N +
      /* v_prime  */ sizeof(float) * BLOCK_M * K +
      /* Btmp     */ sizeof(uint16_t) * BLOCK_N * K +
      /* Q_block  */ sizeof(uint16_t) * BLOCK_M * K +
      /* alignment padding */ 64 * 8;
  auto buffer = at::empty({num_threads, per_thread_bytes}, pair.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(pair.scalar_type(), "fused_grid_attention_v2", [&] {
    fused_attention_core_impl<scalar_t, BLOCK_M, BLOCK_N>(
        gated_attn.data_ptr<scalar_t>(),
        qkvg.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        B,
        N,
        H,
        K,
        bias_strideH,
        bias_strideM,
        static_cast<float>(sm_scale),
        buffer.data_ptr(),
        per_thread_bytes);
  });

  // ----- Stage C: final output projection -----
  auto gated_attn_2d = gated_attn.view({static_cast<int64_t>(B) * N, D});
  auto out_2d = weight_packed_linear(gated_attn_2d, output_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  return out_2d.view({B, N, D});
}

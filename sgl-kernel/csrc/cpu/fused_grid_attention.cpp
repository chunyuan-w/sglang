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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Gate per-call stage timing for fused_grid_attention_v2 behind SGL_V2_PROFILE=1.
// Reading the env once at first use keeps the fast path free of getenv calls.
inline bool v2_profile_enabled() {
  static const bool enabled = []{
    const char* s = std::getenv("SGL_V2_PROFILE");
    return s && s[0] != '\0' && std::strcmp(s, "0") != 0;
  }();
  return enabled;
}

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

// Per-row softmax for the non-flash core: scale + bias, max, exp+sum, normalize,
// cast to bf16, zero the [N, N_pad) tail so it contributes 0 to attn @ V.
//
// row_fp32 is rewritten in place (scratch for the three passes).
template <typename scalar_t>
inline void softmax_row_to_bf16(
    float* __restrict__ row_fp32,
    scalar_t* __restrict__ row_out,
    const scalar_t* __restrict__ bias_row,
    int N,
    int N_pad,
    float sm_scale) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kBVec = bVec::size();  // 32 for bf16
  constexpr int kFVec = fVec::size();  // 16 for float

  const fVec vscale(sm_scale);

  // Pass 1: row = row * scale + bias; track max.
  fVec vmax(-std::numeric_limits<float>::infinity());
  int n = 0;
  for (; n <= N - kBVec; n += kBVec) {
    fVec l0 = fVec::loadu(row_fp32 + n);
    fVec l1 = fVec::loadu(row_fp32 + n + kFVec);
    fVec b0, b1;
    std::tie(b0, b1) = load_float_vec2<scalar_t>(bias_row + n);
    l0 = l0 * vscale + b0;
    l1 = l1 * vscale + b1;
    l0.store(row_fp32 + n);
    l1.store(row_fp32 + n + kFVec);
    vmax = at::vec::maximum(vmax, at::vec::maximum(l0, l1));
  }
  float max_val = at::vec::vec_reduce_all<float>(
      [](fVec a, fVec b) { return at::vec::maximum(a, b); }, vmax);
  for (; n < N; ++n) {
    float v = row_fp32[n] * sm_scale + static_cast<float>(bias_row[n]);
    row_fp32[n] = v;
    if (v > max_val) max_val = v;
  }

  // Pass 2: row = exp(row - max); sum.
  const fVec vmax2(max_val);
  fVec vsum(0.f);
  n = 0;
  for (; n <= N - kBVec; n += kBVec) {
    fVec e0 = (fVec::loadu(row_fp32 + n) - vmax2).exp_u20();
    fVec e1 = (fVec::loadu(row_fp32 + n + kFVec) - vmax2).exp_u20();
    e0.store(row_fp32 + n);
    e1.store(row_fp32 + n + kFVec);
    vsum = vsum + e0 + e1;
  }
  float sum_val = at::vec::vec_reduce_all<float>(
      [](fVec a, fVec b) { return a + b; }, vsum);
  for (; n < N; ++n) {
    float e = std::exp(row_fp32[n] - max_val);
    row_fp32[n] = e;
    sum_val += e;
  }

  // Pass 3: row_out = bf16(row / sum); zero the [N, N_pad) tail.
  const float inv_sum = 1.f / sum_val;
  const fVec vinv(inv_sum);
  n = 0;
  for (; n <= N - kBVec; n += kBVec) {
    fVec l0 = fVec::loadu(row_fp32 + n) * vinv;
    fVec l1 = fVec::loadu(row_fp32 + n + kFVec) * vinv;
    bVec bout = convert_from_float_ext<scalar_t>(l0, l1);
    bout.store(row_out + n);
  }
  for (; n < N; ++n) {
    row_out[n] = static_cast<scalar_t>(row_fp32[n] * inv_sum);
  }
  for (int p = N; p < N_pad; ++p) {
    row_out[p] = scalar_t(0);
  }
}

// v2 core: consumes an already-projected QKVG buffer [B, N, 4*D] produced by a
// single concat weight_packed_linear (stage A), and emits the gated attention
// output [B, N, D].  The final out_proj is run outside as stage C.  This keeps
// QKV+gating in a large-M GEMM (pair streamed once per projection-group) and
// only fuses the attention + sigmoid-gate tail that actually benefits.
//
// Uses a non-flash softmax: per (bs, h) we pack K and V to VNNI layout once,
// then per M-block do two brgemms (Q @ K^T over the full N_pad, softmax @ V
// over the full N_pad).  Materialized logits [BLOCK_M, N_pad] fp32 fit in L2
// for N up to ~8k with BLOCK_M=32, so flash-style streaming softmax adds
// per-chunk rescaling overhead with no cache-fit benefit.
template <typename scalar_t, int BLOCK_M>
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
  const int N_pad = div_up(N, TILE_K) * TILE_K;  // pad to AMX K-tile alignment

  parallel_for(B * H, [&](int begin, int end) {
    int bs{0}, head_id{0};
    data_index_init(begin, bs, B, head_id, H);

    int tid = get_thread_num();
    char* base = reinterpret_cast<char*>(buffer) + static_cast<size_t>(tid) * buffer_size_per_thread;
    char* base0 = base;

    auto bump = [&](size_t nbytes) {
      void* p = base;
      base += (nbytes + 63) & ~size_t{63};
      return p;
    };

    // Gather buffers for this head's K/V (strided read from qkvg then packed).
    scalar_t* K_contig = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));
    scalar_t* V_contig = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));

    // VNNI-packed K and V, reused across all M-blocks of this (bs, head).
    scalar_t* K_vnni = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * K));
    scalar_t* V_vnni = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * K));

    // Non-flash scratch: full [BLOCK_M, N_pad] logits + bf16 softmax for the
    // attn @ V brgemm.  For N=4655 this is ~600KB + ~300KB, inside GNR's 2MB L2.
    float* logits_fp32 = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * N_pad));
    scalar_t* softmax_bf16 = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * N_pad));
    float* v_prime = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * K));
    scalar_t* Q_block = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * K));
    (void)base0;  // silence unused warning; base tracks offsets via closure

    // Zero the packed-K/V tail regions once per thread: pack_vnni{,2} write only
    // columns/rows in [0, N); the [N, N_pad) padding must be zero so the next
    // brgemm gets 0-contributions there.  Safe to do on first entry since these
    // buffers are not touched elsewhere beyond pack_vnni output.
    std::memset(K_vnni, 0, sizeof(scalar_t) * N_pad * K);
    std::memset(V_vnni, 0, sizeof(scalar_t) * N_pad * K);

    for (int idx = begin; idx < end; ++idx) {
      const scalar_t* qkvg_bs = qkvg + static_cast<size_t>(bs) * N * QKVG_STRIDE;
      scalar_t* attn_bs = gated_attn + static_cast<size_t>(bs) * N * D;

      // In qkvg, each row is [Q(D) | K(D) | V(D) | G(D)].
      const size_t Q_off = static_cast<size_t>(head_id) * K;
      const size_t K_off = static_cast<size_t>(D) + static_cast<size_t>(head_id) * K;
      const size_t V_off = static_cast<size_t>(2 * D) + static_cast<size_t>(head_id) * K;
      const size_t G_off = static_cast<size_t>(3 * D) + static_cast<size_t>(head_id) * K;

      // Gather this head's K/V into contiguous scratch once per (bs, head).
      for (int n = 0; n < N; ++n) {
        const scalar_t* src_k = qkvg_bs + static_cast<size_t>(n) * QKVG_STRIDE + K_off;
        const scalar_t* src_v = qkvg_bs + static_cast<size_t>(n) * QKVG_STRIDE + V_off;
        std::memcpy(K_contig + static_cast<size_t>(n) * K, src_k, sizeof(scalar_t) * K);
        std::memcpy(V_contig + static_cast<size_t>(n) * K, src_v, sizeof(scalar_t) * K);
      }

      // Pack K into [K/2, N_pad, 2] and V into [N_pad/2, K, 2] once per (bs, h).
      pack_vnni<scalar_t>(
          /* dst */ K_vnni,
          /* src */ K_contig,
          /* N */ N,
          /* K */ K,
          /* ld_src */ K,
          /* ld_dst */ N_pad);
      pack_vnni2<scalar_t>(
          /* dst */ V_vnni,
          /* src */ V_contig,
          /* K */ N,
          /* N */ K,
          /* ld_src */ K,
          /* ld_dst */ K);

      const scalar_t* bias_h = bias + static_cast<size_t>(head_id) * bias_strideH;

      for (int m = 0; m < N; m += BLOCK_M) {
        int m_size = std::min(BLOCK_M, N - m);

        // Gather Q block to a contig buffer for brgemm.
        for (int r = 0; r < m_size; ++r) {
          const scalar_t* src_q = qkvg_bs + static_cast<size_t>(m + r) * QKVG_STRIDE + Q_off;
          std::memcpy(Q_block + static_cast<size_t>(r) * K, src_q, sizeof(scalar_t) * K);
        }

        // ---- Q @ K^T -> logits_fp32 [m_size, N_pad] ----
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ N_pad, /* K */ K,
            /* lda */ K, /* ldb */ N_pad, /* ldc */ N_pad,
            /* add_C */ false,
            Q_block,
            K_vnni,
            logits_fp32);

        // ---- softmax per row (scale + bias + normalize + cast to bf16) ----
        for (int r = 0; r < m_size; ++r) {
          const scalar_t* bias_row = bias_h + static_cast<size_t>(m + r) * bias_strideM;
          softmax_row_to_bf16<scalar_t>(
              logits_fp32 + static_cast<size_t>(r) * N_pad,
              softmax_bf16 + static_cast<size_t>(r) * N_pad,
              bias_row,
              N,
              N_pad,
              sm_scale);
        }

        // ---- softmax @ V -> v_prime [m_size, K] ----
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ K, /* K */ N_pad,
            /* lda */ N_pad, /* ldb */ K, /* ldc */ K,
            /* add_C */ false,
            softmax_bf16,
            V_vnni,
            v_prime);

        // ---- sigmoid(G) * v_prime -> gated_attn (softmax already normalized) ----
        for (int r = 0; r < m_size; ++r) {
          scalar_t* dst = attn_bs + static_cast<size_t>(m + r) * D + static_cast<size_t>(head_id) * K;
          const scalar_t* g_row = qkvg_bs + static_cast<size_t>(m + r) * QKVG_STRIDE + G_off;
          sigmoid_mul_scale_bf16gate_stub<scalar_t>(dst, g_row, v_prime + r * K, /*inv_s=*/1.f, K);
        }
      }

      data_index_step(bs, B, head_id, H);
    }
    at::native::cpublas::brgemm_release();
  });
}

// v5 core: parallelism inverted to per-b, with a per-thread qkvg_row [N, 4*D]
// scratch that takes the place of v2's global [B, N, 4*D] qkvg.  Each thread
// owns one bs at a time and runs:
//     (A) pair[bs] @ qkvg_weight  -> qkvg_row  (serial, per-thread)
//     (B) for each head: gather K/V/G, pack VNNI, attention over all BLOCK_M,
//         write gated_attn[bs] head slice
// This keeps v2's single wide-N concat-QKVG GEMM shape but eliminates the
// 22 GB global qkvg intermediate that drove THP / first-touch-fault variance.
template <typename scalar_t, int BLOCK_M>
void fused_attention_concat_proj_core_impl(
    scalar_t* __restrict__ gated_attn,          // [B, N, H*K]
    const scalar_t* __restrict__ pair,          // [B, N, H*K]
    const scalar_t* __restrict__ bias,          // [H, N, N]
    const scalar_t* __restrict__ qkvg_weight,   // packed: [D/2, 4*D, 2]
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
  const int QKVG_STRIDE = 4 * D;
  const int N_pad = div_up(N, TILE_K) * TILE_K;
  TORCH_CHECK(
      K == block_size_n(),
      "fused_grid_attention_v5 requires per-head dim K == BLOCK_N (=",
      block_size_n(),
      "), got K=",
      K);

  parallel_for(B, [&](int begin, int end) {
    int tid = get_thread_num();
    char* base = reinterpret_cast<char*>(buffer) + static_cast<size_t>(tid) * buffer_size_per_thread;

    auto bump = [&](size_t nbytes) {
      void* p = base;
      base += (nbytes + 63) & ~size_t{63};
      return p;
    };

    // Per-thread qkvg scratch [N, 4*D]: ~4.77 MB at N=4655/D=128.  Replaces
    // v2's global 22 GB qkvg.  Written once per bs (stage A), then gathered
    // from by all H heads (stage B).
    scalar_t* qkvg_row = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * 4 * D));

    // Gather buffers for this head's K/V (strided read from qkvg_row, packed).
    scalar_t* K_contig = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));
    scalar_t* V_contig = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));

    // VNNI-packed K and V, reused across all M-blocks of this (bs, head).
    scalar_t* K_vnni = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * K));
    scalar_t* V_vnni = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * K));

    // Attention scratch.
    float* logits_fp32 = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * N_pad));
    scalar_t* softmax_bf16 = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * N_pad));
    float* v_prime = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * K));
    scalar_t* Q_block = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * K));

    // Zero the [N, N_pad) padding tails once per thread (see v2 core).
    std::memset(K_vnni, 0, sizeof(scalar_t) * N_pad * K);
    std::memset(V_vnni, 0, sizeof(scalar_t) * N_pad * K);

    for (int bs = begin; bs < end; ++bs) {
      const scalar_t* pair_bs = pair + static_cast<size_t>(bs) * N * D;
      scalar_t* attn_bs = gated_attn + static_cast<size_t>(bs) * N * D;

      // ----- Stage A: qkvg_row = pair_bs @ qkvg_weight (serial) -----
      // Row stride: pair_bs is [N, D] contiguous, so strideM = D.
      // qkvg_row is [N, 4*D] contiguous, so strideM = 4*D.
      weight_packed_linear_serial_impl<scalar_t>(
          qkvg_row,
          pair_bs,
          qkvg_weight,
          /* M */ N,
          /* N */ 4 * D,
          /* K */ D,
          /* mat1_strideM */ D,
          /* out_strideM  */ 4 * D);

      // ----- Stage B: per-head attention -----
      for (int head_id = 0; head_id < H; ++head_id) {
        const size_t Q_off = static_cast<size_t>(head_id) * K;
        const size_t K_off = static_cast<size_t>(D) + static_cast<size_t>(head_id) * K;
        const size_t V_off = static_cast<size_t>(2 * D) + static_cast<size_t>(head_id) * K;
        const size_t G_off = static_cast<size_t>(3 * D) + static_cast<size_t>(head_id) * K;

        // Gather this head's K/V into contiguous scratch.
        for (int n = 0; n < N; ++n) {
          const scalar_t* src_k = qkvg_row + static_cast<size_t>(n) * QKVG_STRIDE + K_off;
          const scalar_t* src_v = qkvg_row + static_cast<size_t>(n) * QKVG_STRIDE + V_off;
          std::memcpy(K_contig + static_cast<size_t>(n) * K, src_k, sizeof(scalar_t) * K);
          std::memcpy(V_contig + static_cast<size_t>(n) * K, src_v, sizeof(scalar_t) * K);
        }

        pack_vnni<scalar_t>(
            /* dst */ K_vnni,
            /* src */ K_contig,
            /* N */ N,
            /* K */ K,
            /* ld_src */ K,
            /* ld_dst */ N_pad);
        pack_vnni2<scalar_t>(
            /* dst */ V_vnni,
            /* src */ V_contig,
            /* K */ N,
            /* N */ K,
            /* ld_src */ K,
            /* ld_dst */ K);

        const scalar_t* bias_h = bias + static_cast<size_t>(head_id) * bias_strideH;

        for (int m = 0; m < N; m += BLOCK_M) {
          int m_size = std::min(BLOCK_M, N - m);

          // Gather Q block from qkvg_row.
          for (int r = 0; r < m_size; ++r) {
            const scalar_t* src_q = qkvg_row + static_cast<size_t>(m + r) * QKVG_STRIDE + Q_off;
            std::memcpy(Q_block + static_cast<size_t>(r) * K, src_q, sizeof(scalar_t) * K);
          }

          // Q @ K^T -> logits_fp32 [m_size, N_pad]
          at::native::cpublas::brgemm(
              /* M */ m_size, /* N */ N_pad, /* K */ K,
              /* lda */ K, /* ldb */ N_pad, /* ldc */ N_pad,
              /* add_C */ false,
              Q_block,
              K_vnni,
              logits_fp32);

          for (int r = 0; r < m_size; ++r) {
            const scalar_t* bias_row = bias_h + static_cast<size_t>(m + r) * bias_strideM;
            softmax_row_to_bf16<scalar_t>(
                logits_fp32 + static_cast<size_t>(r) * N_pad,
                softmax_bf16 + static_cast<size_t>(r) * N_pad,
                bias_row,
                N,
                N_pad,
                sm_scale);
          }

          // softmax @ V -> v_prime [m_size, K]
          at::native::cpublas::brgemm(
              /* M */ m_size, /* N */ K, /* K */ N_pad,
              /* lda */ N_pad, /* ldb */ K, /* ldc */ K,
              /* add_C */ false,
              softmax_bf16,
              V_vnni,
              v_prime);

          for (int r = 0; r < m_size; ++r) {
            scalar_t* dst = attn_bs + static_cast<size_t>(m + r) * D + static_cast<size_t>(head_id) * K;
            const scalar_t* g_row = qkvg_row + static_cast<size_t>(m + r) * QKVG_STRIDE + G_off;
            sigmoid_mul_scale_bf16gate_stub<scalar_t>(dst, g_row, v_prime + r * K, /*inv_s=*/1.f, K);
          }
        }
      }
    }
    at::native::cpublas::brgemm_release();
  });
}

// v3 core: v1's per-head tiled projection layout (no [B,N,4*D] intermediate)
// married to v2's full-[BLOCK_M, N_pad] attention core.  Stays entirely on
// per-thread scratch; the only multi-GB allocation is the final gated output
// (same as v1 / TPP), so the THP / first-touch-fault variance that plagues v2
// disappears while keeping v2's single-softmax / single-GEMM tail.
template <typename scalar_t, int BLOCK_M>
void fused_attention_proj_core_impl(
    scalar_t* __restrict__ gated_attn,   // [B, N, H*K]
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
  const int N_pad = div_up(N, TILE_K) * TILE_K;
  TORCH_CHECK(
      K == block_size_n(),
      "fused_grid_attention_v3 requires per-head dim K == BLOCK_N (=",
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

    // Per-head full projections (bf16, reused across all M-blocks).
    scalar_t* K_full = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));
    scalar_t* V_full = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N * K));

    // VNNI-packed K and V, reused across all M-blocks of this (bs, head).
    scalar_t* K_vnni = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * K));
    scalar_t* V_vnni = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * K));

    // Attention scratch.
    float* logits_fp32 = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * N_pad));
    scalar_t* softmax_bf16 = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * N_pad));
    float* v_prime = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * K));

    // Projection scratch (fp32 brgemm accumulator + bf16 Q row cache).
    float* Ctmp = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * K));
    scalar_t* Q_block = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * K));

    // Zero the [N, N_pad) tail of packed K/V so the attention brgemm's padded
    // tile contributes nothing.  Done once per thread.
    std::memset(K_vnni, 0, sizeof(scalar_t) * N_pad * K);
    std::memset(V_vnni, 0, sizeof(scalar_t) * N_pad * K);

    for (int idx = begin; idx < end; ++idx) {
      const scalar_t* pair_bs = pair + static_cast<size_t>(bs) * N * D;
      scalar_t* attn_bs = gated_attn + static_cast<size_t>(bs) * N * D;

      // Per-head slice of each packed weight: block `head_id` in the NB dim.
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

      // Pack K once as [K/2, N_pad, 2] and V once as [N_pad/2, K, 2].
      pack_vnni<scalar_t>(
          /* dst */ K_vnni,
          /* src */ K_full,
          /* N */ N,
          /* K */ K,
          /* ld_src */ K,
          /* ld_dst */ N_pad);
      pack_vnni2<scalar_t>(
          /* dst */ V_vnni,
          /* src */ V_full,
          /* K */ N,
          /* N */ K,
          /* ld_src */ K,
          /* ld_dst */ K);

      const scalar_t* bias_h = bias + static_cast<size_t>(head_id) * bias_strideH;

      for (int m = 0; m < N; m += BLOCK_M) {
        int m_size = std::min(BLOCK_M, N - m);

        // ---- Q projection for this block ----
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ K, /* K */ D,
            /* lda */ D, /* ldb */ K, /* ldc */ K,
            /* add_C */ false,
            pair_bs + static_cast<size_t>(m) * D,
            q_w_h,
            Ctmp);
        for (int r = 0; r < m_size; ++r) {
          cast_copy_stub<scalar_t>(Q_block + static_cast<size_t>(r) * K, Ctmp + r * K, K);
        }

        // ---- Q @ K^T -> logits_fp32 [m_size, N_pad] ----
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ N_pad, /* K */ K,
            /* lda */ K, /* ldb */ N_pad, /* ldc */ N_pad,
            /* add_C */ false,
            Q_block,
            K_vnni,
            logits_fp32);

        // ---- softmax per row (scale + bias + normalize + cast to bf16) ----
        for (int r = 0; r < m_size; ++r) {
          const scalar_t* bias_row = bias_h + static_cast<size_t>(m + r) * bias_strideM;
          softmax_row_to_bf16<scalar_t>(
              logits_fp32 + static_cast<size_t>(r) * N_pad,
              softmax_bf16 + static_cast<size_t>(r) * N_pad,
              bias_row,
              N,
              N_pad,
              sm_scale);
        }

        // ---- softmax @ V -> v_prime [m_size, K] ----
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ K, /* K */ N_pad,
            /* lda */ N_pad, /* ldb */ K, /* ldc */ K,
            /* add_C */ false,
            softmax_bf16,
            V_vnni,
            v_prime);

        // ---- Gating projection for this block (fp32 Ctmp) ----
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ K, /* K */ D,
            /* lda */ D, /* ldb */ K, /* ldc */ K,
            /* add_C */ false,
            pair_bs + static_cast<size_t>(m) * D,
            g_w_h,
            Ctmp);

        // ---- sigmoid(G) * v_prime -> gated_attn (softmax already normalized) ----
        for (int r = 0; r < m_size; ++r) {
          scalar_t* dst = attn_bs + static_cast<size_t>(m + r) * D + static_cast<size_t>(head_id) * K;
          sigmoid_mul_scale_stub<scalar_t>(dst, Ctmp + r * K, v_prime + r * K, /*inv_s=*/1.f, K);
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

  using clock = std::chrono::high_resolution_clock;
  const bool prof = v2_profile_enabled();
  auto t_start = prof ? clock::now() : clock::time_point{};

  TORCH_CHECK(
      qkvg_weight.size(0) * 2 == 4 * D || qkvg_weight.numel() == 4 * D * D,
      "qkvg_weight must map D -> 4*D");

  // ----- Stage A: one concat QKVG GEMM, pair streamed once -----
  auto pair_2d = pair.view({static_cast<int64_t>(B) * N, D});
  auto qkvg_2d = at::empty({static_cast<int64_t>(B) * N, 4 * D}, pair.options());
  auto t_qkvg_alloc = prof ? clock::now() : clock::time_point{};

  weight_packed_linear_out(qkvg_2d, pair_2d, qkvg_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  auto t_qkvg_gemm = prof ? clock::now() : clock::time_point{};

  auto qkvg = qkvg_2d.view({B, N, 4 * D});

  // ----- Stage B: fused flash-attn + sigmoid(G) * attn -----
  auto gated_attn = at::empty({B, N, D}, pair.options());
  auto t_gated_alloc = prof ? clock::now() : clock::time_point{};

  constexpr int BLOCK_M = 32;
  const int N_pad = div_up(N, TILE_K) * TILE_K;

  const int num_threads = at::get_num_threads();
  const int per_thread_bytes =
      /* K_contig     */ sizeof(uint16_t) * N * K +
      /* V_contig     */ sizeof(uint16_t) * N * K +
      /* K_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* V_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* logits_fp32  */ sizeof(float) * BLOCK_M * N_pad +
      /* softmax_bf16 */ sizeof(uint16_t) * BLOCK_M * N_pad +
      /* v_prime      */ sizeof(float) * BLOCK_M * K +
      /* Q_block      */ sizeof(uint16_t) * BLOCK_M * K +
      /* alignment padding */ 64 * 10;
  auto buffer = at::empty({num_threads, per_thread_bytes}, pair.options().dtype(at::kChar));
  auto t_buf_alloc = prof ? clock::now() : clock::time_point{};

  AT_DISPATCH_REDUCED_FLOATING_TYPES(pair.scalar_type(), "fused_grid_attention_v2", [&] {
    fused_attention_core_impl<scalar_t, BLOCK_M>(
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
  auto t_attn = prof ? clock::now() : clock::time_point{};

  // ----- Stage C: final output projection -----
  auto gated_attn_2d = gated_attn.view({static_cast<int64_t>(B) * N, D});
  auto out_2d = at::empty({static_cast<int64_t>(B) * N, D}, pair.options());
  auto t_out_alloc = prof ? clock::now() : clock::time_point{};

  weight_packed_linear_out(out_2d, gated_attn_2d, output_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  auto t_out_gemm = prof ? clock::now() : clock::time_point{};

  if (prof) {
    auto ms = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    // Stage A split into qkvg-alloc (~22 GB at N=4655) vs gemm,
    // Stage B split into gated_attn-alloc (~5.5 GB) / per-thread scratch / attn core,
    // Stage C split into output-alloc (~5.5 GB) vs output gemm.
    std::fprintf(
        stderr,
        "[v2] qkvg_alloc=%7.2f qkvg_gemm=%7.2f gated_alloc=%7.2f buf_alloc=%6.2f attn=%8.2f out_alloc=%7.2f out_gemm=%7.2f total=%8.2f ms\n",
        ms(t_start, t_qkvg_alloc),
        ms(t_qkvg_alloc, t_qkvg_gemm),
        ms(t_qkvg_gemm, t_gated_alloc),
        ms(t_gated_alloc, t_buf_alloc),
        ms(t_buf_alloc, t_attn),
        ms(t_attn, t_out_alloc),
        ms(t_out_alloc, t_out_gemm),
        ms(t_start, t_out_gemm));
    std::fflush(stderr);
  }

  return out_2d.view({B, N, D});
}

// v4: B-tiled v2.  Keeps v2's single concat-QKVG GEMM but processes the outer
// batch dim in chunks of BLOCK_B rows, so the qkvg intermediate lives in a
// small reusable scratch (~76 MB at BLOCK_B=16, N=4655/bf16) instead of a
// fresh 22 GB allocation per call.  The scratch fits in L3, so stage-A writes
// and stage-B reads stay mostly on-chip, and there is no fault storm or THP
// lottery on multi-GB mmap memory.  Stateless: no persistent cache needed.
//
//   pair          : [B, N, D]
//   bias          : [H, N, N]
//   qkvg_weight   : packed [4*D, D] (concat of Q|K|V|G)
//   output_weight : packed [D, D]
//   returns       : [B, N, D]
at::Tensor fused_grid_attention_v4(
    at::Tensor& pair,
    at::Tensor& bias,
    at::Tensor& qkvg_weight,
    at::Tensor& output_weight,
    int64_t num_heads,
    bool is_vnni) {
  RECORD_FUNCTION(
      "sgl_kernel::fused_grid_attention_v4",
      std::vector<c10::IValue>({pair, bias, qkvg_weight, output_weight, num_heads}));

  CHECK_INPUT(pair);
  CHECK_DIM(3, pair);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(bias);
  CHECK_DIM(3, bias);
  CHECK_INPUT(qkvg_weight);
  CHECK_INPUT(output_weight);
  TORCH_CHECK(is_vnni, "fused_grid_attention_v4 currently requires pre-packed weights (is_vnni=True).");

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

  constexpr int BLOCK_M = 32;
  // B-chunk size.  Chosen so qkvg_chunk = BLOCK_B * N * 4D * sizeof(bf16)
  // stays in the ~100 MB ballpark — fits in L3 on SPR/GNR class CPUs at
  // N ≤ ~5k, which is where we care about this kernel.  Tune per machine if
  // N or L3 differs significantly.
  constexpr int BLOCK_B = 16;
  const int N_pad = div_up(N, TILE_K) * TILE_K;

  // Full-size output buffer, same as v2/v3.
  auto gated_attn = at::empty({B, N, D}, pair.options());

  // Reusable B-chunk qkvg scratch.  Allocated at full BLOCK_B so the last
  // (potentially smaller) tail chunk reuses the same buffer.
  auto qkvg_chunk = at::empty({static_cast<int64_t>(BLOCK_B) * N, 4 * D}, pair.options());

  const int num_threads = at::get_num_threads();
  const int per_thread_bytes =
      /* K_contig     */ sizeof(uint16_t) * N * K +
      /* V_contig     */ sizeof(uint16_t) * N * K +
      /* K_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* V_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* logits_fp32  */ sizeof(float) * BLOCK_M * N_pad +
      /* softmax_bf16 */ sizeof(uint16_t) * BLOCK_M * N_pad +
      /* v_prime      */ sizeof(float) * BLOCK_M * K +
      /* Q_block      */ sizeof(uint16_t) * BLOCK_M * K +
      /* alignment padding */ 64 * 10;
  auto buffer = at::empty({num_threads, per_thread_bytes}, pair.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(pair.scalar_type(), "fused_grid_attention_v4", [&] {
    for (int b_start = 0; b_start < B; b_start += BLOCK_B) {
      const int b_size = std::min(BLOCK_B, B - b_start);

      // Stage A: pair_chunk @ qkvg_weight -> qkvg_chunk (first b_size*N rows).
      auto pair_chunk_2d = pair
          .narrow(/*dim=*/0, /*start=*/b_start, /*length=*/b_size)
          .reshape({static_cast<int64_t>(b_size) * N, D});
      auto qkvg_chunk_2d = qkvg_chunk.narrow(/*dim=*/0, /*start=*/0, /*length=*/static_cast<int64_t>(b_size) * N);
      weight_packed_linear_out(
          qkvg_chunk_2d, pair_chunk_2d, qkvg_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);

      // Stage B: attention on this chunk, writing into the chunk's slice of gated_attn.
      fused_attention_core_impl<scalar_t, BLOCK_M>(
          gated_attn.data_ptr<scalar_t>() + static_cast<size_t>(b_start) * N * D,
          qkvg_chunk.data_ptr<scalar_t>(),
          bias.data_ptr<scalar_t>(),
          b_size,
          N,
          H,
          K,
          bias_strideH,
          bias_strideM,
          static_cast<float>(sm_scale),
          buffer.data_ptr(),
          per_thread_bytes);
    }
  });

  // ----- Stage C: final output projection -----
  auto gated_attn_2d = gated_attn.view({static_cast<int64_t>(B) * N, D});
  auto out_2d = weight_packed_linear(gated_attn_2d, output_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  return out_2d.view({B, N, D});
}

// v5: v2's concat-QKVG GEMM shape with parallelism inverted to per-b, and the
// global [B, N, 4*D] qkvg replaced by a per-thread [N, 4*D] scratch row.  Each
// thread owns one bs, runs the concat projection serially into its scratch,
// then loops over H heads for attention — keeping v2's single wide-N GEMM
// shape while eliminating the 22 GB global intermediate that caused THP /
// first-touch-fault variance.
//
//   pair          : [B, N, D]
//   bias          : [H, N, N]
//   qkvg_weight   : packed [4*D, D] (concat of Q|K|V|G)
//   output_weight : packed [D, D]
//   returns       : [B, N, D]
at::Tensor fused_grid_attention_v5(
    at::Tensor& pair,
    at::Tensor& bias,
    at::Tensor& qkvg_weight,
    at::Tensor& output_weight,
    int64_t num_heads,
    bool is_vnni) {
  RECORD_FUNCTION(
      "sgl_kernel::fused_grid_attention_v5",
      std::vector<c10::IValue>({pair, bias, qkvg_weight, output_weight, num_heads}));

  CHECK_INPUT(pair);
  CHECK_DIM(3, pair);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(bias);
  CHECK_DIM(3, bias);
  CHECK_INPUT(qkvg_weight);
  CHECK_INPUT(output_weight);
  TORCH_CHECK(is_vnni, "fused_grid_attention_v5 currently requires pre-packed weights (is_vnni=True).");

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

  auto gated_attn = at::empty({B, N, D}, pair.options());

  constexpr int BLOCK_M = 32;
  const int N_pad = div_up(N, TILE_K) * TILE_K;

  const int num_threads = at::get_num_threads();
  const int per_thread_bytes =
      /* qkvg_row     */ sizeof(uint16_t) * N * 4 * D +
      /* K_contig     */ sizeof(uint16_t) * N * K +
      /* V_contig     */ sizeof(uint16_t) * N * K +
      /* K_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* V_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* logits_fp32  */ sizeof(float) * BLOCK_M * N_pad +
      /* softmax_bf16 */ sizeof(uint16_t) * BLOCK_M * N_pad +
      /* v_prime      */ sizeof(float) * BLOCK_M * K +
      /* Q_block      */ sizeof(uint16_t) * BLOCK_M * K +
      /* alignment padding */ 64 * 10;
  auto buffer = at::empty({num_threads, per_thread_bytes}, pair.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(pair.scalar_type(), "fused_grid_attention_v5", [&] {
    fused_attention_concat_proj_core_impl<scalar_t, BLOCK_M>(
        gated_attn.data_ptr<scalar_t>(),
        pair.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        qkvg_weight.data_ptr<scalar_t>(),
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

// v3: v1's per-head tiled projection layout + v2's full-logit attention core.
// No [B, N, 4*D] qkvg intermediate.  Only multi-GB allocation is the [B, N, D]
// gated_attn output (same footprint as v1 / TPP), so THP / first-touch-fault
// variance is eliminated while keeping v2's single-softmax speed.
//
//   pair          : [B, N, D]
//   bias          : [H, N, N]
//   q/k/v/g_w     : packed [D/2, D, 2]
//   output_weight : packed [D, D]
//   returns       : [B, N, D]
at::Tensor fused_grid_attention_v3(
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
      "sgl_kernel::fused_grid_attention_v3",
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
  TORCH_CHECK(is_vnni, "fused_grid_attention_v3 currently requires pre-packed weights (is_vnni=True).");

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

  constexpr int BLOCK_M = 32;
  const int N_pad = div_up(N, TILE_K) * TILE_K;

  const int num_threads = at::get_num_threads();
  const int per_thread_bytes =
      /* K_full       */ sizeof(uint16_t) * N * K +
      /* V_full       */ sizeof(uint16_t) * N * K +
      /* K_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* V_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* logits_fp32  */ sizeof(float) * BLOCK_M * N_pad +
      /* softmax_bf16 */ sizeof(uint16_t) * BLOCK_M * N_pad +
      /* v_prime      */ sizeof(float) * BLOCK_M * K +
      /* Ctmp         */ sizeof(float) * BLOCK_M * K +
      /* Q_block      */ sizeof(uint16_t) * BLOCK_M * K +
      /* alignment padding */ 64 * 10;

  // Function-local persistent scratch — mirrors libxsmm's scratchpad pool.
  // Keeps gated_attn (~5.5 GB @ N=4655) and the per-thread scratch (~240 MB)
  // across calls so we don't re-pay first-touch / THP-coalesce cost each time.
  // Shape key covers everything that changes the layout; on mismatch we
  // realloc + pre-touch.
  struct V3Cache {
    at::Tensor gated_attn;
    at::Tensor scratch;
    int64_t B = -1, N = -1, D = -1;
    int num_threads = -1;
    int64_t per_thread_bytes = -1;
    c10::ScalarType dtype = c10::ScalarType::Undefined;
  };
  static V3Cache cache;

  const bool shape_changed = !cache.gated_attn.defined()
      || cache.B != B || cache.N != N || cache.D != D
      || cache.num_threads != num_threads
      || cache.per_thread_bytes != per_thread_bytes
      || cache.dtype != pair.scalar_type();
  if (shape_changed) {
    cache.gated_attn = at::empty({B, N, D}, pair.options());
    cache.scratch = at::empty({num_threads, per_thread_bytes}, pair.options().dtype(at::kChar));
    cache.gated_attn.fill_(0);  // pre-touch once so first-use faults stay off the hot path
    cache.scratch.fill_(0);
    cache.B = B;
    cache.N = N;
    cache.D = D;
    cache.num_threads = num_threads;
    cache.per_thread_bytes = per_thread_bytes;
    cache.dtype = pair.scalar_type();
  }
  auto& gated_attn = cache.gated_attn;
  auto& buffer = cache.scratch;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(pair.scalar_type(), "fused_grid_attention_v3", [&] {
    fused_attention_proj_core_impl<scalar_t, BLOCK_M>(
        gated_attn.data_ptr<scalar_t>(),
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

  auto gated_attn_2d = gated_attn.view({static_cast<int64_t>(B) * N, D});
  auto out_2d = weight_packed_linear(gated_attn_2d, output_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  return out_2d.view({B, N, D});
}

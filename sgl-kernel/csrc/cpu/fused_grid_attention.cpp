/*****************************************************************************************
 * Fused grid self-attention kernel for sglang CPU backend.
 *
 * Stage-split fusion (the surviving design after benchmarking several variants):
 *   - Stage A: one concat QKV+gating projection (weight_packed_linear) over the
 *     whole [B*N, D] pair, so the pair is streamed once and Q/K/V/gating land in
 *     a single [B, N, 4*D] buffer.
 *   - Stage B: a fused attention tail.  Per (bs, head) pack K/V to VNNI once, then
 *     per M-block run Q @ K^T, a non-flash per-row softmax (scale + bias), and
 *     softmax @ V; multiply by sigmoid(gate) and store interleaved across heads
 *     into a [B, N, H*K] buffer.
 *   - Stage C: the final output projection as a standard weight_packed_linear,
 *     written back into `pair` in place (matching how TPP splits its loops).
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

// Gate per-call stage timing for fused_grid_attention behind SGL_FGA_PROFILE=1.
// Reading the env once at first use keeps the fast path free of getenv calls.
inline bool fga_profile_enabled() {
  static const bool enabled = []{
    const char* s = std::getenv("SGL_FGA_PROFILE");
    return s && s[0] != '\0' && std::strcmp(s, "0") != 0;
  }();
  return enabled;
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

// Per-row softmax for the non-flash core: scale + bias, max, exp+sum, normalize,
// cast to bf16, zero the [N, N_pad) tail so it contributes 0 to attn @ V.
//
// row_fp32 is rewritten in place (scratch for the three passes).
// Pass 1 computes row = row * sm_scale + bias.
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

// Core: consumes an already-projected QKVG buffer [B, N, 4*D] produced by a
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

    // VNNI-packed K and V, reused across all M-blocks of this (bs, head).
    scalar_t* K_vnni = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * K));
    scalar_t* V_vnni = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * K));

    // Non-flash scratch: full [BLOCK_M, N_pad] logits + bf16 softmax for the
    // attn @ V brgemm.  For N=4655 this is ~600KB + ~300KB, inside GNR's 2MB L2.
    float* logits_fp32 = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * N_pad));
    scalar_t* softmax_bf16 = reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * BLOCK_M * N_pad));
    float* v_prime = reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * K));
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

      // Pack K/V directly from qkvg with strided reads (ld_src = QKVG_STRIDE
      // skips over the interleaved Q/V/G or Q/K/G slots between rows).
      pack_vnni<scalar_t>(
          /* dst */ K_vnni,
          /* src */ qkvg_bs + K_off,
          /* N */ N,
          /* K */ K,
          /* ld_src */ QKVG_STRIDE,
          /* ld_dst */ N_pad);
      pack_vnni2<scalar_t>(
          /* dst */ V_vnni,
          /* src */ qkvg_bs + V_off,
          /* K */ N,
          /* N */ K,
          /* ld_src */ QKVG_STRIDE,
          /* ld_dst */ K);

      const scalar_t* bias_h = bias + static_cast<size_t>(head_id) * bias_strideH;

      for (int m = 0; m < N; m += BLOCK_M) {
        int m_size = std::min(BLOCK_M, N - m);

        // ---- Q @ K^T -> logits_fp32 [m_size, N_pad] ----
        // Read Q directly from qkvg with strided lda (no gather needed).
        const scalar_t* Q_ptr = qkvg_bs + static_cast<size_t>(m) * QKVG_STRIDE + Q_off;
        at::native::cpublas::brgemm(
            /* M */ m_size, /* N */ N_pad, /* K */ K,
            /* lda */ QKVG_STRIDE, /* ldb */ N_pad, /* ldc */ N_pad,
            /* add_C */ false,
            Q_ptr,
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

}  // anonymous namespace

// Split stage-A QKVG concat GEMM + fused attention-tail core + stage-C out_proj.
// Keeps QKV+gating in one large-M GEMM for good pair-read bandwidth.
//
//   pair          : [B, N, D]              (D = H * K)
//   bias          : [H, N, N]
//   qkvg_weight   : packed [4*D, D] (pack of torch.cat([Q, K, V, G], dim=0))
//   output_weight : packed [D, D]
//   returns       : [B, N, D]
at::Tensor fused_grid_attention(
    at::Tensor& pair,
    at::Tensor& bias,
    at::Tensor& qkvg_weight,
    at::Tensor& output_weight,
    int64_t num_heads,
    bool is_vnni) {
  RECORD_FUNCTION(
      "sgl_kernel::fused_grid_attention",
      std::vector<c10::IValue>({pair, bias, qkvg_weight, output_weight, num_heads}));

  CHECK_INPUT(pair);
  CHECK_DIM(3, pair);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(bias);
  CHECK_DIM(3, bias);
  CHECK_INPUT(qkvg_weight);
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

  using clock = std::chrono::high_resolution_clock;
  const bool prof = fga_profile_enabled();
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
      /* K_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* V_vnni       */ sizeof(uint16_t) * N_pad * K +
      /* logits_fp32  */ sizeof(float) * BLOCK_M * N_pad +
      /* softmax_bf16 */ sizeof(uint16_t) * BLOCK_M * N_pad +
      /* v_prime      */ sizeof(float) * BLOCK_M * K +
      /* alignment padding */ 64 * 10;
  auto buffer = at::empty({num_threads, per_thread_bytes}, pair.options().dtype(at::kChar));
  auto t_buf_alloc = prof ? clock::now() : clock::time_point{};

  AT_DISPATCH_REDUCED_FLOATING_TYPES(pair.scalar_type(), "fused_grid_attention", [&] {
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
  // Write back into `pair` in place (TPP trick).  The Python
  // wrapper always feeds a fresh pair (post-LayerNorm), so clobbering it is
  // safe.  Removes one 5.5 GB at::empty/munmap cycle per call.  Reuses
  // `pair_2d` declared above for Stage A.
  auto gated_attn_2d = gated_attn.view({static_cast<int64_t>(B) * N, D});
  auto t_out_alloc = prof ? clock::now() : clock::time_point{};

  weight_packed_linear_out(pair_2d, gated_attn_2d, output_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  auto t_out_gemm = prof ? clock::now() : clock::time_point{};

  if (prof) {
    auto ms = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    // Stage A split into qkvg-alloc (~22 GB at N=4655) vs gemm,
    // Stage B split into gated_attn-alloc (~5.5 GB) / per-thread scratch / attn core,
    // Stage C: out_proj writes back into pair (no alloc).
    std::fprintf(
        stderr,
        "[fga] qkvg_alloc=%7.2f qkvg_gemm=%7.2f gated_alloc=%7.2f buf_alloc=%6.2f attn=%8.2f out_view=%7.2f out_gemm=%7.2f total=%8.2f ms\n",
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

  return pair;
}

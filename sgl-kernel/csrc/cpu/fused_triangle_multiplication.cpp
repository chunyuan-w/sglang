/*****************************************************************************************
 * Fused triangle multiplication kernel for sglang CPU backend.
 *
 * Mirrors the v2 grid-attention design (see af3-fused-cpu-kernel-perf skill):
 *   - Stage A: one concat (proj|gate) GEMM, pair streamed once, then per-row
 *     scatter that applies mask, sigmoid(gate)*proj and splits into bmm-ready
 *     `a [C, N, N]` and `b [C, N, N]` outputs.
 *   - Stage B: per-c parallel bmm core. b[c] is VNNI-packed into per-thread
 *     scratch (one batch at a time, same idiom as v2's K_vnni). Result is
 *     written directly in NHC layout, skipping the eager `.contiguous()`.
 *   - Stage C: layernorm + (out_proj | gating) + sigmoid(gate)*out_proj +
 *     residual, all fused per-row into the caller's pre-layernorm pair buffer
 *     (Choice 6: clobber-safe out_proj).
 *
 * Python wrapper contract (matches v2):
 *   - `pair_orig` is the pre-LayerNorm input AND the output buffer (clobbered
 *     in place by Stage C).
 *   - `pair_normed = left_norm(pair_orig)` is applied OUTSIDE the kernel.
 *   - All weights are pre-packed via convert_weight_packed (is_vnni=True).
 *
 * Layout conventions:
 *   pair_orig, pair_normed : [N, N, C]            bf16, contiguous
 *   mask                   : [N, N]               bf16, contiguous, binary {0,1}
 *   proj_gate_weight       : packed [4C, C] = cat(proj.W [2C,C] | gate.W [2C,C])
 *   center_norm_w / _b     : [C]                  bf16 / fp32-or-bf16
 *   out_proj_weight        : packed [C, C]
 *   gating_weight          : packed [C, C]
 *   outgoing               : bool                  true => cik,cjk->cij
 *                                                  false=> ckj,cki->cij
 *
 * Status:
 *   - Wrapper, Stage A, Stage C: complete (correctness-first, perf to tune).
 *   - Stage B: scaffolded. Inner brgemm path marked with TODO(perf-stage-b);
 *     starting point is per-c VNNI pack + brgemm-with-NHC-scatter.
 ****************************************************************************************/
#include "common.h"
#include "gemm.h"
#include "vec.h"
#include "vec_pack.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// SGL_TM_PROFILE=1 to dump per-stage timing.  Same idiom as v2's
// SGL_V2_PROFILE; reading the env once at first use keeps the fast path free.
inline bool tm_profile_enabled() {
  static const bool enabled = [] {
    const char* s = std::getenv("SGL_TM_PROFILE");
    return s && s[0] != '\0' && std::strcmp(s, "0") != 0;
  }();
  return enabled;
}

// ---------------------------------------------------------------------------
// Small element-wise helpers (per-row, fp32 intermediates).
// ---------------------------------------------------------------------------

template <typename scalar_t>
inline float sigmoid_fp32(scalar_t x) {
  float f = static_cast<float>(x);
  return 1.f / (1.f + std::exp(-f));
}

// Vectorized sigmoid_mul_add for Stage C tail:
//   pair_orig[d]  =  bf16( fp32(pair_orig[d]) + sigmoid(gate_f[d]) * proj_f[d] )
// Both gate_f / proj_f are fp32 (from brgemm Ctmp).
template <typename scalar_t>
inline void sigmoid_mul_add_inplace(
    scalar_t* __restrict__ pair_orig,
    const float* __restrict__ gate_f,
    const float* __restrict__ proj_f,
    int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec one(1.f);
  int d = 0;
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec g0 = fVec::loadu(gate_f + d);
    fVec g1 = fVec::loadu(gate_f + d + fVec::size());
    fVec s0 = one / (one + g0.neg().exp_u20());
    fVec s1 = one / (one + g1.neg().exp_u20());
    fVec p0 = fVec::loadu(proj_f + d);
    fVec p1 = fVec::loadu(proj_f + d + fVec::size());

    bVec orig = bVec::loadu(pair_orig + d);
    fVec o0, o1;
    std::tie(o0, o1) = at::vec::convert_to_float(orig);
    fVec r0 = o0 + s0 * p0;
    fVec r1 = o1 + s1 * p1;
    bVec out_vec = convert_from_float_ext<scalar_t>(r0, r1);
    out_vec.store(pair_orig + d);
  }
  for (; d < size; ++d) {
    float g = gate_f[d];
    float sg = 1.f / (1.f + std::exp(-g));
    float r = static_cast<float>(pair_orig[d]) + sg * proj_f[d];
    pair_orig[d] = static_cast<scalar_t>(r);
  }
}

// ---------------------------------------------------------------------------
// Stage A: pre-einsum.
//
// Reads pair_normed [M=N*N, C] and applies:
//     proj_gate = pair_normed @ proj_gate_weight.T              [M, 4C]
//     proj = proj_gate[:, :2C];  gate = proj_gate[:, 2C:]
//     proj *= mask.unsqueeze(-1)                                 [M, 2C]
//     proj *= sigmoid(gate)                                      [M, 2C]
//     for each row m=(i,j), for c in [0, C):
//       a[c, i, j] = proj[m, 2c    ]
//       b[c, i, j] = proj[m, 2c + 1]
//
// The big GEMM is shared with weight_packed_linear_out (no scratch beyond
// proj_gate itself).  The per-row scatter is hand-rolled here so we can fuse
// mask + sigmoid_mul + split + transpose write in one pass.
// ---------------------------------------------------------------------------

template <typename scalar_t>
void tm_pre_einsum_scatter_impl(
    scalar_t* __restrict__ a_out,                        // [C, N, N_pad]
    scalar_t* __restrict__ b_out,                        // [C, N, N_pad]
    const scalar_t* __restrict__ proj_gate,              // [M=N*N, 4C] (bf16, from GEMM)
    const scalar_t* __restrict__ mask,                   // [N, N] (bf16, binary)
    int N,
    int N_pad,
    int C) {
  const int64_t M = static_cast<int64_t>(N) * N;
  // a/b are allocated [C, N, N_pad].  The [N, N_pad) tail columns are
  // pre-zeroed by the at::zeros caller so the K-padded brgemm sees 0
  // contributions there.  Stage A's scatter only touches j in [0, N).
  const int64_t row_stride = static_cast<int64_t>(N_pad);
  const int64_t plane_stride = static_cast<int64_t>(N) * row_stride;
  const int64_t four_C = 4 * static_cast<int64_t>(C);

  at::parallel_for(0, M, /*grain_size=*/1024, [&](int64_t m_begin, int64_t m_end) {
    for (int64_t m = m_begin; m < m_end; ++m) {
      const int64_t i = m / N;
      const int64_t j = m - i * N;
      const float mask_ij = static_cast<float>(mask[m]);

      const scalar_t* row = proj_gate + m * four_C;
      const scalar_t* proj_row = row;                    // [0 .. 2C)
      const scalar_t* gate_row = row + 2 * C;            // [2C .. 4C)

      // TODO(perf-stage-a): vectorize this by processing 2 c values at a time
      // (one full lane pair = 32 bf16) and using scatter stores or a small
      // contig tile + per-c memcpy.
      for (int c = 0; c < C; ++c) {
        float p_a = static_cast<float>(proj_row[2 * c]);
        float p_b = static_cast<float>(proj_row[2 * c + 1]);
        float g_a = static_cast<float>(gate_row[2 * c]);
        float g_b = static_cast<float>(gate_row[2 * c + 1]);

        float sg_a = 1.f / (1.f + std::exp(-g_a));
        float sg_b = 1.f / (1.f + std::exp(-g_b));

        float a_val = mask_ij * sg_a * p_a;
        float b_val = mask_ij * sg_b * p_b;

        a_out[static_cast<int64_t>(c) * plane_stride + i * row_stride + j] =
            static_cast<scalar_t>(a_val);
        b_out[static_cast<int64_t>(c) * plane_stride + i * row_stride + j] =
            static_cast<scalar_t>(b_val);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Stage B: einsum core (bmm with batch=C, output in NHC layout).
//
// For outgoing  cik,cjk->cij :
//   out[i, j, c] = sum_k a[c, i, k] * b[c, j, k]
//   per-c bmm:   out_c [N, N] = a[c] @ b[c]^T
//
// For incoming  ckj,cki->cij :
//   out[i, j, c] = sum_k a[c, k, j] * b[c, k, i]
//   per-c bmm:   out_c [N, N] = b[c]^T @ a[c]
//   (equivalent to swapping operands of the outgoing form)
//
// Output layout is [N, N, C], NOT [C, N, N].  We write each (i, j, c) directly
// to skip the `permute(1,2,0).contiguous()` the eager path pays before
// center_norm (273 ms in the current SGL profile).
// ---------------------------------------------------------------------------

template <typename scalar_t, int BLOCK_M>
void tm_einsum_impl(
    scalar_t* __restrict__ einsum_out,                   // [N, N, C]
    const scalar_t* __restrict__ a,                      // [C, N, N_pad]
    const scalar_t* __restrict__ b,                      // [C, N, N_pad]
    int N,
    int N_pad,
    int C,
    bool outgoing,
    void* __restrict__ buffer,
    int buffer_size_per_thread) {
  // a/b plane stride is N * N_pad (K-padded inner dim).
  const int64_t plane_stride = static_cast<int64_t>(N) * N_pad;

  parallel_for(C, [&](int begin, int end) {
    int tid = get_thread_num();
    char* base = reinterpret_cast<char*>(buffer)
               + static_cast<size_t>(tid) * buffer_size_per_thread;

    auto bump = [&](size_t nbytes) {
      void* p = base;
      base += (nbytes + 63) & ~size_t{63};
      return p;
    };

    // Per-thread VNNI-packed b[c] (one c at a time) + fp32 brgemm Ctmp.
    // b_vnni holds [N_pad/2, N_pad, 2] = N_pad * N_pad elements after pack.
    scalar_t* b_vnni =
        reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * N_pad));
    float* logits_fp32 =
        reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * N_pad));

    // Zero the b_vnni scratch once per thread.  The tail rows / cols beyond
    // (N, N_pad) must read as 0 to keep brgemm correct.  pack_vnni fills only
    // the [0, N) × [0, N_pad) region of the VNNI grid.
    std::memset(b_vnni, 0, sizeof(scalar_t) * N_pad * N_pad);

    for (int c = begin; c < end; ++c) {
      const scalar_t* a_c = a + static_cast<int64_t>(c) * plane_stride;
      const scalar_t* b_c = b + static_cast<int64_t>(c) * plane_stride;

      // ---------------------------------------------------------------
      // VNNI-pack b[c] once per c.
      //
      // outgoing  cik,cjk->cij  : pack b as [N=out, K=contracted].
      // Source b[c] is [N, N_pad] with stride [N_pad, 1].  pack_vnni's
      // (N, K) args are (output features, contracted dim).
      // ---------------------------------------------------------------
      if (outgoing) {
        pack_vnni<scalar_t>(
            /*dst*/ b_vnni,
            /*src*/ b_c,
            /*N*/ N,
            /*K*/ N_pad,
            /*ld_src*/ N_pad,
            /*ld_dst*/ N_pad);
      } else {
        pack_vnni2<scalar_t>(
            /*dst*/ b_vnni,
            /*src*/ b_c,
            /*K*/ N,
            /*N*/ N_pad,
            /*ld_src*/ N_pad,
            /*ld_dst*/ N_pad);
      }

      // ---------------------------------------------------------------
      // Tile over M=N rows of the output for this c.
      //
      // brgemm: logits_fp32 [m_size, N_pad] = a[c, m:m+m_size, :] @ b_vnni
      // K dim = N_pad (AMX requires TILE_K-multiple K).
      // ---------------------------------------------------------------
      for (int m = 0; m < N; m += BLOCK_M) {
        int m_size = std::min(BLOCK_M, N - m);

        const scalar_t* a_block = a_c + static_cast<int64_t>(m) * N_pad;

        at::native::cpublas::brgemm(
            /*M*/ m_size, /*N*/ N_pad, /*K*/ N_pad,
            /*lda*/ N_pad, /*ldb*/ N_pad, /*ldc*/ N_pad,
            /*add_C*/ false,
            a_block,
            b_vnni,
            logits_fp32);

        // -------------------------------------------------------------
        // Cast + scatter: einsum_out[m+r, j, c] = bf16(logits_fp32[r, j])
        // dst stride between consecutive j's is C (since out is NHC).
        // TODO(perf-stage-b): vectorize the strided store.  For C=128
        // an 8x bf16 vector + scatter (or transposed-store via a tile
        // buffer) should be straightforward.
        // -------------------------------------------------------------
        for (int r = 0; r < m_size; ++r) {
          const float* src_row = logits_fp32 + static_cast<int64_t>(r) * N_pad;
          scalar_t* dst_base =
              einsum_out + (static_cast<int64_t>(m + r) * N) * C + c;
          for (int j = 0; j < N; ++j) {
            dst_base[static_cast<int64_t>(j) * C] =
                static_cast<scalar_t>(src_row[j]);
          }
        }
      }
    }

    at::native::cpublas::brgemm_release();
  });
}

// Sigmoid-mul-add tail, bf16 operands.  Reads gate_bf, proj_bf, pair_orig in
// bf16 and writes pair_orig in place:
//   pair_orig[d] = bf16( fp32(pair_orig[d]) + sigmoid(fp32(gate_bf[d])) * fp32(proj_bf[d]) )
template <typename scalar_t>
inline void sigmoid_mul_add_inplace_bf16(
    scalar_t* __restrict__ pair_orig,
    const scalar_t* __restrict__ gate_bf,
    const scalar_t* __restrict__ proj_bf,
    int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec one(1.f);
  int d = 0;
  for (; d <= size - kVecSize; d += kVecSize) {
    bVec g_bv = bVec::loadu(gate_bf + d);
    bVec p_bv = bVec::loadu(proj_bf + d);
    bVec o_bv = bVec::loadu(pair_orig + d);
    fVec g0, g1, p0, p1, o0, o1;
    std::tie(g0, g1) = at::vec::convert_to_float(g_bv);
    std::tie(p0, p1) = at::vec::convert_to_float(p_bv);
    std::tie(o0, o1) = at::vec::convert_to_float(o_bv);

    fVec s0 = one / (one + g0.neg().exp_u20());
    fVec s1 = one / (one + g1.neg().exp_u20());
    fVec r0 = o0 + s0 * p0;
    fVec r1 = o1 + s1 * p1;
    bVec out_vec = convert_from_float_ext<scalar_t>(r0, r1);
    out_vec.store(pair_orig + d);
  }
  for (; d < size; ++d) {
    float g = static_cast<float>(gate_bf[d]);
    float p = static_cast<float>(proj_bf[d]);
    float sg = 1.f / (1.f + std::exp(-g));
    float r = static_cast<float>(pair_orig[d]) + sg * p;
    pair_orig[d] = static_cast<scalar_t>(r);
  }
}

// ---------------------------------------------------------------------------
// Stage C: post-einsum.
//
// Given `centered [M, C]` (= layernormed einsum_out) and `pair_normed [M, C]`,
// computes:
//   out_proj_buf = centered    @ out_proj_w.T     [M, C]   (via weight_packed_linear_out)
//   gate_buf     = pair_normed @ gating_w.T       [M, C]   (via weight_packed_linear_out)
//   pair_orig[m, c] += sigmoid(gate_buf[m, c]) * out_proj_buf[m, c]
//
// TODO(perf-stage-c): the two [M, C] intermediates are ~5.5 GB each at N=4655.
// Replace with a per-tile fused core (two brgemms + sigmoid_mul_add per BLOCK_M
// rows, no full-tensor intermediates) once Stage A/B numbers are validated.
// Custom core needs the right `ldb` for packed bf16 weights — see the
// `weight_packed_linear_kernel_impl` reference in gemm.cpp:506 (ldb = nb_size,
// not C, because the kernel iterates the packed B in BLOCK_N tiles).
// ---------------------------------------------------------------------------

template <typename scalar_t>
void tm_post_einsum_fused_tail_impl(
    scalar_t* __restrict__ pair_orig,                    // [M, C]  in/out
    const scalar_t* __restrict__ out_proj_buf,           // [M, C]
    const scalar_t* __restrict__ gate_buf,               // [M, C]
    int64_t M,
    int C) {
  at::parallel_for(0, M, /*grain_size=*/256, [&](int64_t mb, int64_t me) {
    for (int64_t m = mb; m < me; ++m) {
      sigmoid_mul_add_inplace_bf16<scalar_t>(
          pair_orig    + m * C,
          gate_buf     + m * C,
          out_proj_buf + m * C,
          C);
    }
  });
}

}  // anonymous namespace

// Forward decl for the layernorm helper we reuse for center_norm.  Defined in
// norm.cpp and registered as sgl_kernel::layernorm_cpu; calling it as a normal
// C++ function avoids the torch dispatcher overhead per call.
void layernorm_cpu(
    at::Tensor& input,
    at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    double eps);

// ---------------------------------------------------------------------------
// Public entry.
// ---------------------------------------------------------------------------
//
//   pair_orig          : [N, N, C]   pre-LayerNorm input AND output buffer
//   pair_normed        : [N, N, C]   left_norm(pair_orig), computed in wrapper
//   mask               : [N, N]      binary {0,1} bf16
//   proj_gate_weight   : packed [4C, C] = cat(proj.W [2C,C] | gate.W [2C,C])
//   center_norm_weight : [C]
//   center_norm_bias   : optional [C]  (None allowed; layernorm_cpu handles)
//   out_proj_weight    : packed [C, C]
//   gating_weight      : packed [C, C]
//   outgoing           : true => cik,cjk->cij;  false => ckj,cki->cij
//   is_vnni            : true (we require pre-packed weights)
//
// Returns pair_orig (clobbered in place).
//
at::Tensor fused_triangle_multiplication(
    at::Tensor& pair_orig,
    at::Tensor& pair_normed,
    at::Tensor& mask,
    at::Tensor& proj_gate_weight,
    at::Tensor& center_norm_weight,
    const std::optional<at::Tensor>& center_norm_bias,
    at::Tensor& out_proj_weight,
    at::Tensor& gating_weight,
    bool outgoing,
    bool is_vnni) {
  RECORD_FUNCTION(
      "sgl_kernel::fused_triangle_multiplication",
      std::vector<c10::IValue>(
          {pair_orig, pair_normed, mask, proj_gate_weight, out_proj_weight, gating_weight, outgoing}));

  CHECK_INPUT(pair_orig);
  CHECK_INPUT(pair_normed);
  CHECK_INPUT(mask);
  CHECK_INPUT(proj_gate_weight);
  CHECK_INPUT(out_proj_weight);
  CHECK_INPUT(gating_weight);
  CHECK_INPUT(center_norm_weight);
  CHECK_DIM(3, pair_orig);
  CHECK_DIM(3, pair_normed);
  CHECK_DIM(2, mask);
  TORCH_CHECK(is_vnni, "fused_triangle_multiplication currently requires pre-packed weights (is_vnni=True).");
  // TODO(stage-b-incoming): the incoming equation needs a verified operand
  // ordering for the per-c brgemm (b.T @ a vs a @ b.T) and probably a
  // different pack variant.  Restrict to outgoing for the first pass; lift
  // once Stage B is validated end-to-end.
  TORCH_CHECK(outgoing, "fused_triangle_multiplication: incoming equation not yet supported; use the SGL drop-in for now.");

  const int N = static_cast<int>(pair_orig.size(0));
  const int N2 = static_cast<int>(pair_orig.size(1));
  const int C = static_cast<int>(pair_orig.size(2));
  TORCH_CHECK(N == N2, "pair must be square in spatial dims, got (", N, ", ", N2, ")");
  CHECK_EQ(pair_normed.size(0), N);
  CHECK_EQ(pair_normed.size(1), N);
  CHECK_EQ(pair_normed.size(2), C);
  CHECK_EQ(mask.size(0), N);
  CHECK_EQ(mask.size(1), N);

  using clock = std::chrono::high_resolution_clock;
  const bool prof = tm_profile_enabled();
  auto t_start = prof ? clock::now() : clock::time_point{};

  const int64_t M = static_cast<int64_t>(N) * N;
  const int64_t four_C = 4 * static_cast<int64_t>(C);
  constexpr int BLOCK_M = 32;

  // ----- Stage A: concat (proj | gate) GEMM, pair_normed streamed once. -----
  auto pair_normed_2d = pair_normed.view({M, static_cast<int64_t>(C)});
  auto proj_gate = at::empty({M, four_C}, pair_normed.options());
  auto t_proj_gate_alloc = prof ? clock::now() : clock::time_point{};

  weight_packed_linear_out(
      proj_gate, pair_normed_2d, proj_gate_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  auto t_proj_gate_gemm = prof ? clock::now() : clock::time_point{};

  // a, b each [C, N, N_pad] — bmm-ready inputs for Stage B.  K-padded inner
  // dim (N_pad = N rounded up to TILE_K=32) so the Stage B brgemm gets an
  // AMX-aligned K.  at::zeros pre-fills the [N, N_pad) tail with 0.
  // TODO(perf): replace at::zeros with at::empty + targeted tail-zero in
  // Stage A's parallel_for (~370 ms one-time saving at N=4655 bf16).
  const int N_pad = div_up(N, TILE_K) * TILE_K;
  auto a_tensor = at::zeros({C, N, N_pad}, pair_normed.options());
  auto b_tensor = at::zeros({C, N, N_pad}, pair_normed.options());
  auto t_ab_alloc = prof ? clock::now() : clock::time_point{};

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      pair_normed.scalar_type(), "tm_pre_einsum_scatter_impl", [&] {
        tm_pre_einsum_scatter_impl<scalar_t>(
            a_tensor.data_ptr<scalar_t>(),
            b_tensor.data_ptr<scalar_t>(),
            proj_gate.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            N,
            N_pad,
            C);
      });
  // proj_gate not needed anymore.  Drop the reference so the 22 GB blob can be
  // freed before Stage B inflates the per-thread b_vnni scratch.
  proj_gate = at::Tensor();
  auto t_stage_a = prof ? clock::now() : clock::time_point{};

  // ----- Stage B: per-c parallel bmm, output written in NHC directly. -----
  auto einsum_out = at::empty({N, N, C}, pair_normed.options());
  auto t_einsum_alloc = prof ? clock::now() : clock::time_point{};

  const int num_threads = at::get_num_threads();
  const int per_thread_bytes_b =
      /* b_vnni      */ sizeof(uint16_t) * N_pad * N_pad +
      /* logits_fp32 */ sizeof(float)    * BLOCK_M * N_pad +
      /* alignment   */ 64 * 4;
  auto buffer_b = at::empty(
      {num_threads, per_thread_bytes_b}, pair_normed.options().dtype(at::kChar));
  auto t_buf_b_alloc = prof ? clock::now() : clock::time_point{};

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      pair_normed.scalar_type(), "tm_einsum_impl", [&] {
        tm_einsum_impl<scalar_t, BLOCK_M>(
            einsum_out.data_ptr<scalar_t>(),
            a_tensor.data_ptr<scalar_t>(),
            b_tensor.data_ptr<scalar_t>(),
            N,
            N_pad,
            C,
            outgoing,
            buffer_b.data_ptr(),
            per_thread_bytes_b);
      });
  // Drop a and b — Stage C doesn't need them.
  a_tensor = at::Tensor();
  b_tensor = at::Tensor();
  auto t_stage_b = prof ? clock::now() : clock::time_point{};

  // ----- center_norm, in place on einsum_out (NHC layout, ready to consume). -----
  auto einsum_out_2d = einsum_out.view({M, static_cast<int64_t>(C)});
  layernorm_cpu(
      einsum_out_2d,
      center_norm_weight,
      center_norm_bias,
      /*eps=*/1e-5);
  auto t_center_norm = prof ? clock::now() : clock::time_point{};

  // ----- Stage C: out_proj GEMM + gating GEMM + (sigmoid_mul + residual). -----
  // For the skeleton we reuse `weight_packed_linear_out` for both GEMMs.  This
  // materializes two [M, C] intermediates (~11 GB total at N=4655 bf16) but
  // sidesteps direct `at::native::cpublas::brgemm` calls with packed weights —
  // those need the BLOCK_N-tile ldb handling that lives inside
  // weight_packed_linear_kernel_impl.  See the TODO above tm_post_einsum_*.
  auto pair_orig_2d = pair_orig.view({M, static_cast<int64_t>(C)});
  auto out_proj_buf = at::empty({M, static_cast<int64_t>(C)}, pair_normed.options());
  weight_packed_linear_out(
      out_proj_buf, einsum_out_2d, out_proj_weight,
      /*bias=*/std::nullopt, /*is_vnni=*/true);
  auto t_out_proj_gemm = prof ? clock::now() : clock::time_point{};
  // einsum_out no longer needed
  einsum_out = at::Tensor();

  auto gate_buf = at::empty({M, static_cast<int64_t>(C)}, pair_normed.options());
  weight_packed_linear_out(
      gate_buf, pair_normed_2d, gating_weight,
      /*bias=*/std::nullopt, /*is_vnni=*/true);
  auto t_gate_gemm = prof ? clock::now() : clock::time_point{};

  // Fused sigmoid(gate) * out_proj + pair_orig (in place).
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      pair_normed.scalar_type(), "tm_post_einsum_fused_tail_impl", [&] {
        tm_post_einsum_fused_tail_impl<scalar_t>(
            pair_orig_2d.data_ptr<scalar_t>(),
            out_proj_buf.data_ptr<scalar_t>(),
            gate_buf.data_ptr<scalar_t>(),
            M,
            C);
      });
  auto t_stage_c = prof ? clock::now() : clock::time_point{};

  if (prof) {
    auto ms = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(
        stderr,
        "[tm] proj_gate_alloc=%6.2f proj_gate_gemm=%7.2f ab_alloc=%6.2f stageA=%7.2f "
        "einsum_alloc=%6.2f buf_b_alloc=%6.2f stageB=%8.2f center_norm=%6.2f "
        "out_proj_gemm=%7.2f gate_gemm=%7.2f stageC_tail=%7.2f total=%8.2f ms\n",
        ms(t_start,           t_proj_gate_alloc),
        ms(t_proj_gate_alloc, t_proj_gate_gemm),
        ms(t_proj_gate_gemm,  t_ab_alloc),
        ms(t_ab_alloc,        t_stage_a),
        ms(t_stage_a,         t_einsum_alloc),
        ms(t_einsum_alloc,    t_buf_b_alloc),
        ms(t_buf_b_alloc,     t_stage_b),
        ms(t_stage_b,         t_center_norm),
        ms(t_center_norm,     t_out_proj_gemm),
        ms(t_out_proj_gemm,   t_gate_gemm),
        ms(t_gate_gemm,       t_stage_c),
        ms(t_start,           t_stage_c));
    std::fflush(stderr);
  }

  return pair_orig;
}

/*****************************************************************************************
 * Fused triangle multiplication kernel for sglang CPU backend.
 *
 *   - Stage A: one concat (proj|gate) GEMM, pair streamed once, then per-row
 *     scatter that applies mask, sigmoid(gate)*proj and splits into bmm-ready
 *     `a [C, N, N]` and `b [C, N, N]` outputs.
 *   - Stage B: at::bmm over batch=C.  Output is [C, N, N] CHW; the wrapper
 *     permutes+contigs to NHC right before center_norm.
 *   - Stage C: center_norm + out_proj GEMM + gating GEMM + sigmoid(gate) *
 *     out_proj + residual (in place into pair_orig).
 *
 * Python wrapper contract:
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

// Vectorized fp32 -> bf16/fp16 row store.  Used by Stage B to write the
// brgemm Ctmp tile into the output buffer.
template <typename scalar_t>
inline void cast_fp32_to_bf16_row(
    scalar_t* __restrict__ out, const float* __restrict__ src, int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  int d = 0;
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec a0 = fVec::loadu(src + d);
    fVec a1 = fVec::loadu(src + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(a0, a1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(src[d]);
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

// Stage A scatter.  Two perf-critical things happen here:
//
//   1) Per row m=(i, j) we apply mask * sigmoid(gate) * proj to 2C bf16
//      values.  Done with vectorized exp_u20() to avoid the scalar std::exp
//      path that would otherwise be ~50 cycles/element on AF3 shapes.
//
//   2) The result is split between a[c, i, j] and b[c, i, j] — i.e. 2C
//      cache-line-spread bf16 stores per row.  We stage BLOCK_J=32 rows of
//      a/b values in per-thread L1 scratch then bulk-memcpy each c's
//      contig BLOCK_J chunk into a_out / b_out.  Each output cache line
//      gets one full 64-byte write instead of 32 partial stores
//      (which would all trigger RFOs).
//
//   a/b are allocated [C, N, N_pad] with N_pad = round_up(N, TILE_K=32) so
//   the custom Stage B brgemm sees an AMX-aligned K.  This scatter writes
//   every (i, j) position in [0, N) and zeros the per-row tail [N, N_pad)
//   so the caller can allocate via at::empty (no pre-zero needed).
template <typename scalar_t>
void tm_pre_einsum_scatter_impl(
    scalar_t* __restrict__ a_out,                        // [C, N, N_pad]
    scalar_t* __restrict__ b_out,                        // [C, N, N_pad]
    const scalar_t* __restrict__ proj_gate,              // [M=N*N, 4C]
    const scalar_t* __restrict__ mask,                   // [N, N]
    int N,
    int N_pad,
    int C,
    void* __restrict__ buffer,
    int buffer_size_per_thread) {
  constexpr int BLOCK_J = 32;
  const int64_t row_stride = static_cast<int64_t>(N_pad);
  const int64_t plane_stride = static_cast<int64_t>(N) * row_stride;
  const int N_tail = N_pad - N;                          // 0..31
  const int64_t four_C = 4 * static_cast<int64_t>(C);
  const int twoC = 2 * C;

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec one(1.f);

  parallel_for(N, [&](int begin, int end) {
    int tid = get_thread_num();
    char* base = reinterpret_cast<char*>(buffer)
               + static_cast<size_t>(tid) * buffer_size_per_thread;
    auto bump = [&](size_t nbytes) {
      void* p = base;
      base += (nbytes + 63) & ~size_t{63};
      return p;
    };

    // [C, BLOCK_J] L1 tiles for the per-c contig chunks that get bulk-copied
    // into a_out / b_out at the end of each j-block.
    scalar_t* scratch_a =
        reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * C * BLOCK_J));
    scalar_t* scratch_b =
        reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * C * BLOCK_J));
    // [2C] tiny row tile holding mask * sigmoid(gate) * proj for one row,
    // interleaved as the GEMM produced it (a0, b0, a1, b1, ...).
    scalar_t* row_tile =
        reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * twoC));

    for (int i = begin; i < end; ++i) {
      for (int jb = 0; jb < N; jb += BLOCK_J) {
        int jb_size = std::min(BLOCK_J, N - jb);

        // Phase 1: build the [C, jb_size] scratch tiles for a and b.
        for (int jl = 0; jl < jb_size; ++jl) {
          int64_t j = jb + jl;
          int64_t m = static_cast<int64_t>(i) * N + j;
          const scalar_t* proj_row = proj_gate + m * four_C;     // [0 .. 2C)
          const scalar_t* gate_row = proj_row + twoC;            // [2C .. 4C)
          const float mask_f = static_cast<float>(mask[m]);
          const fVec mask_v(mask_f);

          // Vectorized: row_tile[d] = mask * sigmoid(gate_row[d]) * proj_row[d]
          int d = 0;
          for (; d <= twoC - kVecSize; d += kVecSize) {
            bVec p_bv = bVec::loadu(proj_row + d);
            bVec g_bv = bVec::loadu(gate_row + d);
            fVec p0, p1, g0, g1;
            std::tie(p0, p1) = at::vec::convert_to_float(p_bv);
            std::tie(g0, g1) = at::vec::convert_to_float(g_bv);
            fVec s0 = one / (one + g0.neg().exp_u20());
            fVec s1 = one / (one + g1.neg().exp_u20());
            fVec r0 = mask_v * s0 * p0;
            fVec r1 = mask_v * s1 * p1;
            bVec out = convert_from_float_ext<scalar_t>(r0, r1);
            out.store(row_tile + d);
          }
          for (; d < twoC; ++d) {
            float g = static_cast<float>(gate_row[d]);
            float p = static_cast<float>(proj_row[d]);
            float sg = 1.f / (1.f + std::exp(-g));
            row_tile[d] = static_cast<scalar_t>(mask_f * sg * p);
          }

          // De-interleave row_tile[2c]   → scratch_a[c, jl]
          //                row_tile[2c+1] → scratch_b[c, jl]
          // Scalar; happens entirely in L1 (scratch_a/b ~16 KB total).
          // TODO(perf-stage-a): replace with an AVX-512 deinterleave (mm512
          // permute + 2 stores) once the rest is validated.
          for (int c = 0; c < C; ++c) {
            scratch_a[c * BLOCK_J + jl] = row_tile[2 * c];
            scratch_b[c * BLOCK_J + jl] = row_tile[2 * c + 1];
          }
        }

        // Phase 2: bulk-copy each c's jb_size-element chunk into a_out / b_out.
        // For full BLOCK_J=32 bf16 this is 64 bytes per memcpy — one coalesced
        // cache-line write, no RFO amplification.
        const size_t chunk_bytes =
            static_cast<size_t>(jb_size) * sizeof(scalar_t);
        for (int c = 0; c < C; ++c) {
          scalar_t* a_dst = a_out
              + static_cast<int64_t>(c) * plane_stride
              + static_cast<int64_t>(i) * row_stride
              + jb;
          scalar_t* b_dst = b_out
              + static_cast<int64_t>(c) * plane_stride
              + static_cast<int64_t>(i) * row_stride
              + jb;
          std::memcpy(a_dst, scratch_a + c * BLOCK_J, chunk_bytes);
          std::memcpy(b_dst, scratch_b + c * BLOCK_J, chunk_bytes);
        }
      }

      // Tail-zero pass: zero a_out[c, i, N..N_pad) and b_out[c, i, N..N_pad)
      // so the K-padded brgemm in Stage B sees 0 contributions in the tail.
      // Tiny (~17 elems × 128 c × 2 sides ≈ 4 KB/row), runs in parallel with
      // the i-loop above.
      if (N_tail > 0) {
        const size_t tail_bytes =
            static_cast<size_t>(N_tail) * sizeof(scalar_t);
        for (int c = 0; c < C; ++c) {
          scalar_t* a_tail = a_out
              + static_cast<int64_t>(c) * plane_stride
              + static_cast<int64_t>(i) * row_stride
              + N;
          scalar_t* b_tail = b_out
              + static_cast<int64_t>(c) * plane_stride
              + static_cast<int64_t>(i) * row_stride
              + N;
          std::memset(a_tail, 0, tail_bytes);
          std::memset(b_tail, 0, tail_bytes);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Stage B: einsum core, custom per-c brgemm with j-strip parallelism.
//
// For outgoing  cik,cjk->cij :
//   out[c, i, j] = sum_k a[c, i, k] * b[c, j, k]   ==  a[c] @ b[c]^T
//
// Parallelization:
//   - Outer c loop is serial.  Within each c, a[c] (~22 MB at N=4655 bf16)
//     stays resident in the shared L3 while all 40 threads cooperate on
//     the inner j-strip parallel_for.  This is what xfold does and what
//     beats at::bmm: bmm parallelizes over c too, causing 40 threads ×
//     22 MB = 880 MB of per-thread A working sets that thrash a 105 MB L3.
//   - Inner loop is parallel over j-strips of BLOCK_N=32 cols of output.
//     Per task: VNNI-pack a 32-col strip of b into a 300 KB L2-resident
//     buffer, then tile M with brgemm(M=BLOCK_M, N=BLOCK_N, K=N_pad).
//
// K must be a multiple of TILE_K=32 for AMX brgemm, so a/b are K-padded
// to N_pad = round_up(N, 32) with the tail zeroed by Stage A.
//
// Output stays in [C, N, N] CHW so the stores are coalesced; the wrapper
// permutes+contigs to NHC right before center_norm.  In-kernel CHW->NHC
// scatter at this shape was ~10x slower than a separate contig pass (RFO
// amplification on partial cache-line writes).
// ---------------------------------------------------------------------------

template <typename scalar_t>
void tm_einsum_outgoing_impl(
    scalar_t* __restrict__ einsum_out,                   // [C, N, N]
    const scalar_t* __restrict__ a,                      // [C, N, N_pad]
    const scalar_t* __restrict__ b,                      // [C, N, N_pad]
    int N,
    int N_pad,
    int C,
    void* __restrict__ buffer,
    int buffer_size_per_thread) {
  constexpr int BLOCK_N = 32;                            // 2 * TILE_N
  constexpr int BLOCK_M = 32;                            // 2 * TILE_M
  const int NB = div_up(N, BLOCK_N);
  const int MB = div_up(N, BLOCK_M);

  const int64_t plane_in  = static_cast<int64_t>(N) * N_pad;
  const int64_t plane_out = static_cast<int64_t>(N) * N;

#if defined(_OPENMP)
  #pragma omp parallel
#endif
  {
    const int tid = get_thread_num();
    char* base = reinterpret_cast<char*>(buffer)
               + static_cast<size_t>(tid) * buffer_size_per_thread;
    auto bump = [&](size_t nbytes) {
      void* p = base;
      base += (nbytes + 63) & ~size_t{63};
      return p;
    };
    // Per-thread scratch: VNNI-packed 32-col strip of b + fp32 brgemm Ctmp.
    scalar_t* b_vnni =
        reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * N_pad * BLOCK_N));
    float* Ctmp =
        reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_M * BLOCK_N));

    for (int c = 0; c < C; ++c) {
      const scalar_t* a_c   = a + static_cast<int64_t>(c) * plane_in;
      const scalar_t* b_c   = b + static_cast<int64_t>(c) * plane_in;
      scalar_t*       out_c = einsum_out + static_cast<int64_t>(c) * plane_out;

#if defined(_OPENMP)
      #pragma omp for schedule(static)
#endif
      for (int nb = 0; nb < NB; ++nb) {
        const int j_start = nb * BLOCK_N;
        const int j_size  = std::min(BLOCK_N, N - j_start);

        // VNNI-pack b[c, j_start:j_start+j_size, :] into b_vnni.  Source is
        // row-major [j_size, N_pad] at b_c + j_start*N_pad; dst is the
        // [K=N_pad, N=BLOCK_N] VNNI layout pack_vnni expects.  ld_dst is the
        // N dim of dst (BLOCK_N).  If the strip is partial (j_size < BLOCK_N)
        // we zero the dst first so pack_vnni's untouched cols read as 0;
        // brgemm reads only the first j_size cols anyway.
        if (j_size < BLOCK_N) {
          std::memset(
              b_vnni, 0, sizeof(scalar_t) * static_cast<size_t>(N_pad) * BLOCK_N);
        }
        pack_vnni<scalar_t>(
            /*dst*/ b_vnni,
            /*src*/ b_c + static_cast<int64_t>(j_start) * N_pad,
            /*N*/ j_size,
            /*K*/ N_pad,
            /*ld_src*/ N_pad,
            /*ld_dst*/ BLOCK_N);

        // Tile over M rows of the output.
        for (int mb = 0; mb < MB; ++mb) {
          const int m_start = mb * BLOCK_M;
          const int m_size  = std::min(BLOCK_M, N - m_start);

          at::native::cpublas::brgemm(
              /*M*/ m_size, /*N*/ j_size, /*K*/ N_pad,
              /*lda*/ N_pad, /*ldb*/ BLOCK_N, /*ldc*/ BLOCK_N,
              /*add_C*/ false,
              a_c + static_cast<int64_t>(m_start) * N_pad,
              b_vnni,
              Ctmp);

          // Cast (fp32 -> bf16) and store contiguously into
          // out[c, m_start+r, j_start..j_start+j_size).
          for (int r = 0; r < m_size; ++r) {
            cast_fp32_to_bf16_row<scalar_t>(
                out_c + static_cast<int64_t>(m_start + r) * N + j_start,
                Ctmp + static_cast<int64_t>(r) * BLOCK_N,
                j_size);
          }
        }
      }
      // Implicit barrier at end of `omp for` so all threads finish this c
      // before a[c+1] is accessed.
    }

    at::native::cpublas::brgemm_release();
  }
}

// Sigmoid-mul-add helper for the fused stage C tail.  Reads gate values
// from an fp32 brgemm Ctmp tile and out_proj from a bf16 buffer; writes
// pair_orig in place:
//   pair_orig[d] = bf16( fp32(pair_orig[d]) + sigmoid(gate_fp32[d]) * fp32(out_proj[d]) )
template <typename scalar_t>
inline void vec_sigmoid_mul_add_fp32gate(
    scalar_t* __restrict__ pair_orig,
    const float* __restrict__ gate_fp32,
    const scalar_t* __restrict__ out_proj,
    int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec one(1.f);
  int d = 0;
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec g0 = fVec::loadu(gate_fp32 + d);
    fVec g1 = fVec::loadu(gate_fp32 + d + fVec::size());
    fVec s0 = one / (one + g0.neg().exp_u20());
    fVec s1 = one / (one + g1.neg().exp_u20());
    bVec p_bv = bVec::loadu(out_proj  + d);
    bVec o_bv = bVec::loadu(pair_orig + d);
    fVec p0, p1, o0, o1;
    std::tie(p0, p1) = at::vec::convert_to_float(p_bv);
    std::tie(o0, o1) = at::vec::convert_to_float(o_bv);
    fVec r0 = o0 + s0 * p0;
    fVec r1 = o1 + s1 * p1;
    bVec out_vec = convert_from_float_ext<scalar_t>(r0, r1);
    out_vec.store(pair_orig + d);
  }
  for (; d < size; ++d) {
    float s = 1.f / (1.f + std::exp(-gate_fp32[d]));
    float r = static_cast<float>(pair_orig[d]) + s * static_cast<float>(out_proj[d]);
    pair_orig[d] = static_cast<scalar_t>(r);
  }
}

// ---------------------------------------------------------------------------
// Stage C: post-einsum, fully fused per-tile.
//
// Given:
//   pair_orig    : [M, C]  (residual base; clobbered in place)
//   out_proj_buf : [M, C]  (already = centered @ out_proj_w.T)
//   pair_normed  : [M, C]  (input for the gate GEMM)
//   gating_w     : [C, C]  VNNI-packed
//
// Computes per (m_tile, n_tile) of BLOCK_M × BLOCK_N output rows/cols:
//   gate_tile_fp32 = pair_normed[m_tile] @ gating_w[n_tile].T   (brgemm into Ctmp)
//   pair_orig[m_tile, n_tile] += sigmoid(gate_tile_fp32) * out_proj_buf[m_tile, n_tile]
//
// No full-tensor [M, C] gate intermediate — Ctmp stays in registers/L1
// per tile.  Saves the gate_buf allocation (~5.5 GB at N=4655) and one
// memory-bandwidth pass over [M, C].
// ---------------------------------------------------------------------------

template <typename scalar_t>
void tm_fused_gate_sigmoid_mul_add_impl(
    scalar_t* __restrict__ pair_orig,                    // [M, C]  in/out
    const scalar_t* __restrict__ pair_normed,            // [M, C]
    const scalar_t* __restrict__ gating_w,               // [C, C]  packed VNNI
    const scalar_t* __restrict__ out_proj_buf,           // [M, C]
    int64_t M,
    int C) {
  constexpr int BLOCK_M = 32;                            // 2 * TILE_M
  constexpr int BLOCK_N = 32;                            // 2 * TILE_N
  const int K = C;                                       // gemm K = C
  const int64_t MB = div_up(M, static_cast<int64_t>(BLOCK_M));
  const int64_t NB = div_up(C, BLOCK_N);

  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    for (int64_t mb = mb0; mb < mb1; ++mb) {
      const int64_t mb_start = mb * BLOCK_M;
      const int mb_size = static_cast<int>(std::min<int64_t>(M - mb_start, BLOCK_M));
      for (int64_t nb = nb0; nb < nb1; ++nb) {
        const int64_t nb_start = nb * BLOCK_N;
        const int nb_size = static_cast<int>(std::min<int64_t>(C - nb_start, BLOCK_N));

        // brgemm: Ctmp [mb_size, nb_size] = pair_normed[mb_block, :] @ gating_w[nb_block, :].T
        // ldb = nb_size matches weight_packed_linear_kernel_impl convention
        // (the packed weight is laid out [NB, K, BLOCK_N] with each nb-block
        // contiguous in K-then-N order).
        at::native::cpublas::brgemm(
            /*M*/ mb_size, /*N*/ nb_size, /*K*/ K,
            /*lda*/ C, /*ldb*/ nb_size, /*ldc*/ BLOCK_N,
            /*add_C*/ false,
            pair_normed + mb_start * C,
            gating_w + nb_start * K,
            Ctmp);

        // Fused sigmoid(Ctmp) * out_proj_buf + pair_orig, row by row.
        for (int r = 0; r < mb_size; ++r) {
          vec_sigmoid_mul_add_fp32gate<scalar_t>(
              pair_orig + (mb_start + r) * C + nb_start,
              Ctmp + static_cast<int64_t>(r) * BLOCK_N,
              out_proj_buf + (mb_start + r) * C + nb_start,
              nb_size);
        }
      }
    }

    at::native::cpublas::brgemm_release();
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
  // TODO(stage-b-incoming): the at::bmm-based Stage B has a branch for the
  // incoming equation (bmm(b.T, a) instead of bmm(a, b.T)) but no test
  // coverage yet.  Restrict to outgoing until correctness is verified.
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

  // ----- Stage A: concat (proj | gate) GEMM, pair_normed streamed once. -----
  auto pair_normed_2d = pair_normed.view({M, static_cast<int64_t>(C)});
  auto proj_gate = at::empty({M, four_C}, pair_normed.options());
  auto t_proj_gate_alloc = prof ? clock::now() : clock::time_point{};

  weight_packed_linear_out(
      proj_gate, pair_normed_2d, proj_gate_weight, /*bias=*/std::nullopt, /*is_vnni=*/true);
  auto t_proj_gate_gemm = prof ? clock::now() : clock::time_point{};

  // a, b each [C, N, N_pad] with N_pad = round_up(N, TILE_K=32) — Stage B's
  // brgemm needs AMX-aligned K.  Stage A scatter writes every (i, j) in
  // [0, N) and zeros the [N, N_pad) tail per row, so at::empty is safe
  // (no separate memset / at::zeros).
  const int N_pad = div_up(N, TILE_K) * TILE_K;
  const int num_threads = at::get_num_threads();
  auto a_tensor = at::empty({C, N, N_pad}, pair_normed.options());
  auto b_tensor = at::empty({C, N, N_pad}, pair_normed.options());
  auto t_ab_alloc = prof ? clock::now() : clock::time_point{};

  // Per-thread L1 scratch for Stage A's tiled scatter.
  //   scratch_a / scratch_b : [C, BLOCK_J=32] bf16 (~8 KB each)
  //   row_tile              : [2C] bf16 (~256 bytes)
  // Sized to comfortably fit in L1 across all C values supported by AF3.
  constexpr int BLOCK_J_A = 32;
  const int per_thread_bytes_a =
      /* scratch_a */ sizeof(uint16_t) * C * BLOCK_J_A +
      /* scratch_b */ sizeof(uint16_t) * C * BLOCK_J_A +
      /* row_tile  */ sizeof(uint16_t) * 2 * C +
      /* alignment */ 64 * 3;
  auto buffer_a = at::empty(
      {num_threads, per_thread_bytes_a}, pair_normed.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      pair_normed.scalar_type(), "tm_pre_einsum_scatter_impl", [&] {
        tm_pre_einsum_scatter_impl<scalar_t>(
            a_tensor.data_ptr<scalar_t>(),
            b_tensor.data_ptr<scalar_t>(),
            proj_gate.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            N,
            N_pad,
            C,
            buffer_a.data_ptr(),
            per_thread_bytes_a);
      });
  // proj_gate not needed anymore.  Drop the reference so the [M, 4C] blob
  // (~11 GB at N=4655) can be freed before Stage B's output tensor allocates.
  proj_gate = at::Tensor();
  auto t_stage_a = prof ? clock::now() : clock::time_point{};

  // ----- Stage B: custom per-c brgemm with j-strip parallelism. -----
  // Per-thread scratch: VNNI-packed 32-col strip of b (~300 KB) + Ctmp.
  constexpr int BLOCK_N_B = 32;
  constexpr int BLOCK_M_B = 32;
  const int per_thread_bytes_b =
      /* b_vnni */ sizeof(uint16_t) * N_pad * BLOCK_N_B +
      /* Ctmp   */ sizeof(float)    * BLOCK_M_B * BLOCK_N_B +
      /* align  */ 64 * 3;
  auto buffer_b = at::empty(
      {num_threads, per_thread_bytes_b}, pair_normed.options().dtype(at::kChar));

  auto einsum_out_chw = at::empty({C, N, N}, pair_normed.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      pair_normed.scalar_type(), "tm_einsum_outgoing_impl", [&] {
        tm_einsum_outgoing_impl<scalar_t>(
            einsum_out_chw.data_ptr<scalar_t>(),
            a_tensor.data_ptr<scalar_t>(),
            b_tensor.data_ptr<scalar_t>(),
            N,
            N_pad,
            C,
            buffer_b.data_ptr(),
            per_thread_bytes_b);
      });
  // Drop a and b — Stage C doesn't need them.
  a_tensor = at::Tensor();
  b_tensor = at::Tensor();
  auto t_stage_b = prof ? clock::now() : clock::time_point{};

  // Permute [C, N, N] → [N, N, C] for center_norm.  bmm leaves the result
  // CHW-contiguous; the contig pass here is the same shape transpose that
  // the eager bench pays.
  // TODO(perf): replace with an in-kernel CHW->NHC transpose that fuses with
  // center_norm; saves ~5.5 GB of bf16 write traffic at N=4655.
  auto einsum_out = einsum_out_chw.permute({1, 2, 0}).contiguous();
  einsum_out_chw = at::Tensor();
  auto einsum_out_2d = einsum_out.view({M, static_cast<int64_t>(C)});
  layernorm_cpu(
      einsum_out_2d,
      center_norm_weight,
      center_norm_bias,
      /*eps=*/1e-5);
  auto t_center_norm = prof ? clock::now() : clock::time_point{};

  // ----- Stage C: out_proj GEMM + gating GEMM + (sigmoid_mul + residual). -----
  // NOTE: tried fusing the gate GEMM into a per-tile sigmoid_mul_add kernel
  // (saved gate_buf intermediate, ~50ms in its own timer) but it shifted
  // the allocator/page-fault pattern and pushed Stage B + center_norm out
  // by ~500 ms total.  tm_fused_gate_sigmoid_mul_add_impl is kept above for
  // when we have stable buffer pools; the path below is the stable variant.
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

  // sigmoid(gate_buf) * out_proj_buf + pair_orig (in place over pair_orig).
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      pair_normed.scalar_type(), "tm_post_einsum_tail_impl", [&] {
        at::parallel_for(0, M, /*grain_size=*/256, [&](int64_t mb, int64_t me) {
          for (int64_t m = mb; m < me; ++m) {
            using bVec = at::vec::Vectorized<scalar_t>;
            using fVec = at::vec::Vectorized<float>;
            constexpr int kVecSize = bVec::size();
            const fVec one(1.f);
            scalar_t* po = pair_orig_2d.data_ptr<scalar_t>() + m * C;
            const scalar_t* gb = gate_buf.data_ptr<scalar_t>() + m * C;
            const scalar_t* op = out_proj_buf.data_ptr<scalar_t>() + m * C;
            int d = 0;
            for (; d <= C - kVecSize; d += kVecSize) {
              bVec g_bv = bVec::loadu(gb + d);
              bVec p_bv = bVec::loadu(op + d);
              bVec o_bv = bVec::loadu(po + d);
              fVec g0, g1, p0, p1, o0, o1;
              std::tie(g0, g1) = at::vec::convert_to_float(g_bv);
              std::tie(p0, p1) = at::vec::convert_to_float(p_bv);
              std::tie(o0, o1) = at::vec::convert_to_float(o_bv);
              fVec s0 = one / (one + g0.neg().exp_u20());
              fVec s1 = one / (one + g1.neg().exp_u20());
              fVec r0 = o0 + s0 * p0;
              fVec r1 = o1 + s1 * p1;
              bVec out_vec = convert_from_float_ext<scalar_t>(r0, r1);
              out_vec.store(po + d);
            }
            for (; d < C; ++d) {
              float g = static_cast<float>(gb[d]);
              float p = static_cast<float>(op[d]);
              float sg = 1.f / (1.f + std::exp(-g));
              po[d] = static_cast<scalar_t>(static_cast<float>(po[d]) + sg * p);
            }
          }
        });
      });
  auto t_stage_c = prof ? clock::now() : clock::time_point{};

  if (prof) {
    auto ms = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(
        stderr,
        "[tm] proj_gate_alloc=%6.2f proj_gate_gemm=%7.2f ab_alloc=%6.2f stageA=%7.2f "
        "stageB=%8.2f center_norm=%7.2f out_proj_gemm=%7.2f gate_gemm=%7.2f "
        "stageC_tail=%7.2f total=%8.2f ms\n",
        ms(t_start,           t_proj_gate_alloc),
        ms(t_proj_gate_alloc, t_proj_gate_gemm),
        ms(t_proj_gate_gemm,  t_ab_alloc),
        ms(t_ab_alloc,        t_stage_a),
        ms(t_stage_a,         t_stage_b),
        ms(t_stage_b,         t_center_norm),
        ms(t_center_norm,     t_out_proj_gemm),
        ms(t_out_proj_gemm,   t_gate_gemm),
        ms(t_gate_gemm,       t_stage_c),
        ms(t_start,           t_stage_c));
    std::fflush(stderr);
  }

  return pair_orig;
}

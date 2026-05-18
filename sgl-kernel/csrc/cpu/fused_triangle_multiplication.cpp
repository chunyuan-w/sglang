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

// Fused Stage A: pre-einsum projection + mask + sigmoid_mul + scatter, all
// per-(i, j_block) tile.  This is the xfold pre_einsum pattern adapted for
// at::native::cpublas::brgemm; it eliminates the 22 GB `proj_gate` [M, 4C]
// intermediate that the old two-step (weight_packed_linear_out + scatter)
// path had to materialize.
//
// Per task (i, j_block of BLOCK_J=32 cols):
//   1) Brgemm-tile the [4C] output channels in BLOCK_N=32 chunks, writing
//      directly into a per-thread fp32 Ctmp_full [BLOCK_J, 4C] (~64 KB).
//      Brgemm signature is (M=BLOCK_J, N=BLOCK_N, K=C) with ldc = 4C so each
//      call lands at the right channel offset within Ctmp_full.
//   2) Per row r: mask_f = mask[i, j_block+r]; for each c in [0, C):
//        a[c, i, j_block+r] = mask_f * sigmoid(gate[r, 2c])   * proj[r, 2c]
//        b[c, i, j_block+r] = mask_f * sigmoid(gate[r, 2c+1]) * proj[r, 2c+1]
//      Computed in fp32, stored to per-thread scratch_a / scratch_b in bf16.
//   3) Bulk-copy each c's BLOCK_J-wide chunk into a_out / b_out so each
//      cache line gets one full 64-byte write (no RFO amplification).
//
// a/b are allocated [C, N, N_pad] with N_pad = round_up(N, TILE_K=32) so
// Stage B's brgemm sees an AMX-aligned K.  This kernel writes every (i, j)
// in [0, N) and zeros the per-row tail [N, N_pad); caller uses at::empty.
template <typename scalar_t>
void tm_pre_einsum_fused_impl(
    scalar_t* __restrict__ a_out,                        // [C, N, N_pad]
    scalar_t* __restrict__ b_out,                        // [C, N, N_pad]
    const scalar_t* __restrict__ pair_normed,            // [N, N, C]  contig
    const scalar_t* __restrict__ mask,                   // [N, N]      contig
    const scalar_t* __restrict__ proj_gate_weight,       // packed [4C, C]
    int N,
    int N_pad,
    int C,
    void* __restrict__ buffer,
    int buffer_size_per_thread) {
  constexpr int BLOCK_J = 32;
  constexpr int BLOCK_N = 32;                            // 2 * TILE_N
  const int64_t row_stride   = static_cast<int64_t>(N_pad);
  const int64_t plane_stride = static_cast<int64_t>(N) * row_stride;
  const int N_tail = N_pad - N;                          // 0..31
  const int twoC  = 2 * C;
  const int fourC = 4 * C;
  const int NB    = (fourC + BLOCK_N - 1) / BLOCK_N;     // total proj+gate tiles

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

    // Per-thread fp32 brgemm Ctmp [BLOCK_J, 4C] — holds proj (first 2C cols)
    // and gate (next 2C cols) as the packed weight produces.
    float* Ctmp_full =
        reinterpret_cast<float*>(bump(sizeof(float) * BLOCK_J * fourC));
    // [C, BLOCK_J] L1 tiles for the per-c contig chunks that get bulk-copied
    // into a_out / b_out at the end of each j-block.
    scalar_t* scratch_a =
        reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * C * BLOCK_J));
    scalar_t* scratch_b =
        reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * C * BLOCK_J));
    // [2C] interleaved (a0, b0, a1, b1, ...) row tile in bf16 — the vectorized
    // mask*sigmoid(gate)*proj writes here, then a scalar de-interleave splits
    // even/odd indices into scratch_a / scratch_b.
    scalar_t* row_tile =
        reinterpret_cast<scalar_t*>(bump(sizeof(scalar_t) * twoC));

    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int kVecSize = bVec::size();  // 32 bf16 per AVX-512 vec
    const fVec one(1.f);

#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
      for (int jb = 0; jb < N; jb += BLOCK_J) {
        const int jb_size = std::min(BLOCK_J, N - jb);
        const scalar_t* pn_tile = pair_normed
            + (static_cast<int64_t>(i) * N + jb) * C;     // [jb_size, C] contig

        // ----- Phase 1a: brgemm-tile the full [4C] output into Ctmp_full.
        for (int nb_start = 0; nb_start < fourC; nb_start += BLOCK_N) {
          at::native::cpublas::brgemm(
              /*M*/ jb_size, /*N*/ BLOCK_N, /*K*/ C,
              /*lda*/ C, /*ldb*/ BLOCK_N, /*ldc*/ fourC,
              /*add_C*/ false,
              pn_tile,
              proj_gate_weight + static_cast<int64_t>(nb_start) * C,
              Ctmp_full + nb_start);
        }
        (void)NB;  // NB derived for clarity only

        // ----- Phase 1b: apply mask * sigmoid(gate) * proj per row and
        //                de-interleave into scratch_a / scratch_b.
        // Vectorized AVX-512 exp_u20 (~5 cycles/elem) — scalar std::exp would
        // be ~50 cycles/elem × 2C × jb_size × N²/BLOCK_J ≈ 5 B calls/run.
        for (int r = 0; r < jb_size; ++r) {
          const float mask_f =
              static_cast<float>(mask[static_cast<int64_t>(i) * N + jb + r]);
          const fVec mask_v(mask_f);
          const float* proj_row = Ctmp_full + static_cast<int64_t>(r) * fourC;
          const float* gate_row = proj_row + twoC;

          int d = 0;
          for (; d <= twoC - kVecSize; d += kVecSize) {
            fVec p0 = fVec::loadu(proj_row + d);
            fVec p1 = fVec::loadu(proj_row + d + fVec::size());
            fVec g0 = fVec::loadu(gate_row + d);
            fVec g1 = fVec::loadu(gate_row + d + fVec::size());
            fVec s0 = one / (one + g0.neg().exp_u20());
            fVec s1 = one / (one + g1.neg().exp_u20());
            fVec r0 = mask_v * s0 * p0;
            fVec r1 = mask_v * s1 * p1;
            bVec out_vec = convert_from_float_ext<scalar_t>(r0, r1);
            out_vec.store(row_tile + d);
          }
          for (; d < twoC; ++d) {
            float p = proj_row[d];
            float g = gate_row[d];
            float sg = 1.f / (1.f + std::exp(-g));
            row_tile[d] = static_cast<scalar_t>(mask_f * sg * p);
          }

          // De-interleave row_tile[2c]   → scratch_a[c, r]
          //                row_tile[2c+1] → scratch_b[c, r]
          // Scalar; happens entirely in L1 (scratch_a/b ~16 KB combined).
          for (int c = 0; c < C; ++c) {
            scratch_a[c * BLOCK_J + r] = row_tile[2 * c];
            scratch_b[c * BLOCK_J + r] = row_tile[2 * c + 1];
          }
        }

        // ----- Phase 2: bulk-copy each c's jb_size-element chunk into a_out / b_out.
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

      // ----- Tail-zero: zero a_out[c, i, N..N_pad) / b_out[c, i, N..N_pad).
      // Stage B's K-padded brgemm reads these as 0 contributions.
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

    at::native::cpublas::brgemm_release();
  }
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

  // ----- Stage A: fused projection + mask + sigmoid_mul + scatter. -----
  // proj_gate is NOT materialized as a full [M, 4C] tensor (~22 GB at N=4655);
  // it's computed per-(i, j_block) tile inside tm_pre_einsum_fused_impl and
  // immediately consumed.  This matches xfold's pre_einsum and is what makes
  // the kernel allocator-pattern-stable on multi-socket targets (the 22 GB
  // intermediate was the dominant per-call transient driving the alternating
  // fast/slow runs we observed).
  auto pair_normed_2d = pair_normed.view({M, static_cast<int64_t>(C)});
  auto t_proj_gate_alloc = prof ? clock::now() : clock::time_point{};
  auto t_proj_gate_gemm  = t_proj_gate_alloc;          // folded into stageA

  // a, b each [C, N, N_pad] with N_pad = round_up(N, TILE_K=32) — Stage B's
  // brgemm needs AMX-aligned K.  Stage A writes every (i, j) in [0, N) and
  // zeros the [N, N_pad) tail per row, so at::empty is safe (no pre-fill).
  const int N_pad = div_up(N, TILE_K) * TILE_K;
  const int num_threads = at::get_num_threads();
  auto a_tensor = at::empty({C, N, N_pad}, pair_normed.options());
  auto b_tensor = at::empty({C, N, N_pad}, pair_normed.options());
  auto t_ab_alloc = prof ? clock::now() : clock::time_point{};

  // Per-thread scratch for the fused stage A:
  //   Ctmp_full : [BLOCK_J=32, 4C] fp32  (32*512*4 = 64 KB at C=128)
  //   scratch_a / scratch_b : [C, BLOCK_J] bf16  (~8 KB each)
  //   row_tile  : [2C] bf16  (~512 bytes)
  constexpr int BLOCK_J_A = 32;
  const int per_thread_bytes_a =
      /* Ctmp_full */ sizeof(float)    * BLOCK_J_A * (4 * C) +
      /* scratch_a */ sizeof(uint16_t) * C * BLOCK_J_A +
      /* scratch_b */ sizeof(uint16_t) * C * BLOCK_J_A +
      /* row_tile  */ sizeof(uint16_t) * 2 * C +
      /* alignment */ 64 * 4;
  auto buffer_a = at::empty(
      {num_threads, per_thread_bytes_a}, pair_normed.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      pair_normed.scalar_type(), "tm_pre_einsum_fused_impl", [&] {
        tm_pre_einsum_fused_impl<scalar_t>(
            a_tensor.data_ptr<scalar_t>(),
            b_tensor.data_ptr<scalar_t>(),
            pair_normed.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            proj_gate_weight.data_ptr<scalar_t>(),
            N,
            N_pad,
            C,
            buffer_a.data_ptr(),
            per_thread_bytes_a);
      });
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

  // ----- Stage C: out_proj GEMM + fused (gating GEMM | sigmoid_mul | residual). -----
  // The gate GEMM is folded into the per-tile fused kernel so we don't
  // materialize a [M, C] gate intermediate (~5.5 GB at N=4655).  Combined
  // with the fused Stage A above, total per-call transient drops from
  // ~55 GB to ~17 GB (matching xfold), which is the size at which the
  // allocator stops alternating slot assignments on multi-socket targets.
  auto pair_orig_2d = pair_orig.view({M, static_cast<int64_t>(C)});
  auto out_proj_buf = at::empty({M, static_cast<int64_t>(C)}, pair_normed.options());
  weight_packed_linear_out(
      out_proj_buf, einsum_out_2d, out_proj_weight,
      /*bias=*/std::nullopt, /*is_vnni=*/true);
  auto t_out_proj_gemm = prof ? clock::now() : clock::time_point{};
  // einsum_out no longer needed.
  einsum_out = at::Tensor();

  // Fused per-tile: brgemm(pair_normed, gating_w) -> sigmoid -> mul out_proj_buf
  // -> add into pair_orig (all in place).
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      pair_normed.scalar_type(), "tm_fused_gate_sigmoid_mul_add_impl", [&] {
        tm_fused_gate_sigmoid_mul_add_impl<scalar_t>(
            pair_orig_2d.data_ptr<scalar_t>(),
            pair_normed_2d.data_ptr<scalar_t>(),
            gating_weight.data_ptr<scalar_t>(),
            out_proj_buf.data_ptr<scalar_t>(),
            M,
            C);
      });
  auto t_gate_gemm = prof ? clock::now() : clock::time_point{};
  auto t_stage_c = t_gate_gemm;

  if (prof) {
    auto ms = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(
        stderr,
        "[tm] ab_alloc=%6.2f stageA=%7.2f stageB=%8.2f center_norm=%7.2f "
        "out_proj_gemm=%7.2f gate_fused_tail=%7.2f total=%8.2f ms\n",
        ms(t_proj_gate_gemm,  t_ab_alloc),
        ms(t_ab_alloc,        t_stage_a),
        ms(t_stage_a,         t_stage_b),
        ms(t_stage_b,         t_center_norm),
        ms(t_center_norm,     t_out_proj_gemm),
        ms(t_out_proj_gemm,   t_gate_gemm),
        ms(t_start,           t_stage_c));
    std::fflush(stderr);
  }

  return pair_orig;
}

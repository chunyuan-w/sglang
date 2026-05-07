// CPU kernel for topk_transform_512: select top-K positions per batch from a
// score row, then translate (page_idx, offset) into a physical (phys_page,
// offset) using each batch's page_table. Mirrors the semantics of
// ``topk_transform_512_pytorch_vectorized`` in
// python/sglang/srt/layers/attention/compressed/indexer.py.
//
// Behavior:
//   - K is fixed at 512 (matches the Python reference / consumer).
//   - If seq_lens[b] <= K, the row is filled with sequential indices
//     [0, 1, ..., seq_lens[b] - 1] and -1 padding (no top-K needed).
//   - Otherwise the kernel selects the K largest finite scores from
//     scores[b, :seq_lens[b]] using a min-heap; entries equal to -inf are
//     skipped (mirrors Python's ``valid_topk = gathered_scores != -inf``).
//   - Each chosen raw index r is translated:
//         phys = page_tables[b, r >> page_bits]
//         out  = (phys << page_bits) | (r & page_mask)
//     and written to out_page_indices[b, k]; -1 fills any remaining slots.
//   - When out_raw_indices is provided, the same selection populates it
//     with raw r values (or -1 for invalid slots).
//
// Output ordering for the top-K branch is deterministic: descending by
// score, ties broken by ascending raw index. The downstream attention
// kernel treats these as an unordered set, but a stable order makes
// testing tractable.

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"

namespace {

constexpr int64_t TOPK_TRANSFORM_K = 512;

inline int64_t int_log2_pow2(int64_t n) {
  // Caller asserts n is a positive power of two.
  int64_t r = 0;
  while ((int64_t(1) << r) < n) ++r;
  return r;
}

}  // namespace

void topk_transform_512_cpu(
    at::Tensor scores,                                  // [B, S] float32
    at::Tensor seq_lens,                                // [B]    int32 or int64
    at::Tensor page_tables,                             // [B, max_pages] int32
    at::Tensor out_page_indices,                        // [B, K] int32 (out)
    int64_t page_size,                                  // power of 2
    std::optional<at::Tensor> out_raw_indices) {        // [B, K] int32 (out, optional)
  RECORD_FUNCTION("sgl-kernel::topk_transform_512_cpu",
                  std::vector<c10::IValue>({scores, seq_lens, page_tables}));

  TORCH_CHECK(scores.device().type() == at::kCPU,
              "topk_transform_512_cpu: scores must be on CPU.");
  CHECK_INPUT(scores);
  CHECK_DIM(2, scores);
  TORCH_CHECK(scores.scalar_type() == at::kFloat,
              "topk_transform_512_cpu: scores must be float32.");

  CHECK_DIM(1, seq_lens);
  CHECK_EQ(seq_lens.size(0), scores.size(0));
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt || seq_lens.scalar_type() == at::kLong,
      "topk_transform_512_cpu: seq_lens must be int32 or int64.");

  CHECK_DIM(2, page_tables);
  CHECK_EQ(page_tables.size(0), scores.size(0));
  CHECK_INPUT(page_tables);
  TORCH_CHECK(page_tables.scalar_type() == at::kInt,
              "topk_transform_512_cpu: page_tables must be int32.");

  CHECK_DIM(2, out_page_indices);
  CHECK_EQ(out_page_indices.size(0), scores.size(0));
  CHECK_EQ(out_page_indices.size(1), TOPK_TRANSFORM_K);
  CHECK_INPUT(out_page_indices);
  TORCH_CHECK(out_page_indices.scalar_type() == at::kInt,
              "topk_transform_512_cpu: out_page_indices must be int32.");

  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0,
              "topk_transform_512_cpu: page_size must be a positive power of 2, got ", page_size);

  const bool has_raw_out = out_raw_indices.has_value();
  if (has_raw_out) {
    const auto& t = out_raw_indices.value();
    CHECK_DIM(2, t);
    CHECK_EQ(t.size(0), scores.size(0));
    CHECK_EQ(t.size(1), TOPK_TRANSFORM_K);
    CHECK_INPUT(t);
    TORCH_CHECK(t.scalar_type() == at::kInt,
                "topk_transform_512_cpu: out_raw_indices must be int32.");
  }

  const int64_t B = scores.size(0);
  const int64_t S = scores.size(1);
  const int64_t max_pages = page_tables.size(1);
  const int64_t page_bits = int_log2_pow2(page_size);
  const int64_t page_mask = page_size - 1;

  const float* __restrict__ scores_ptr = scores.data_ptr<float>();
  const int32_t* __restrict__ pages_ptr = page_tables.data_ptr<int32_t>();
  int32_t* __restrict__ out_page_ptr = out_page_indices.data_ptr<int32_t>();
  int32_t* __restrict__ out_raw_ptr =
      has_raw_out ? out_raw_indices.value().data_ptr<int32_t>() : nullptr;

  // Read seq_lens once (B is small; cost is negligible vs. the per-batch work).
  std::vector<int64_t> seq_lens_vec(B);
  if (seq_lens.scalar_type() == at::kInt) {
    const int32_t* p = seq_lens.data_ptr<int32_t>();
    for (int64_t i = 0; i < B; ++i) seq_lens_vec[i] = p[i];
  } else {
    const int64_t* p = seq_lens.data_ptr<int64_t>();
    for (int64_t i = 0; i < B; ++i) seq_lens_vec[i] = p[i];
  }

  at::parallel_for(0, B, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
    // Per-thread heap buffer reused across batches in [begin, end).
    std::vector<int32_t> heap_idx;

    for (int64_t b = begin; b < end; ++b) {
      const int64_t seq_len = std::min<int64_t>(seq_lens_vec[b], S);
      const float* row_scores = scores_ptr + b * S;
      const int32_t* row_pages = pages_ptr + b * max_pages;
      int32_t* row_out_page = out_page_ptr + b * TOPK_TRANSFORM_K;
      int32_t* row_out_raw = out_raw_ptr ? out_raw_ptr + b * TOPK_TRANSFORM_K : nullptr;

      auto translate = [&](int32_t r) -> int32_t {
        if (r < 0) return -1;
        const int64_t page_idx = static_cast<int64_t>(r) >> page_bits;
        const int64_t offset_in_page = static_cast<int64_t>(r) & page_mask;
        if (page_idx >= max_pages) return -1;  // OOB safety
        const int64_t phys = row_pages[page_idx];
        return static_cast<int32_t>((phys << page_bits) | offset_in_page);
      };

      auto write_invalid = [&](int64_t k_begin) {
        for (int64_t k = k_begin; k < TOPK_TRANSFORM_K; ++k) {
          row_out_page[k] = -1;
          if (row_out_raw) row_out_raw[k] = -1;
        }
      };

      if (seq_len <= TOPK_TRANSFORM_K) {
        // Fast path: dense [0, seq_len) followed by -1 padding.
        for (int64_t k = 0; k < seq_len; ++k) {
          const int32_t r = static_cast<int32_t>(k);
          row_out_page[k] = translate(r);
          if (row_out_raw) row_out_raw[k] = r;
        }
        write_invalid(seq_len);
        continue;
      }

      // True top-K via min-heap of indices, ordered by row_scores.
      // cmp_min returns true when score(a) > score(b) so the heap "max" by
      // cmp_min is the element with the smallest score (= eviction
      // candidate at the heap front).
      auto cmp_min = [&](int32_t a, int32_t b) {
        return row_scores[a] > row_scores[b];
      };

      heap_idx.clear();
      heap_idx.reserve(TOPK_TRANSFORM_K);

      // Fill the heap with the first up-to-K finite scores.
      int64_t i = 0;
      while ((int64_t)heap_idx.size() < TOPK_TRANSFORM_K && i < seq_len) {
        const float s = row_scores[i];
        // Skip -inf and NaN; mirrors Python's `gathered_scores != -inf` filter.
        // (Any non-finite negative value is treated as masked-out.)
        if (std::isfinite(s) || s == std::numeric_limits<float>::infinity()) {
          heap_idx.push_back(static_cast<int32_t>(i));
        }
        ++i;
      }
      std::make_heap(heap_idx.begin(), heap_idx.end(), cmp_min);

      // Stream the rest, replacing the heap min when a larger score arrives.
      // Strict greater-than: ties never displace, so earliest-index wins on ties
      // (matches a deterministic interpretation of torch.topk(sorted=False)).
      for (; i < seq_len; ++i) {
        const float s = row_scores[i];
        if (!(std::isfinite(s) || s == std::numeric_limits<float>::infinity())) {
          continue;
        }
        if ((int64_t)heap_idx.size() < TOPK_TRANSFORM_K) {
          heap_idx.push_back(static_cast<int32_t>(i));
          std::push_heap(heap_idx.begin(), heap_idx.end(), cmp_min);
          continue;
        }
        if (s > row_scores[heap_idx.front()]) {
          std::pop_heap(heap_idx.begin(), heap_idx.end(), cmp_min);
          heap_idx.back() = static_cast<int32_t>(i);
          std::push_heap(heap_idx.begin(), heap_idx.end(), cmp_min);
        }
      }

      // Stable, deterministic output: descending by score, ties by ascending index.
      std::sort(heap_idx.begin(), heap_idx.end(),
                [&](int32_t a, int32_t b) {
                  const float sa = row_scores[a];
                  const float sb = row_scores[b];
                  if (sa != sb) return sa > sb;
                  return a < b;
                });

      const int64_t n = static_cast<int64_t>(heap_idx.size());
      for (int64_t k = 0; k < n; ++k) {
        const int32_t r = heap_idx[k];
        row_out_page[k] = translate(r);
        if (row_out_raw) row_out_raw[k] = r;
      }
      write_invalid(n);
    }
  });
}

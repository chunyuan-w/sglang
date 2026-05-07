"""Unit tests for ``torch.ops.sgl_kernel.topk_transform_512_cpu``.

Compares the CPU kernel against the pure-torch reference
``topk_transform_512_pytorch_vectorized`` from
``sglang/srt/layers/attention/compressed/indexer.py``.
"""

import itertools
import unittest

import torch

from sglang.srt.layers.attention.compressed.indexer import (
    topk_transform_512_pytorch_vectorized,
)
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(0)

TOPK = 512


def _run_reference(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    page_size: int,
    want_raw: bool = False,
):
    """Run the python-vectorized reference; we monkey-route around the
    CPU short-circuit by force-cloning to a non-CPU-named device... but the
    reference logic is platform-agnostic, so we instead temporarily bypass
    the CPU dispatch by calling the body directly via ``__wrapped__``.

    Easiest: just inline the reference body here so the test does not
    depend on whether the indexer module routes CPU calls to the kernel.
    """
    B = scores.shape[0]
    out_page = torch.empty((B, TOPK), dtype=torch.int32)
    out_raw = torch.empty((B, TOPK), dtype=torch.int32) if want_raw else None
    _reference_body(scores, seq_lens, page_tables, out_page, page_size, out_raw)
    return out_page, out_raw


def _reference_body(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices=None,
):
    # Inline copy of the python reference's algorithm (skips the CPU
    # short-circuit that now dispatches to the kernel under test).
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores[~valid_mask] = float("-inf")

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        padding = torch.zeros(
            (batch_size, TOPK - actual_k), dtype=torch.int32, device=device
        )
        raw_indices = torch.cat([raw_indices, padding], dim=1)

    batch_indices = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, TOPK)
    )
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = torch.arange(TOPK, device=device).unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    if needs_sequential.any():
        sequential_indices = (
            torch.arange(TOPK, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        sequential_valid = sequential_indices < seq_lens.unsqueeze(1)
        raw_indices = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK),
            torch.where(
                sequential_valid,
                sequential_indices,
                torch.tensor(-1, device=device, dtype=torch.int32),
            ),
            raw_indices,
        )
        valid_topk = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK), sequential_valid, valid_topk
        )

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)
    page_indices = torch.where(
        valid_topk, page_indices, torch.tensor(-1, device=device, dtype=torch.int32)
    )
    out_page_indices.copy_(page_indices)
    if out_raw_indices is not None:
        out_raw_indices.copy_(
            torch.where(
                valid_topk,
                raw_indices,
                torch.tensor(-1, device=device, dtype=torch.int32),
            )
        )


def _per_row_set(t: torch.Tensor):
    """Return per-row sorted unique values (ignoring -1 padding) so
    order-sensitive comparisons can be turned into set comparisons."""
    rows = []
    for row in t.tolist():
        s = sorted(v for v in row if v != -1)
        rows.append(s)
    return rows


def _make_inputs(
    batch_size: int,
    max_seq_len: int,
    seq_lens_values,
    page_size: int,
    max_pages: int,
    seed: int,
    inject_neg_inf: bool = False,
):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(batch_size, max_seq_len, generator=g, dtype=torch.float32)
    if inject_neg_inf and max_seq_len > 8:
        # Mark a handful of legitimate positions as -inf to exercise the
        # `gathered_scores != -inf` masking.
        scores[0, 1] = float("-inf")
        scores[0, 5] = float("-inf")
        if batch_size > 1:
            scores[1, 0] = float("-inf")
    seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32)
    page_tables = torch.randint(
        0, 4096, (batch_size, max_pages), generator=g, dtype=torch.int32
    )
    return scores, seq_lens, page_tables


class TestTopkTransform512Cpu(CustomTestCase):

    page_sizes = [1, 64, 128]

    def _check_match(self, scores, seq_lens, page_tables, page_size):
        B = scores.shape[0]
        out_page = torch.full((B, TOPK), 0, dtype=torch.int32)
        out_raw = torch.full((B, TOPK), 0, dtype=torch.int32)

        torch.ops.sgl_kernel.topk_transform_512_cpu(
            scores, seq_lens, page_tables, out_page, page_size, out_raw
        )

        ref_page, ref_raw = _run_reference(
            scores, seq_lens, page_tables, page_size, want_raw=True
        )

        # Sequential mode (seq_len <= TOPK): the reference picks
        # [0, 1, ..., seq_len-1, -1, -1, ...]. Both are deterministic and
        # order-sensitive; compare directly.
        # Top-K mode (seq_len > TOPK): both algorithms select the K largest
        # finite scores but neither guarantees a particular order, so we
        # compare per-row sets of valid entries plus the -1 pad count.
        for b in range(B):
            sl = int(seq_lens[b].item())
            if sl <= TOPK:
                torch.testing.assert_close(
                    out_raw[b], ref_raw[b], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    out_page[b], ref_page[b], rtol=0, atol=0
                )
            else:
                # Set-equality on raw indices.
                self.assertEqual(
                    sorted(out_raw[b].tolist()),
                    sorted(ref_raw[b].tolist()),
                    msg=f"raw set mismatch batch={b}",
                )
                # Set-equality on translated page indices.
                self.assertEqual(
                    sorted(out_page[b].tolist()),
                    sorted(ref_page[b].tolist()),
                    msg=f"page set mismatch batch={b}",
                )
                # Padding count must match.
                self.assertEqual(
                    (out_raw[b] == -1).sum().item(),
                    (ref_raw[b] == -1).sum().item(),
                )

    @staticmethod
    def _max_pages_for(seq_max: int, page_size: int) -> int:
        # Need at least ceil(seq_max / page_size) pages; pad a few extra.
        return max(8, (seq_max + page_size - 1) // page_size + 4)

    def test_sequential_short(self):
        # All batches in sequential mode (seq_len <= 512).
        for page_size in self.page_sizes:
            with self.subTest(page_size=page_size):
                S = 300
                scores, seq_lens, page_tables = _make_inputs(
                    batch_size=3,
                    max_seq_len=S,
                    seq_lens_values=[64, S, 1],
                    page_size=page_size,
                    max_pages=self._max_pages_for(S, page_size),
                    seed=1,
                )
                self._check_match(scores, seq_lens, page_tables, page_size)

    def test_sequential_at_boundary(self):
        # seq_len exactly equals TOPK=512: still sequential path.
        for page_size in self.page_sizes:
            with self.subTest(page_size=page_size):
                scores, seq_lens, page_tables = _make_inputs(
                    batch_size=2,
                    max_seq_len=TOPK,
                    seq_lens_values=[TOPK, TOPK - 1],
                    page_size=page_size,
                    max_pages=self._max_pages_for(TOPK, page_size),
                    seed=2,
                )
                self._check_match(scores, seq_lens, page_tables, page_size)

    def test_topk_path(self):
        # seq_len > TOPK: real top-K dispatch. Use larger page_sizes so the
        # python reference's `gather` does not blow up on page_size=1.
        for page_size in [64, 128]:
            with self.subTest(page_size=page_size):
                S = 2048
                scores, seq_lens, page_tables = _make_inputs(
                    batch_size=4,
                    max_seq_len=S,
                    seq_lens_values=[S, 1024, S - 7, 600],
                    page_size=page_size,
                    max_pages=self._max_pages_for(S, page_size),
                    seed=3,
                )
                self._check_match(scores, seq_lens, page_tables, page_size)

    def test_mixed_modes(self):
        # Some batches sequential, some top-K.
        page_size = 64
        S = 4096
        scores, seq_lens, page_tables = _make_inputs(
            batch_size=5,
            max_seq_len=S,
            seq_lens_values=[100, S, 512, S - 1, 0],
            page_size=page_size,
            max_pages=S // page_size + 4,
            seed=4,
        )
        self._check_match(scores, seq_lens, page_tables, page_size)

    def test_neg_inf_scores(self):
        # Legitimate -inf scores in valid positions must be filtered out.
        page_size = 64
        S = 1024
        scores, seq_lens, page_tables = _make_inputs(
            batch_size=2,
            max_seq_len=S,
            seq_lens_values=[S, S],
            page_size=page_size,
            max_pages=S // page_size + 4,
            seed=5,
            inject_neg_inf=True,
        )
        self._check_match(scores, seq_lens, page_tables, page_size)

    def test_seq_lens_int64(self):
        # int64 seq_lens must also work.
        page_size = 64
        S = 800
        scores, seq_lens, page_tables = _make_inputs(
            batch_size=3,
            max_seq_len=S,
            seq_lens_values=[S, 700, 32],
            page_size=page_size,
            max_pages=S // page_size + 4,
            seed=6,
        )
        seq_lens = seq_lens.to(torch.int64)
        self._check_match(scores, seq_lens, page_tables, page_size)

    def test_without_out_raw(self):
        # out_raw_indices is optional; verify the kernel runs without it.
        page_size = 128
        S = 256
        scores, seq_lens, page_tables = _make_inputs(
            batch_size=2,
            max_seq_len=S,
            seq_lens_values=[256, 100],
            page_size=page_size,
            max_pages=S // page_size + 2,
            seed=7,
        )
        out_page = torch.full((2, TOPK), 0, dtype=torch.int32)
        torch.ops.sgl_kernel.topk_transform_512_cpu(
            scores, seq_lens, page_tables, out_page, page_size, None
        )
        ref_page, _ = _run_reference(
            scores, seq_lens, page_tables, page_size, want_raw=False
        )
        # All sequential here, so compare directly.
        torch.testing.assert_close(out_page, ref_page, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

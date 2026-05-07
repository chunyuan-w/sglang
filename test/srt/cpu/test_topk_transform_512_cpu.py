import unittest

import torch

import sgl_kernel  # noqa: F401
from sglang.srt.layers.attention.compressed.indexer import (
    topk_transform_512_pytorch_vectorized,
)
from sglang.test.test_utils import CustomTestCase


TOPK = 512


class TestTopKTransform512CPU(CustomTestCase):
    def _make_inputs(
        self, batch_size, max_seq_len, page_size, seq_lens, page_dtype=torch.int32
    ):
        torch.manual_seed(2026)
        scores = torch.randn(batch_size, max_seq_len, dtype=torch.float32)
        scores += torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(0) * 1e-6
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        num_pages = (max_seq_len + page_size - 1) // page_size
        page_tables = (
            torch.arange(batch_size * num_pages, dtype=page_dtype).reshape(
                batch_size, num_pages
            )
            + 17
        ).contiguous()
        return scores.contiguous(), seq_lens, page_tables

    def _run_pytorch_reference(self, scores, seq_lens, page_tables, page_size):
        ref_page_indices = torch.empty((scores.size(0), TOPK), dtype=torch.int32)
        ref_raw_indices = torch.empty_like(ref_page_indices)
        topk_transform_512_pytorch_vectorized(
            scores,
            seq_lens,
            page_tables,
            ref_page_indices,
            page_size,
            ref_raw_indices,
        )
        return ref_page_indices, ref_raw_indices

    def _assert_rows_equivalent(self, actual, expected, seq_lens):
        for row, seq_len in enumerate(seq_lens.tolist()):
            if seq_len <= TOPK:
                torch.testing.assert_close(actual[row], expected[row], atol=0, rtol=0)
            else:
                actual_valid = actual[row][actual[row] >= 0].sort().values
                expected_valid = expected[row][expected[row] >= 0].sort().values
                torch.testing.assert_close(actual_valid, expected_valid, atol=0, rtol=0)
                self.assertTrue((actual[row][actual[row] < 0] == -1).all())

    def test_matches_reference_with_raw_indices(self):
        page_size = 64
        scores, seq_lens, page_tables = self._make_inputs(
            batch_size=4,
            max_seq_len=1536,
            page_size=page_size,
            seq_lens=[0, 17, 512, 1299],
        )
        out_page_indices = torch.empty((scores.size(0), TOPK), dtype=torch.int32)
        out_raw_indices = torch.empty_like(out_page_indices)

        torch.ops.sgl_kernel.topk_transform_512_cpu(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            out_raw_indices,
        )
        ref_page_indices, ref_raw_indices = self._run_pytorch_reference(
            scores, seq_lens, page_tables, page_size
        )

        self._assert_rows_equivalent(out_raw_indices, ref_raw_indices, seq_lens)
        self._assert_rows_equivalent(out_page_indices, ref_page_indices, seq_lens)

    def test_matches_reference_without_raw_indices(self):
        page_size = 32
        scores, seq_lens, page_tables = self._make_inputs(
            batch_size=3,
            max_seq_len=777,
            page_size=page_size,
            seq_lens=[1, 600, 777],
            page_dtype=torch.int64,
        )
        out_page_indices = torch.empty((scores.size(0), TOPK), dtype=torch.int32)

        torch.ops.sgl_kernel.topk_transform_512_cpu(
            scores,
            seq_lens.to(torch.int64),
            page_tables,
            out_page_indices,
            page_size,
            None,
        )
        ref_page_indices, _ = self._run_pytorch_reference(
            scores, seq_lens, page_tables, page_size
        )

        self._assert_rows_equivalent(out_page_indices, ref_page_indices, seq_lens)


if __name__ == "__main__":
    unittest.main()

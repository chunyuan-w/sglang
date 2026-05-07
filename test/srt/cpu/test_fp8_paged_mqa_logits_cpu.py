import unittest

import torch

import sgl_kernel  # noqa: F401
from sglang.srt.layers.attention.compressed.indexer import (
    fp8_paged_mqa_logits_torch,
)
from sglang.test.test_utils import CustomTestCase


BLOCK_SIZE = 64
HEAD_DIM = 128
HEAD_DIM_WITH_SCALE_BYTES = 132


class TestFp8PagedMqaLogitsCPU(CustomTestCase):
    def _make_inputs(
        self,
        *,
        batch_size: int = 3,
        num_heads: int = 4,
        max_seq_len: int = 192,
        num_blocks: int = 8,
        index_dtype: torch.dtype = torch.int32,
        weight_dtype: torch.dtype = torch.float32,
    ):
        torch.manual_seed(0)

        q = torch.randn(batch_size, 1, num_heads, HEAD_DIM, dtype=torch.float32) * 0.25
        q_fp8 = q.to(torch.float8_e4m3fn).contiguous()

        k = torch.randn(num_blocks, BLOCK_SIZE, HEAD_DIM, dtype=torch.float32) * 0.25
        k_fp8 = k.to(torch.float8_e4m3fn).contiguous()
        k_bytes = k_fp8.view(num_blocks, BLOCK_SIZE * HEAD_DIM).view(dtype=torch.uint8)

        scales = torch.rand(num_blocks, BLOCK_SIZE, dtype=torch.float32) * 0.5 + 0.75
        scale_bytes = scales.contiguous().view(num_blocks, BLOCK_SIZE).view(dtype=torch.uint8)

        kvcache = torch.cat([k_bytes, scale_bytes], dim=1).contiguous()
        kvcache = kvcache.view(num_blocks, BLOCK_SIZE, 1, HEAD_DIM_WITH_SCALE_BYTES)

        weight = torch.randn(batch_size, num_heads, dtype=torch.float32).to(weight_dtype).contiguous()
        seq_lens = torch.tensor([0, 65, max_seq_len - 1], dtype=index_dtype)

        pages_per_batch = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        page_table = torch.empty(batch_size, pages_per_batch, dtype=index_dtype)
        page_table[0] = torch.tensor([0, 1, 2], dtype=index_dtype)
        page_table[1] = torch.tensor([3, 4, 5], dtype=index_dtype)
        page_table[2] = torch.tensor([2, 6, 7], dtype=index_dtype)

        return q_fp8, kvcache, weight, seq_lens, page_table, max_seq_len

    def _assert_matches_reference(self, index_dtype: torch.dtype, weight_dtype: torch.dtype):
        q_fp8, kvcache, weight, seq_lens, page_table, max_seq_len = self._make_inputs(
            index_dtype=index_dtype,
            weight_dtype=weight_dtype,
        )

        actual = torch.ops.sgl_kernel.fp8_paged_mqa_logits_cpu(
            q_fp8,
            kvcache,
            weight,
            seq_lens,
            page_table,
            max_seq_len,
            False,
        )
        expected = fp8_paged_mqa_logits_torch(
            q_fp8,
            kvcache,
            weight,
            seq_lens,
            page_table,
            None,
            max_seq_len,
            False,
        )

        self.assertEqual(actual.shape, (seq_lens.numel(), max_seq_len))
        self.assertEqual(actual.dtype, torch.float32)
        for batch_idx, seq_len in enumerate(seq_lens.tolist()):
            if seq_len == 0:
                continue
            torch.testing.assert_close(
                actual[batch_idx, :seq_len],
                expected[batch_idx, :seq_len],
                rtol=1e-5,
                atol=1e-5,
            )

    @unittest.skipIf(
        not hasattr(torch, "float8_e4m3fn"), "torch.float8_e4m3fn is unavailable"
    )
    def test_matches_torch_reference_int32_float32(self):
        self._assert_matches_reference(torch.int32, torch.float32)

    @unittest.skipIf(
        not hasattr(torch, "float8_e4m3fn"), "torch.float8_e4m3fn is unavailable"
    )
    def test_matches_torch_reference_int64_bfloat16(self):
        self._assert_matches_reference(torch.int64, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

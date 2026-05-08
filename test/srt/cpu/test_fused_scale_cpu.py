import itertools
import unittest

import torch

from sglang.srt.layers.attention.compressed.indexer import fused_scale_torch


class TestFusedScaleCPU(unittest.TestCase):
    shapes = [(1, 1), (2, 8), (3, 128)]
    weight_dtypes = [torch.float32, torch.bfloat16, torch.float16]

    def _run_case(self, shape, weight_dtype):
        torch.manual_seed(1234)
        weight = torch.randn(shape, dtype=torch.float32).to(weight_dtype).contiguous()
        # act_quant always emits fp32 scales; the kernel pins q_scale to fp32
        # and always returns fp32 output.
        q_scale = torch.rand(shape, dtype=torch.float32).contiguous()
        out_scale = 0.03125

        ref = fused_scale_torch(weight, out_scale, q_scale)
        out = torch.ops.sgl_kernel.fused_scale_cpu(weight, out_scale, q_scale)

        self.assertEqual(out.shape, (*shape, 1))
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)

    def test_fused_scale_cpu(self):
        for shape, weight_dtype in itertools.product(self.shapes, self.weight_dtypes):
            with self.subTest(shape=shape, weight_dtype=weight_dtype):
                self._run_case(shape, weight_dtype)

    def test_q_scale_view_shape(self):
        weight = torch.randn((2, 4), dtype=torch.bfloat16).contiguous()
        q_scale = torch.rand((2, 4, 1), dtype=torch.float32).contiguous()

        ref = fused_scale_torch(weight, 0.5, q_scale)
        out = torch.ops.sgl_kernel.fused_scale_cpu(weight, 0.5, q_scale)

        torch.testing.assert_close(out, ref, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
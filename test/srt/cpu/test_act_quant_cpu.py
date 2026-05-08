import itertools
import unittest

import torch

from sglang.srt.layers.attention.compressed.compressor import act_quant_pytorch


def _assert_fp8_equal(ref: torch.Tensor, out: torch.Tensor) -> None:
    assert ref.dtype == torch.float8_e4m3fn
    assert out.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(ref.view(torch.uint8), out.view(torch.uint8), atol=0, rtol=0)


class TestActQuantCPU(unittest.TestCase):
    shapes = [(1, 128), (3, 256), (2, 3, 128), (2, 2, 384)]
    dtypes = [torch.float32, torch.bfloat16, torch.float16]
    scale_fmts = [None, "power2"]

    def _run_case(self, shape, dtype, scale_fmt):
        torch.manual_seed(1234)
        x = (torch.randn(shape, dtype=torch.float32) * 3.0).to(dtype).contiguous()

        ref_y, ref_scale = act_quant_pytorch(x, block_size=128, scale_fmt=scale_fmt)
        out_y, out_scale = torch.ops.sgl_kernel.act_quant_cpu(
            x, 128, scale_fmt
        )

        _assert_fp8_equal(ref_y, out_y)
        torch.testing.assert_close(ref_scale, out_scale, atol=0, rtol=0)

    def test_act_quant_cpu(self):
        for shape, dtype, scale_fmt in itertools.product(
            self.shapes, self.dtypes, self.scale_fmts
        ):
            with self.subTest(shape=shape, dtype=dtype, scale_fmt=scale_fmt):
                self._run_case(shape, dtype, scale_fmt)

    def test_zeros_use_min_scale(self):
        x = torch.zeros((2, 128), dtype=torch.bfloat16)
        ref_y, ref_scale = act_quant_pytorch(x, block_size=128)
        out_y, out_scale = torch.ops.sgl_kernel.act_quant_cpu(x, 128, None)

        _assert_fp8_equal(ref_y, out_y)
        torch.testing.assert_close(ref_scale, out_scale, atol=0, rtol=0)

    def test_invalid_block_size(self):
        x = torch.randn((2, 129), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "Last dimension size must be divisible"):
            torch.ops.sgl_kernel.act_quant_cpu(x, 128, None)


if __name__ == "__main__":
    unittest.main()

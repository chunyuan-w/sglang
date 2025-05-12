import itertools
import unittest

import torch

# TODO: use interface in cpu.py
from sgl_kernel.common_ops import (
    convert_weight_packed,
    fp8_scaled_mm_cpu,
    int8_scaled_mm_cpu,
    int8_scaled_mm_with_quant,
    per_token_quant_int8_cpu,
    weight_packed_linear,
)

from sglang.test.test_utils import CustomTestCase

pres = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}

def per_token_quant_int8(x):
    x = x.float()
    absmax = x.abs().max(dim=-1).values
    absmax = absmax.clamp_min(1e-10).unsqueeze(-1)
    scale_x = absmax / 127
    x_q = x.mul(127 / absmax)
    x_q = torch.round(x_q).to(torch.int8)

    return x_q, scale_x


def native_w8a8_per_token_matmul(A, B, As, Bs, bias, output_dtype=torch.bfloat16):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    if bias is not None:
        C.add_(bias.view(1, -1))

    return C.reshape(origin_C_shape).to(output_dtype)


class TestGemm(CustomTestCase):
    M = [1, 101]
    N = [32 * 13]
    K = [32 * 16]
    has_bias = [False, True]

    # TODO: is this needed?
    # @classmethod
    # def setUpClass(cls):

    def _bf16_gemm(self, M, N, K, has_bias):

        mat1 = torch.randn(M, K, dtype=torch.bfloat16)
        mat2 = torch.randn(N, K, dtype=torch.bfloat16)

        ref = torch.matmul(mat1.float(), mat2.float().t())
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            ref.add_(bias.bfloat16())

        ref = ref.bfloat16()

        out = weight_packed_linear(mat1, mat2, bias if has_bias else None, False)

        packed_mat2 = convert_weight_packed(mat2)
        out2 = weight_packed_linear(mat1, packed_mat2, bias if has_bias else None, True)

        atol = rtol = pres[ref.dtype]
        self.assertTrue(torch.allclose(ref, out, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(ref, out2, atol=atol, rtol=rtol))

    def test_bf16_gemm(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._bf16_gemm(*params)

    def _int8_gemm(self, M, N, K, has_bias):
        dtype = torch.bfloat16
        A = torch.randn((M, K), dtype=dtype) / 10
        Aq, As = per_token_quant_int8(A)

        factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
        Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        Bs = torch.rand(N) * factor_for_scale

        bias = torch.randn(N) if has_bias else None
        ref_out = native_w8a8_per_token_matmul(Aq, Bq, As, Bs, bias, dtype)

        atol = rtol = pres[ref_out.dtype]

        Aq2, As2 = per_token_quant_int8_cpu(A)
        out = int8_scaled_mm_cpu(Aq2, Bq, As2, Bs, bias if has_bias else None, torch.bfloat16, False);
        self.assertTrue(torch.allclose(ref_out, out, atol=atol, rtol=rtol))

        # test the fused version
        fused_out = int8_scaled_mm_with_quant(A, Bq, Bs, bias if has_bias else None, torch.bfloat16, False);
        self.assertTrue(torch.allclose(ref_out, fused_out, atol=atol, rtol=rtol))

    def test_int8_gemm(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._int8_gemm(*params)        
        

if __name__ == "__main__":
    unittest.main()

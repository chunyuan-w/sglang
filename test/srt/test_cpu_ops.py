import itertools
import unittest

import torch

# TODO: use interface in cpu.py
from sgl_kernel.common_ops import (
    convert_weight_packed,
    fp8_scaled_mm_cpu,
    weight_packed_linear,
)

from sglang.test.test_utils import CustomTestCase

pres = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}


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


if __name__ == "__main__":
    unittest.main()

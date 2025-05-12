import itertools
import math
import unittest

import torch
import torch.nn as nn

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


class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)


def convert_weight(weight, scale_block_size, A_dtype):
    N, K = weight.size()
    fp8_max = 448.0
    scale_block_size_N, scale_block_size_K = scale_block_size  # (128, 128)

    pad_N = (scale_block_size_N - (N % scale_block_size_N)) % scale_block_size_N
    pad_K = (scale_block_size_K - (K % scale_block_size_K)) % scale_block_size_K

    if pad_N > 0 or pad_K > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))

    weight_blocks = weight.view(
        math.ceil(N / scale_block_size_N),
        scale_block_size_N,
        math.ceil(K / scale_block_size_K),
        scale_block_size_K
    )  # (8, 128, 8, 128)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3).contiguous()  # (8, 8, 128, 128)

    # Step 2: compute per-block max abs values → scale
    abs_max = weight_blocks.abs().amax(dim=(-2, -1), keepdim=True)  # (8, 8, 1, 1)
    scales = abs_max / fp8_max
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)  # avoid division by zero

    q_fp8 = (weight_blocks / scales).to(torch.float8_e4m3fn)
    q_fp8_reshape = q_fp8.permute(0, 2, 1, 3).contiguous()

    if pad_N > 0 or pad_K > 0:
        q_fp8_reshape = q_fp8_reshape.view(N + pad_N, K + pad_K)
        q_fp8_reshape = q_fp8_reshape[:N, :K].contiguous()
    else:
        q_fp8_reshape = q_fp8_reshape.view(N, K)

    dq_weight = q_fp8.float() * scales
    dq_weight = dq_weight.permute(0, 2, 1, 3).contiguous()  # (8, 128, 8, 128)

    if pad_N > 0 or pad_K > 0:
        w_dq = dq_weight.view(N + pad_N, K + pad_K).to(A_dtype)
        w_dq = w_dq[:N, :K].contiguous()
    else:
        w_dq = dq_weight.view(N, K).to(A_dtype)

    scales = scales.view(
        math.ceil(N / scale_block_size_N),
        math.ceil(K / scale_block_size_K)
    )

    return q_fp8_reshape, scales, w_dq


class TestGemm(CustomTestCase):
    M = [1, 101]
    N = [32 * 13]
    K = [32 * 16]
    has_bias = [False, True]
    
    M_fp8 = [1, 14]
    N_fp8 = [160]
    K_fp8 = [768]

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

    def _fp8_gemm(self, M, N, K, has_bias):
        prepack = True
        chunk = False
        scale_block_size_N = 64
        scale_block_size_K = 128
        assert scale_block_size_N <= N
        assert scale_block_size_K <= K
        A_dtype = torch.bfloat16

        model = Mod(K, N, has_bias).eval()
        if chunk:
            data = torch.randn(M, K + 6, dtype=A_dtype).narrow(1, 0, K)
        else:
            data = torch.randn(M, K, dtype=A_dtype)

        weight = model.linear.weight  # (N, K)

        if has_bias:
            bias = model.linear.bias

        fp8_weight, scales, dq_weight = convert_weight(weight, [scale_block_size_N, scale_block_size_K], A_dtype)

        if has_bias:
            ref = torch.matmul(data.to(A_dtype), dq_weight.T) + bias.to(A_dtype)
        else:
            ref = torch.matmul(data.to(A_dtype), dq_weight.T)

        if prepack:
            fp8_weight = convert_weight_packed(fp8_weight)

        opt = fp8_scaled_mm_cpu(
            data,
            fp8_weight,
            scales,
            [scale_block_size_N, scale_block_size_K],
            bias if has_bias else None,
            data.dtype,
            prepack
        )
        atol = rtol = pres[ref.dtype]
        self.assertTrue(torch.allclose(ref, opt, atol=atol, rtol=rtol))            

    def test_fp8_gemm(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._fp8_gemm(*params)      

if __name__ == "__main__":
    unittest.main()

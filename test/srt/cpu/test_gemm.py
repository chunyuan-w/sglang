import itertools
import unittest

# TODO: use interface in cpu.py
import sgl_kernel
import torch
import torch.nn as nn
from utils import (
    convert_weight,
    native_w8a8_per_token_matmul,
    per_token_quant_int8,
    precision,
)

from sgl_kernel.common_ops import convert_weight_packed, fp8_scaled_mm_cpu


class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)


has_bias = False

M_fp8 = 1
N_fp8 = 128
K_fp8 = 576

def _fp8_gemm(M, N, K, has_bias):
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

    fp8_weight, scales, dq_weight = convert_weight(
        weight, [scale_block_size_N, scale_block_size_K], A_dtype
    )

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
        prepack,
    )
    atol = rtol = precision[ref.dtype]
    
    torch.testing.assert_close(ref, opt, atol=atol, rtol=rtol)
    print("done")

_fp8_gemm(M_fp8, N_fp8, K_fp8, has_bias)


# if __name__ == "__main__":
#     unittest.main()

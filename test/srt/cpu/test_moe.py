import itertools
import math
import unittest

import torch
import torch.nn as nn

# TODO: use interface in cpu.py
from sgl_kernel.common_ops import convert_weight_packed
from sgl_kernel.common_ops import fused_experts_cpu as fused_experts
from sgl_kernel.common_ops import grouped_topk_cpu as grouped_topk
from sgl_kernel.common_ops import shared_expert_cpu as shared_expert
from utils import (
    BLOCK_K,
    BLOCK_N,
    SiluAndMul,
    fp8_factor_for_scale,
    fp8_max,
    fp8_min,
    native_fp8_fused_moe,
    per_token_quant_int8,
    precision,
    scaled_weight,
    torch_naive_fused_moe,
    torch_naive_moe,
    torch_w8a8_per_column_fused_moe,
    torch_w8a8_per_column_moe,
)

from sglang.test.test_utils import CustomTestCase


class TestSharedExpert(CustomTestCase):
    M = [2, 121]
    N = [32, 32 * 4]
    K = [32, 32 * 2]
    routed_scaling_factor = [16]

    M_fp8 = [2, 12]
    N_fp8 = [128, 256]
    K_fp8 = [256, 1024]

    def _bf16_shared_expert(self, m, n, k, routed_scaling_factor):
        dtype = torch.bfloat16
        prepack = True

        hidden_states = torch.randn(m, k, dtype=dtype) / k
        w1 = torch.randn(2 * n, k, dtype=dtype)
        w2 = torch.randn(k, n, dtype=dtype)
        fused_output = torch.randn(m, k, dtype=dtype) / k

        # fused moe mutates content in hs
        hidden_states2 = hidden_states.clone()

        # bfloat16
        ref = torch_naive_moe(
            hidden_states.float(),
            w1.float(),
            w2.float(),
            fused_output.float(),
            routed_scaling_factor,
        ).to(dtype=dtype)
        res = shared_expert(
            hidden_states,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            True,
            False,
            False,
            None,
            None,
            None,
            None,
            None,
            False,
        )

        atol = rtol = precision[ref.dtype]
        self.assertTrue(torch.allclose(ref, res, atol=atol, rtol=rtol))

    def test_bf16_shared_expert(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.routed_scaling_factor,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                routed_scaling_factor=params[3],
            ):
                self._bf16_shared_expert(*params)

    def _int8_shared_expert(self, m, n, k, routed_scaling_factor):
        dtype = torch.bfloat16
        prepack = True

        hidden_states = torch.randn(m, k, dtype=dtype) / k
        w1 = torch.randn(2 * n, k, dtype=dtype)
        w2 = torch.randn(k, n, dtype=dtype)
        fused_output = torch.randn(m, k, dtype=dtype) / k

        # fused moe mutates content in hs
        hidden_states2 = hidden_states.clone()

        w1_q, w1_s = per_token_quant_int8(w1)
        w2_q, w2_s = per_token_quant_int8(w2)
        ref2 = torch_w8a8_per_column_moe(
            hidden_states2.float(),
            w1_q,
            w2_q,
            w1_s,
            w2_s,
            fused_output.float(),
            routed_scaling_factor,
        ).to(dtype=dtype)
        res2 = shared_expert(
            hidden_states2,
            w1_q,
            w2_q,
            fused_output,
            routed_scaling_factor,
            True,
            True,
            False,
            w1_s,
            w2_s,
            None,
            None,
            None,
            False,
        )

        atol = rtol = precision[ref2.dtype]
        self.assertTrue(torch.allclose(ref2, res2, atol=atol, rtol=rtol))

    def test_int8_shared_expert(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.routed_scaling_factor,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                routed_scaling_factor=params[3],
            ):
                self._int8_shared_expert(*params)

    def _fp8_shared_expert(self, M, N, K, routed_scaling_factor):
        dtype = torch.bfloat16
        prepack = True

        a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

        w1_fp32 = torch.randn(1, 2 * N, K)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(1, K, N)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w1s = torch.randn(1, 2 * N // BLOCK_N, K // BLOCK_K) * fp8_factor_for_scale
        w2s = torch.randn(1, K // BLOCK_N, N // BLOCK_K) * fp8_factor_for_scale

        w1_scaled = scaled_weight(w1, w1s).view(2 * N, K)
        w2_scaled = scaled_weight(w2, w2s).view(K, N)

        # change back to 2D
        w1, w2 = w1.squeeze(0), w2.squeeze(0)
        w1s, w2s = w1s.squeeze(0), w2s.squeeze(0)
        w1_scaled, w2_scaled = w1_scaled.squeeze(0), w2_scaled.squeeze(0)

        fused_out = torch.randn(M, K, dtype=dtype) / math.sqrt(K)
        a2 = a.clone()

        # ref
        ic0 = torch.matmul(a.float(), w1_scaled.transpose(0, 1))
        ic1 = SiluAndMul(ic0)
        shared_out = torch.matmul(ic1, w2_scaled.transpose(0, 1))
        ref_out = shared_out + fused_out.float() * routed_scaling_factor
        ref_out = ref_out.to(dtype=dtype)

        w1 = convert_weight_packed(w1)  # [2N, K]
        w2 = convert_weight_packed(w2)  # [K, N]
        out = shared_expert(
            a2,
            w1,
            w2,
            fused_out,
            routed_scaling_factor,
            True,
            False,
            True,
            w1s,
            w2s,
            [BLOCK_N, BLOCK_K],
            None,
            None,
            True,
        )

        atol = rtol = precision[ref_out.dtype]
        self.assertTrue(torch.allclose(ref_out, out, atol=atol, rtol=rtol))

    def test_fp8_shared_expert(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.routed_scaling_factor,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                routed_scaling_factor=params[3],
            ):
                self._fp8_shared_expert(*params)


def fused_moe(a, w1, w2, score, topk, renormalize, prepack):

    G = 1
    topk_group = 1

    B, D = a.shape
    topk_weights = torch.empty(B, topk, dtype=torch.float32)
    topk_ids = torch.empty(B, topk, dtype=torch.int32)
    topk_weights, topk_ids = grouped_topk(a, score, topk, renormalize, G, topk_group)

    # print(topk_weights, topk_weights.size())
    # print(topk_ids, topk_ids.size())

    packed_w1 = convert_weight_packed(w1) if prepack else w1
    packed_w2 = convert_weight_packed(w2) if prepack else w2

    inplace = True
    return fused_experts(
        a,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        inplace,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
        prepack,
    )


class TestFusedExperts(CustomTestCase):
    M = [2, 114]
    N = [32]
    K = [32]
    E = [4]
    topk = [2]
    renormalize = [False, True]

    M_int8 = [1, 39]
    N_int8 = [128]
    K_int8 = [256]
    E_int8 = [8]
    topk_int8 = [2, 3]

    M_fp8 = [2, 121]
    N_fp8 = [128, 512]
    K_fp8 = [128, 1024]
    E_fp8 = [8]
    topk_fp8 = [2, 4]

    def _bf16_moe(self, m, n, k, e, topk, renormalize):
        dtype = torch.bfloat16
        prepack = True

        a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
        score = torch.randn((m, e), device="cpu", dtype=dtype)

        torch_output = torch_naive_fused_moe(a, w1, w2, score, topk, renormalize)
        fused_output = fused_moe(a, w1, w2, score, topk, renormalize, prepack)

        atol = rtol = precision[torch_output.dtype]
        self.assertTrue(
            torch.allclose(torch_output, fused_output, atol=atol, rtol=rtol)
        )

    def test_bf16_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.topk,
            self.renormalize,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                e=params[3],
                topk=params[4],
                renormalize=params[5],
            ):
                self._bf16_moe(*params)

    def _int8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16
        prepack = True

        # Initialize int8 quantization parameters
        factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        # Input tensor
        # M * K
        a = torch.randn((M, K), dtype=dtype) / math.sqrt(K)

        # Generate int8 weights
        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
        w1 = (w1_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
        w2 = (w2_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

        # Generate scale for each column (per-column quantization)
        w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * factor_for_scale
        w2_s = torch.rand(E, K, device=w2_fp32.device) * factor_for_scale

        # Calculate routing
        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        ref_out = torch_w8a8_per_column_fused_moe(
            a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk
        )

        inplace = True
        packed_w1 = convert_weight_packed(w1) if prepack else w1
        packed_w2 = convert_weight_packed(w2) if prepack else w2
        out = fused_experts(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids.to(torch.int32),
            inplace,
            True,
            False,
            w1_s,
            w2_s,
            None,
            None,
            None,
            prepack,
        )

        atol = rtol = precision[ref_out.dtype]
        # Increase the tolerance for large input shapes
        if M > 35:
            atol = rtol = 0.02
        self.assertTrue(torch.allclose(ref_out, out, atol=atol, rtol=rtol))

    def test_int8_moe(self):
        for params in itertools.product(
            self.M_int8,
            self.N_int8,
            self.K_int8,
            self.E_int8,
            self.topk_int8,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
            ):
                self._int8_moe(*params)

    def _fp8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16

        a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

        w1_fp32 = torch.randn(E, 2 * N, K)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(E, K, N)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w1s = torch.randn(E, 2 * N // BLOCK_N, K // BLOCK_K) * fp8_factor_for_scale
        w2s = torch.randn(E, K // BLOCK_N, N // BLOCK_K) * fp8_factor_for_scale

        w1_scaled = scaled_weight(w1, w1s)
        w2_scaled = scaled_weight(w2, w2s)

        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        w1 = convert_weight_packed(w1)
        w2 = convert_weight_packed(w2)

        ref_out = native_fp8_fused_moe(
            a, w1_scaled, w2_scaled, topk_weight, topk_ids, topk
        )
        out = fused_experts(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids.to(torch.int32),
            False,
            False,
            True,
            w1s,
            w2s,
            [BLOCK_N, BLOCK_K],
            None,
            None,
            True,
        )

        atol = rtol = precision[dtype]
        self.assertTrue(torch.allclose(ref_out.bfloat16(), out, atol=atol, rtol=rtol))

    def test_fp8_moe(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.E_fp8,
            self.topk_fp8,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
            ):
                self._fp8_moe(*params)


if __name__ == "__main__":
    unittest.main()

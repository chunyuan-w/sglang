import math
import unittest

# TODO: use interface in cpu.py
import torch

from sglang.srt.layers.amx_utils import CPUQuantMethod

kernel = torch.ops.sgl_kernel

torch.manual_seed(1234)

from utils import (
    BLOCK_K,
    BLOCK_N,
    ClampedSiluAndMul,
    MXFP4QuantizeUtil,
    factor_for_scale,
    fp8_max,
    fp8_min,
    native_fp8_fused_moe,
    parametrize,
    precision,
    scaled_weight,
    torch_naive_fused_moe,
    torch_naive_fused_moe_gptoss,
    torch_w8a8_per_column_fused_moe,
    unpack_and_dequant_awq,
)

from sglang.test.test_utils import CustomTestCase


def fused_moe(a, w1, w2, score, topk, renormalize, prepack):

    G = 1
    topk_group = 1

    B, D = a.shape
    topk_weights = torch.empty(B, topk, dtype=torch.float32)
    topk_ids = torch.empty(B, topk, dtype=torch.int32)
    topk_weights, topk_ids = kernel.grouped_topk_cpu(
        a, score, topk, renormalize, G, topk_group, 0, None, None
    )

    packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
    packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2

    inplace = True
    return kernel.fused_experts_cpu(
        a,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        inplace,
        CPUQuantMethod.UNQUANT,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        prepack,
    )


class TestFusedExperts(CustomTestCase):

    @parametrize(m=[2, 114], n=[32], k=[32], e=[4], topk=[2], renormalize=[False, True])
    def test_bf16_moe(self, m, n, k, e, topk, renormalize):
        dtype = torch.bfloat16
        prepack = True

        a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
        score = torch.randn((m, e), device="cpu", dtype=dtype)

        torch_output = torch_naive_fused_moe(a, w1, w2, score, topk, renormalize)
        fused_output = fused_moe(a, w1, w2, score, topk, renormalize, prepack)

        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    @parametrize(
        m=[1, 32], n=[128, 64], k=[128, 64], e=[4], topk=[2], renormalize=[False]
    )
    def test_bf16_moe_bias(self, m, n, k, e, topk, renormalize):
        dtype = torch.bfloat16

        a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
        w1_b = torch.randn((e, 2 * n), device="cpu", dtype=torch.float) / 10
        w2 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
        w2_b = torch.randn((e, k), device="cpu", dtype=torch.float) / 10
        score = torch.randn((m, e), device="cpu", dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        alpha = 1.702
        limit = 7.0
        torch_output = torch_naive_fused_moe_gptoss(
            a, w1, w2, w1_b, w2_b, topk_weight, topk_ids, renormalize, alpha, limit, e
        )
        packed_w1 = kernel.convert_weight_packed(w1)
        packed_w2 = kernel.convert_weight_packed(w2)
        fused_output = torch.ops.sgl_kernel.fused_experts_cpu(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids.to(torch.int),
            False,  # inplace # See [Note] inplace should be False in fused_experts.
            CPUQuantMethod.UNQUANT,
            None,  # w1_scale
            None,  # w2_scale
            None,  # w1_zp
            None,  # w2_zp
            None,  # block_size
            w1_b,
            w2_b,
            alpha,
            limit,
            True,  # is_vnni
        )
        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    @parametrize(M=[1, 39], N=[128], K=[256], E=[8], topk=[3])
    def test_int8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16
        prepack = True

        # Initialize int8 quantization parameters
        int8_factor_for_scale = 1e-2
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
        w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * int8_factor_for_scale
        w2_s = torch.rand(E, K, device=w2_fp32.device) * int8_factor_for_scale

        # Calculate routing
        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        ref_out = torch_w8a8_per_column_fused_moe(
            a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk
        )

        inplace = True
        packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
        packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2
        out = kernel.fused_experts_cpu(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids.to(torch.int32),
            inplace,
            CPUQuantMethod.INT8_W8A8,
            w1_s,
            w2_s,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            prepack,
        )

        atol = rtol = precision[ref_out.dtype]
        # Increase the tolerance for large input shapes
        if M > 35:
            atol = rtol = 0.02
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    @parametrize(M=[2, 121], N=[352, 512], K=[256, 320], E=[8], topk=[4])
    def test_fp8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16

        a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

        w1_fp32 = torch.randn(E, 2 * N, K)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(E, K, N)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w1s = (
            torch.randn(E, math.ceil(2 * N / BLOCK_N), math.ceil(K / BLOCK_K))
            * factor_for_scale
        )
        w2s = (
            torch.randn(E, math.ceil(K / BLOCK_N), math.ceil(N / BLOCK_K))
            * factor_for_scale
        )

        w1_scaled = scaled_weight(w1, w1s)
        w2_scaled = scaled_weight(w2, w2s)

        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        w1 = kernel.convert_weight_packed(w1)
        w2 = kernel.convert_weight_packed(w2)

        ref_out = native_fp8_fused_moe(
            a, w1_scaled, w2_scaled, topk_weight, topk_ids, topk
        )
        out = kernel.fused_experts_cpu(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids.to(torch.int32),
            False,
            CPUQuantMethod.FP8_W8A16,
            w1s,
            w2s,
            None,
            None,
            [BLOCK_N, BLOCK_K],
            None,
            None,
            None,
            None,
            True,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    @parametrize(M=[2, 121], N=[352, 512], K=[256, 320], E=[8], topk=[4])
    def test_mxfp4_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16

        a = torch.randn(M, K, dtype=dtype) / 10

        w1_bf16 = torch.randn((E, 2 * N, K), dtype=dtype) / 10
        w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
        w1s = w1s.reshape(E, 2 * N, K // 32)
        w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

        w2_bf16 = torch.randn((E, K, N), dtype=dtype) / 10
        w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
        w2s = w2s.reshape(E, K, N // 32)
        w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)

        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        w1 = kernel.convert_weight_packed(w1q)
        w2 = kernel.convert_weight_packed(w2q)
        w1s = kernel.convert_scale_packed(w1s)
        w2s = kernel.convert_scale_packed(w2s)

        ref_out = native_fp8_fused_moe(
            a, w1dq.float(), w2dq.float(), topk_weight, topk_ids, topk
        )
        out = kernel.fused_experts_cpu(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids.to(torch.int32),
            False,
            CPUQuantMethod.MXFP4,
            w1s,
            w2s,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            True,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    @parametrize(
        m=[1, 32], n=[128, 64], k=[128, 64], e=[4], topk=[2], renormalize=[False]
    )
    def test_mxfp4_moe_bias(self, m, n, k, e, topk, renormalize):
        dtype = torch.bfloat16

        a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
        w1_bf16 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
        w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
        w1s = w1s.reshape(e, 2 * n, k // 32)
        w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)
        w1_b = torch.randn((e, 2 * n), device="cpu", dtype=torch.float32) / 10
        w2_bf16 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
        w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
        w2s = w2s.reshape(e, k, n // 32)
        w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)
        w2_b = torch.randn((e, k), device="cpu", dtype=torch.float32) / 10
        score = torch.randn((m, e), device="cpu", dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        alpha = 1.702
        limit = 7.0
        torch_output = torch_naive_fused_moe_gptoss(
            a,
            w1dq,
            w2dq,
            w1_b,
            w2_b,
            topk_weight,
            topk_ids,
            renormalize,
            alpha,
            limit,
            e,
        )

        w1 = kernel.convert_weight_packed(w1q)
        w2 = kernel.convert_weight_packed(w2q)
        w1s = kernel.convert_scale_packed(w1s)
        w2s = kernel.convert_scale_packed(w2s)

        fused_output = torch.ops.sgl_kernel.fused_experts_cpu(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids.to(torch.int32),
            False,  # inplace # See [Note] inplace should be False in fused_experts.
            CPUQuantMethod.MXFP4,  # use_mxfp4
            w1s,  # w1_scale
            w2s,  # w2_scale
            None,  # w1_zp
            None,  # w2_zp
            None,  # block_size
            w1_b,
            w2_b,
            alpha,
            limit,
            True,  # is_vnni
        )
        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    # DeepSeek-V4-Flash routed-expert shape (H=4096 hidden, I=2048 moe_intermediate,
    # 256 experts but reduced for test speed, top-6 routing). DSV4 has no bias and
    # uses silu_and_mul activation in the default `2604` mode (no swiglu clamp).
    #
    # This test mirrors the real create_weights → _process_weights_cpu path in
    # DeepSeekMxfp4MoEMethod: weights start as int8 (packed FP4) and scales as
    # float32 (as allocated by create_weights), then _process_weights_cpu
    # view-casts int8→uint8 and float32→float8_e8m0fnu→uint8 before VNNI packing.
    @parametrize(M=[1, 16], N=[256], K=[512], E=[8], topk=[6], limit=[None, 10.0])
    def test_mxfp4_moe_dsv4_shape(self, M, N, K, E, topk, limit):
        dtype = torch.bfloat16
        fp4_block_k = 32

        a = torch.randn(M, K, dtype=dtype) / 10

        # --- Step 1: generate realistic FP4 weights via quantization, then
        # re-express them in the same storage format as create_weights (int8
        # for packed FP4, float32 for E8M0 scales). ---
        w1_bf16 = torch.randn((E, 2 * N, K), dtype=dtype) / 10
        w1q_uint8, w1s_raw = MXFP4QuantizeUtil.quantize(w1_bf16)
        w1s_uint8 = w1s_raw.reshape(E, 2 * N, K // fp4_block_k)

        w2_bf16 = torch.randn((E, K, N), dtype=dtype) / 10
        w2q_uint8, w2s_raw = MXFP4QuantizeUtil.quantize(w2_bf16)
        w2s_uint8 = w2s_raw.reshape(E, K, N // fp4_block_k)

        # Simulate checkpoint storage: packed FP4 as int8, E8M0 as float32
        w1_weight = w1q_uint8.view(torch.int8)   # create_weights dtype
        w2_weight = w2q_uint8.view(torch.int8)
        w1_scale = torch.exp2(w1s_uint8.float() - 127)  # float32 form
        w2_scale = torch.exp2(w2s_uint8.float() - 127)

        # --- Step 2: build reference uint8 weights & scales via independent math ---
        # int8 → uint8 equivalent without .view(): mask to unsigned range
        w1q_ref = (w1_weight.to(torch.int16) & 0xFF).to(torch.uint8)
        w2q_ref = (w2_weight.to(torch.int16) & 0xFF).to(torch.uint8)
        # float32 → E8M0 uint8 equivalent without .to(float8_e8m0fnu):
        # since scales are exact powers of 2, biased_exp = log2(scale) + 127
        w1s_ref = (torch.log2(w1_scale) + 127).to(torch.uint8)
        w2s_ref = (torch.log2(w2_scale) + 127).to(torch.uint8)

        # Dequantize from the independently-derived uint8 for the golden reference
        w1dq = MXFP4QuantizeUtil.dequantize(w1q_ref, dtype, w1s_ref)
        w2dq = MXFP4QuantizeUtil.dequantize(w2q_ref, dtype, w2s_ref)

        # --- Step 3: _process_weights_cpu dtype conversions (code under test) ---
        # int8 → uint8 (bit-equivalent reinterpret)
        w1q = w1_weight.view(torch.uint8)
        w2q = w2_weight.view(torch.uint8)
        # float32 → float8_e8m0fnu → uint8
        w1s = w1_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
        w2s = w2_scale.to(torch.float8_e8m0fnu).view(torch.uint8)

        # Verify the _process_weights_cpu conversions match the independent reference
        torch.testing.assert_close(w1q, w1q_ref)
        torch.testing.assert_close(w2q, w2q_ref)
        torch.testing.assert_close(w1s, w1s_ref)
        torch.testing.assert_close(w2s, w2s_ref)

        # --- Step 4: VNNI packing (same as _process_weights_cpu) ---
        w1 = kernel.convert_weight_packed(w1q)
        w2 = kernel.convert_weight_packed(w2q)
        w1s = kernel.convert_scale_packed(w1s)
        w2s = kernel.convert_scale_packed(w2s)

        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        # --- Step 5: compute reference output ---
        # limit=None → 2604 mode (plain silu_and_mul)
        # limit=float → 2604B mode (clamped_silu_and_mul, no alpha)
        w1f, w2f = w1dq.float(), w2dq.float()
        B, D = a.shape
        a_rep = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).float()
        ref = torch.zeros(B * topk, w2f.shape[1], dtype=torch.float32)
        tw = topk_weight.view(-1)
        ti = topk_ids.view(-1)
        for i in range(w1f.shape[0]):
            mask = ti == i
            if mask.sum():
                ic0 = torch.matmul(a_rep[mask], w1f[i].transpose(0, 1))
                if limit is not None:
                    ic1 = ClampedSiluAndMul(ic0, limit)
                else:
                    from utils import SiluAndMul
                    ic1 = SiluAndMul(ic0)
                ref[mask] = torch.matmul(ic1, w2f[i].transpose(0, 1))
        ref_out = (
            (ref.view(B, -1, w2f.shape[1]) * tw.view(B, -1, 1).to(ref.dtype))
            .sum(dim=1)
            .to(a_rep.dtype)
        )

        out = kernel.fused_experts_cpu(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids.to(torch.int32),
            False,  # inplace
            CPUQuantMethod.MXFP4,
            w1s,
            w2s,
            None,  # w1_zp
            None,  # w2_zp
            None,  # block_size
            None,  # w1_bias  (DSV4 has no bias)
            None,  # w2_bias
            None,  # gemm1_alpha  (2604B has no alpha, only limit)
            limit,  # gemm1_clamp_limit: None for 2604, float for 2604B
            True,  # is_vnni — pre-packed above
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    @parametrize(M=[1, 6], N=[512], K=[256], E=[8], topk=[4])
    def test_int4_moe(self, M, N, K, E, topk, group_size=128):
        dtype = torch.bfloat16

        a = torch.rand(M, K, dtype=dtype) / math.sqrt(K)

        awq_w13_weight = torch.randint(-127, 128, (E, K, 2 * N // 8)).to(torch.int)
        awq_w13_zero = torch.randint(0, 10, (E, K // group_size, 2 * N // 8)).to(
            torch.int
        )
        awq_w13_scales = torch.rand(E, int(K // group_size), 2 * N).to(torch.bfloat16)

        awq_w2_weight = torch.randint(-127, 128, (E, N, K // 8)).to(torch.int)
        awq_w2_zero = torch.randint(0, 10, (E, N // group_size, K // 8)).to(torch.int)
        awq_w2_scales = torch.rand(E, int(N // group_size), K).to(torch.bfloat16)
        bf16_w13_weight = []
        bf16_w2_weight = []
        for i in range(E):
            bf16_w13_weight_i, _ = unpack_and_dequant_awq(
                awq_w13_weight[i], awq_w13_zero[i], awq_w13_scales[i], 4, 128
            )
            bf16_w2_weight_i, _ = unpack_and_dequant_awq(
                awq_w2_weight[i], awq_w2_zero[i], awq_w2_scales[i], 4, 128
            )
            bf16_w13_weight.append(bf16_w13_weight_i)
            bf16_w2_weight.append(bf16_w2_weight_i)
        bf16_w13_weight = torch.stack(bf16_w13_weight).detach()
        bf16_w2_weight = torch.stack(bf16_w2_weight).detach()

        score = torch.rand((M, E), dtype=dtype)

        ref_out = torch_naive_fused_moe(
            a, bf16_w13_weight, bf16_w2_weight, score, topk, False
        )
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        awq_w13_weight_pack, awq_w13_zero_pack, awq_w13_scales_pack = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_w13_weight, awq_w13_zero, awq_w13_scales
            )
        )
        awq_w2_weight_pack, awq_w2_zero_pack, awq_w2_scales_pack = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_w2_weight, awq_w2_zero, awq_w2_scales
            )
        )

        out = kernel.fused_experts_cpu(
            a,
            awq_w13_weight_pack,
            awq_w2_weight_pack,
            topk_weight,
            topk_ids.to(torch.int32),
            False,
            CPUQuantMethod.INT4_W4A8,
            awq_w13_scales_pack,
            awq_w2_scales_pack,
            awq_w13_zero_pack,
            awq_w2_zero_pack,
            None,
            None,
            None,
            None,
            None,
            True,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
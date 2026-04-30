"""Native (pure-torch) MoE for CPU debug.

Goal: provide a *correctness-guaranteed* MoE path so that if it produces wrong
output we can rule the MoE in/out as the bug source.

Independence from sgl-kernel:
  - Dequantization delegates to ``MXFP4QuantizeUtil`` in ``mxfp4_tensor.py``,
    a verbatim copy from NVIDIA TensorRT-Model-Optimizer (see the URL in that
    file's header). It is independent of any sgl-kernel C++ code, so a
    spec-misread bug in the AMX kernel would NOT propagate here.
  - The MoE forward is a textbook BF16 expert loop: gate/up split (matching
    the sglang loader's w13[0:N]=gate, w13[N:2N]=up convention, see
    ``fused_moe_triton/layer.py:418``), optional symmetric clamp on up and
    upper-bound clamp on gate (DSV4 2604B), SiLU(gate)*up, down-proj,
    topk-weighted accumulate.

Run ``python python/sglang/srt/layers/quantization/mxfp4_native_cpu.py`` to
execute the self-tests (round-trip BF16->MXFP4->BF16 + LUT spot-check).
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil


def _scale_to_e8m0_uint8(scale: torch.Tensor) -> torch.Tensor:
    """Normalize a scale tensor into raw E8M0 bytes (uint8) for modelopt.

    MXFP4 scales are E8M0 (exact powers of 2). The post-loader scale on a
    layer can arrive in three forms:
      - ``torch.float8_e8m0fnu``: bit-equivalent reinterpret to uint8.
      - ``torch.uint8``: already raw bytes.
      - ``torch.float32``: a multiplier 2^k; convert via float8_e8m0fnu cast.
    """
    if scale.dtype == torch.uint8:
        return scale
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.view(torch.uint8)
    if scale.dtype == torch.float32:
        return scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    raise ValueError(f"Unsupported MXFP4 scale dtype: {scale.dtype}")


def dequantize_mxfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 32,
) -> torch.Tensor:
    """Dequantize MXFP4 (E2M1 + E8M0 block-32) to ``out_dtype``.

    packed: [..., D/2] uint8
    scale:  [..., D/32] fp32 / float8_e8m0fnu / uint8

    Delegates to ``MXFP4QuantizeUtil.dequantize`` (NVIDIA TensorRT-Model-
    Optimizer reference implementation). This makes the dequant correctness
    independent of any sgl-kernel code.
    """
    assert packed.dtype == torch.uint8, f"expect uint8, got {packed.dtype}"
    D = packed.shape[-1] * 2
    assert D % block_size == 0
    assert scale.shape[-1] == D // block_size, (
        f"scale last-dim {scale.shape[-1]} != D/block_size {D // block_size}"
    )

    scale_u8 = _scale_to_e8m0_uint8(scale).to(packed.device)
    return MXFP4QuantizeUtil.dequantize(
        packed, out_dtype, scale_u8, block_sizes=[block_size]
    )


def native_fused_experts_bf16_cpu(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gemm1_clamp_limit: Optional[float],
) -> torch.Tensor:
    """Pure-torch BF16 MoE forward.

    hidden_states: [M, K]                bf16/fp16
    w13:           [E, 2N, K]            bf16 (already dequantized)
    w2:            [E,  K, N]            bf16
    topk_weights:  [M, topk]             fp32 (caller may have pre-applied via
                                         ``apply_topk_weights_cpu``; ones are
                                         passed through unchanged)
    topk_ids:      [M, topk]             int32
    gemm1_clamp_limit: optional float (DSV4 2604B); None => plain silu_and_mul
    """
    assert hidden_states.dim() == 2
    M, K = hidden_states.shape
    E, two_N, K_w = w13.shape
    assert K_w == K, f"K mismatch: w13 K={K_w} vs hidden K={K}"
    assert two_N % 2 == 0
    N = two_N // 2
    topk = topk_ids.shape[1]
    dtype = hidden_states.dtype

    L: Optional[float] = (
        float(gemm1_clamp_limit) if gemm1_clamp_limit is not None else None
    )

    out = torch.zeros_like(hidden_states)

    flat_ids = topk_ids.reshape(-1).to(torch.long)
    flat_weights = topk_weights.reshape(-1).to(torch.float32)
    token_index = (
        torch.arange(M, device=hidden_states.device).repeat_interleave(topk)
    )

    for e in range(E):
        sel = (flat_ids == e).nonzero(as_tuple=False).squeeze(-1)
        if sel.numel() == 0:
            continue
        sel_tok = token_index[sel]
        sel_w = flat_weights[sel]

        x = hidden_states.index_select(0, sel_tok).to(torch.float32)
        W1 = w13[e].to(torch.float32)  # [2N, K]
        W2 = w2[e].to(torch.float32)   # [K, N]

        gate_up = x @ W1.T  # [n_sel, 2N]
        gate = gate_up[..., :N]
        up = gate_up[..., N:]

        if L is not None:
            gate = torch.clamp(gate, max=L)
            up = torch.clamp(up, min=-L, max=L)

        inter = torch.nn.functional.silu(gate) * up  # [n_sel, N]
        down = inter @ W2.T  # [n_sel, K]

        contrib = (down * sel_w.unsqueeze(-1)).to(dtype)
        out.index_add_(0, sel_tok, contrib)

    return out


# -----------------------------------------------------------------------------
# Self-tests. Run via:
#   python -m sglang.srt.layers.quantization.mxfp4_native_cpu
# -----------------------------------------------------------------------------

def _quantize_mxfp4_reference(x: torch.Tensor, block_size: int = 32):
    """Inline-copy of ``MXFP4QuantizeUtil.quantize`` minus the broken class
    constructor at its tail. The quantize math itself (the lines that compute
    ``input_q`` and ``e8m0_scale``) is unchanged from the NVIDIA modelopt
    reference -- only the wrapper return is replaced with raw tensors.
    """
    E2M1_max = 6.0
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], device=x.device)

    def cast_fp4(v):
        sign = torch.sign(v)
        sign_bit = (2 - sign) // 2
        ord_ = torch.sum((v.abs().unsqueeze(-1) - E2M1_bounds) > 0, dim=-1)
        return (sign_bit * 0b1000 + ord_).to(torch.uint8)

    def fuse(u4):
        left = u4[..., 0::2]
        right = u4[..., 1::2]
        out = (right.clone() << 4)
        out[..., : left.shape[-1]] += left
        return out

    original_shape = x.shape
    flat = x.reshape(-1, block_size)
    amax = flat.abs().max(dim=-1, keepdim=True).values
    descale = amax / E2M1_max
    e8m0 = torch.ceil(
        torch.maximum(torch.log2(descale), torch.tensor(-127.0, device=x.device))
    )
    scaled = (flat / torch.exp2(e8m0)).reshape(original_shape)
    u4 = cast_fp4(scaled)
    packed = fuse(u4)
    e8m0_byte = (e8m0 + 127).to(torch.uint8).reshape(*original_shape[:-1], -1)
    return packed, e8m0_byte


def _self_test_dequant_roundtrip() -> None:
    """Round-trip a random tensor through MXFP4 quant -> dequant. The
    quantize side uses inline modelopt-spec math; the dequant goes through
    ``dequantize_mxfp4`` -> ``MXFP4QuantizeUtil.dequantize``.

    With only 8 magnitudes per block, the RMS error on Gaussian data is
    empirically ~10-15% -- that's intrinsic FP4 noise, not a bug. The real
    correctness signal is *cosine similarity*: a wrong LUT, swapped sign bit,
    wrong nibble order, or wrong scale axis would tank correlation, not just
    add noise. We require cos >= 0.985 and bias-free recon.
    """
    torch.manual_seed(123)
    for shape in [(4096,), (16, 1024), (8, 32, 256)]:
        x = torch.randn(*shape, dtype=torch.float32) * 1.5
        packed, e8m0 = _quantize_mxfp4_reference(x, block_size=32)
        x_recon = dequantize_mxfp4(packed, e8m0, torch.float32)

        rms = ((x_recon - x).float() ** 2).mean().sqrt()
        x_rms = (x.float() ** 2).mean().sqrt()
        rel_rms = (rms / x_rms).item()

        cos = torch.nn.functional.cosine_similarity(
            x_recon.reshape(-1).unsqueeze(0),
            x.reshape(-1).unsqueeze(0),
        ).item()
        bias = (x_recon - x).mean().item() / x_rms.item()

        assert cos > 0.985, (
            f"Round-trip cosine too low ({cos:.4f}) for shape {shape} "
            "-- suggests a structural bug (LUT/sign/packing/scale axis)."
        )
        assert abs(bias) < 0.02, f"Round-trip bias suspicious: {bias:.4f}"
        print(
            f"  shape={shape}  cos={cos:.6f}  rel_rms={rel_rms:.2%}  "
            f"bias={bias:.4f}"
        )


def _self_test_dequant_lut() -> None:
    """Spot-check that nibble values 0..15 decode to the expected E2M1 LUT
    values when scale=2^0=1. This is the ground-truth LUT from the OCP MX
    spec; matches NVIDIA modelopt and sgl-kernel/csrc/cpu/vec.h:166."""
    expected = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    nibbles = torch.arange(32, dtype=torch.uint8) % 16
    low = nibbles[0::2]
    high = nibbles[1::2]
    packed = (high << 4) | low
    scale_e8m0 = torch.tensor([127], dtype=torch.uint8)  # 2^0 = 1
    out = dequantize_mxfp4(packed, scale_e8m0, torch.float32)
    assert torch.equal(out, expected.repeat(2)), (
        f"LUT mismatch:\n  got: {out}\n  exp: {expected.repeat(2)}"
    )
    print("  LUT spot-check: OK")


def _self_test_moe_smoke() -> None:
    """End-to-end MoE smoke: random FP4 weights, BF16 forward, sane output."""
    torch.manual_seed(0)
    M, K, E, N, topk = 32, 128, 4, 64, 2
    hs = torch.randn(M, K, dtype=torch.bfloat16) * 0.5
    w13_packed = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8)
    w2_packed = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8)
    w13_scale = torch.randint(125, 130, (E, 2 * N, K // 32), dtype=torch.uint8)
    w2_scale = torch.randint(125, 130, (E, K, N // 32), dtype=torch.uint8)

    w13_bf16 = dequantize_mxfp4(w13_packed, w13_scale, torch.bfloat16)
    w2_bf16 = dequantize_mxfp4(w2_packed, w2_scale, torch.bfloat16)
    assert w13_bf16.shape == (E, 2 * N, K)
    assert w2_bf16.shape == (E, K, N)

    gating = torch.randn(M, E)
    tw, ti = gating.softmax(-1).topk(topk, dim=-1)

    out = native_fused_experts_bf16_cpu(
        hs, w13_bf16, w2_bf16, tw.float(), ti.to(torch.int32),
        gemm1_clamp_limit=None,
    )
    assert out.shape == hs.shape
    assert torch.isfinite(out).all()
    print(
        f"  smoke: out range=[{out.min().item():.2f}, {out.max().item():.2f}]  "
        f"finite=OK  shape={tuple(out.shape)}"
    )

    # 2604B clamp path: just verify it runs and stays finite.
    out_c = native_fused_experts_bf16_cpu(
        hs, w13_bf16, w2_bf16, tw.float(), ti.to(torch.int32),
        gemm1_clamp_limit=7.0,
    )
    assert torch.isfinite(out_c).all()
    print("  clamp-path smoke: OK")


if __name__ == "__main__":
    print("[1/3] dequant LUT spot-check")
    _self_test_dequant_lut()
    print("[2/3] dequant round-trip (BF16 -> MXFP4 -> BF16)")
    _self_test_dequant_roundtrip()
    print("[3/3] MoE forward smoke")
    _self_test_moe_smoke()
    print("OK: native MXFP4 MoE for CPU is internally consistent.")

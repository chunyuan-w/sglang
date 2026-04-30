
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_flashinfer_available,
    log_info_on_rank0,
    set_weight_attrs,
)
from sglang.srt.utils.common import next_power_of_2

if is_flashinfer_available():
    from flashinfer import mxfp8_quantize, shuffle_matrix_a, shuffle_matrix_sf_a
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput


from sglang.srt.debug_utils.deepseek_v4_debug_utils import deepseek_v4_moe_code_path_checker
from sglang.srt.environ import envs
from sglang.srt.utils.common import get_bool_env_var

_USE_OFFICIAL_SHUFFLE = get_bool_env_var(
    "SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE", default="true"
)

# Debug toggle: replace the AMX sgl-kernel CPU MoE with a pure-torch
# dequant->bf16 reference (see mxfp4_native_cpu.py). Used to bisect
# whether wrong-sentence outputs come from the FP4 quant kernel itself.
_USE_NATIVE_MXFP4_CPU = get_bool_env_var(
    "SGLANG_NATIVE_MXFP4_MOE_CPU", default="false"
)

_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()

class PackTopkIds:

    @classmethod
    def execute(
        cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        return cls.triton(topk_ids, topk_weights)

    @classmethod
    def vanilla(
        cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        weight_bits = (
            topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        )
        return (topk_ids.to(torch.int32) << 16) | weight_bits

    @classmethod
    def triton(cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        assert (
            topk_ids.shape == topk_weights.shape
        ), f"shape mismatch: {topk_ids.shape=} vs {topk_weights.shape=}"
        assert topk_ids.ndim >= 1, f"expected >=1D, got {topk_ids.shape=}"

        assert (
            topk_ids.dtype == torch.int32
        ), f"topk_ids must be int32, got {topk_ids.dtype}"
        assert (
            topk_weights.dtype == torch.float32
        ), f"topk_weights must be float32, got {topk_weights.dtype}"

        assert topk_ids.is_contiguous(), "topk_ids must be contiguous"
        assert topk_weights.is_contiguous(), "topk_weights must be contiguous"

        out = torch.empty_like(topk_ids, dtype=torch.int32)
        numel = out.numel()
        if numel == 0:
            return out

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _pack_topk_ids_triton_kernel[grid](
            topk_ids,
            topk_weights,
            out,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out


@triton.jit
def _pack_topk_ids_triton_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    ids = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    w = tl.load(topk_weights_ptr + offsets, mask=mask, other=0.0)

    w_bf16 = w.to(tl.bfloat16)
    w_i16 = w_bf16.to(tl.int16, bitcast=True)
    w_i32 = w_i16.to(tl.int32) & 0xFFFF

    ids_i32 = ids.to(tl.int32)
    packed = (ids_i32 << 16) | w_i32

    tl.store(out_ptr + offsets, packed, mask=mask)


class DeepSeekMxfp4MoEMethod:

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        self.flashinfer_mxfp4_moe_precision = (
            get_global_server_args().flashinfer_mxfp4_moe_precision
        )

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config

        swiglu_limit = moe_runner_config.swiglu_limit
        is_2604b = envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B"
        assert is_2604b == (swiglu_limit is not None), (
            f"swiglu_limit must be non-None iff submode=2604B "
            f"(got submode={envs.SGLANG_DSV4_2604_SUBMODE.get()!r}, "
            f"swiglu_limit={swiglu_limit!r})"
        )
        self._gemm1_clamp_limit_tensor = (
            torch.full(
                (layer.num_local_experts,),
                swiglu_limit,
                dtype=torch.float32,
                device=layer.w13_weight.device,
            )
            if swiglu_limit is not None
            else None
        )

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        fp4_block_k = 32

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.utils import reorder_w1w3_to_w3w1

        if _is_cpu and _USE_NATIVE_MXFP4_CPU:
            self._process_weights_cpu_native(layer)
            return

        if _is_cpu and _is_cpu_amx_available:
            self._process_weights_cpu(layer)
            return

        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        w13_w, w13_s = reorder_w1w3_to_w3w1(
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        layer.w13_weight = Parameter(w13_w, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_s, requires_grad=False)

        log_info_on_rank0(
            logger,
            f"Shuffling FP4 expert weights for TRT-LLM MxFP4 kernel "
            f"(layer: {self.prefix})...",
        )

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data
        num_experts = w13.shape[0]

        if w13_scale.dtype == torch.float32:
            w13_scale = w13_scale.to(torch.float8_e8m0fnu)
            w2_scale = w2_scale.to(torch.float8_e8m0fnu)

        epilogue_tile_m = 128
        g1_w, g1_s, g2_w, g2_s = [], [], [], []
        if _USE_OFFICIAL_SHUFFLE:
            cache: dict = {}
            for i in range(num_experts):
                w13_u8 = w13[i].view(torch.uint8)
                w13_s_u8 = w13_scale[i].view(torch.uint8)
                w2_u8 = w2[i].view(torch.uint8)
                w2_s_u8 = w2_scale[i].view(torch.uint8)

                perm = _maybe_get_cached_w3_w1_permute_indices(
                    cache,
                    w13_u8,
                    epilogue_tile_m,
                )
                g1_w.append(w13_u8[perm.to(w13_u8.device)].contiguous())
                perm_sf = _maybe_get_cached_w3_w1_permute_indices(
                    cache,
                    w13_s_u8,
                    epilogue_tile_m,
                    num_elts_per_sf=16,
                )
                g1_s.append(
                    block_scale_interleave(
                        w13_s_u8[perm_sf.to(w13_s_u8.device)].contiguous()
                    )
                )

                perm = get_w2_permute_indices_with_cache(
                    cache,
                    w2_u8,
                    epilogue_tile_m,
                )
                g2_w.append(w2_u8[perm.to(w2_u8.device)].contiguous())
                perm_sf = get_w2_permute_indices_with_cache(
                    cache,
                    w2_s_u8,
                    epilogue_tile_m,
                    num_elts_per_sf=16,
                )
                g2_s.append(
                    block_scale_interleave(
                        w2_s_u8[perm_sf.to(w2_s_u8.device)].contiguous()
                    )
                )
        else:
            for i in range(num_experts):
                g1_w.append(shuffle_matrix_a(w13[i].view(torch.uint8), epilogue_tile_m))
                g1_s.append(
                    shuffle_matrix_sf_a(w13_scale[i].view(torch.uint8), epilogue_tile_m)
                )
                g2_w.append(shuffle_matrix_a(w2[i].view(torch.uint8), epilogue_tile_m))
                g2_s.append(
                    shuffle_matrix_sf_a(w2_scale[i].view(torch.uint8), epilogue_tile_m)
                )

        layer.w13_weight = Parameter(torch.stack(g1_w), requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(
            torch.stack(g1_s)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, w13.shape[1], -1),
            requires_grad=False,
        )
        layer.w2_weight = Parameter(torch.stack(g2_w), requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(
            torch.stack(g2_s)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, w2.shape[1], -1),
            requires_grad=False,
        )

        if envs.SGLANG_OPT_MXFP4_STATIC_SCALE_ONES.get():
            self._register_static_scale_ones(layer)
        torch.cuda.empty_cache()

    def _apply_cpu(self, layer: Module, hidden_states, topk_output):
        from sglang.srt.layers.amx_utils import CPUQuantMethod
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input,
            topk_weights,
            hidden_states,
        )

        gemm1_clamp_limit = (
            float(self.moe_runner_config.swiglu_limit)
            if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B"
            and self.moe_runner_config.swiglu_limit is not None
            else None
        )

        # Mirror the GPU path's 2604B-path checker so the per-layer assertion
        # at deepseek_v4.py:1461 (observed == 1) passes on CPU too.
        if (
            envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B"
            and gemm1_clamp_limit is not None
        ):
            deepseek_v4_moe_code_path_checker.observed += 1

        if getattr(layer, "_native_mxfp4_cpu", False):
            from sglang.srt.layers.quantization.mxfp4_native_cpu import (
                native_fused_experts_bf16_cpu,
            )

            return native_fused_experts_bf16_cpu(
                hidden_states=x,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids.to(torch.int32),
                gemm1_clamp_limit=gemm1_clamp_limit,
            )

        return torch.ops.sgl_kernel.fused_experts_cpu(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids.to(torch.int32),
            False,  # inplace See [Note] inplace should be False in fused_experts.
            CPUQuantMethod.MXFP4,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            None,  # w1_zp
            None,  # w2_zp
            None,  # block_size
            None,  # w1_bias
            None,  # w2_bias
            None,  # gemm1_alpha
            gemm1_clamp_limit,
            True,  # is_vnni
        )

    def _process_weights_cpu_native(self, layer: Module) -> None:
        # Dequantize FP4 -> BF16 once here so the native forward is plain BF16
        # matmul. Drops the FP4 packed weights and block scales from the layer.
        from sglang.srt.layers.quantization.mxfp4_native_cpu import (
            dequantize_mxfp4,
        )

        w13_packed = layer.w13_weight.data.view(torch.uint8)
        w2_packed = layer.w2_weight.data.view(torch.uint8)
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data

        w13_bf16 = dequantize_mxfp4(w13_packed, w13_scale, torch.bfloat16).contiguous()
        w2_bf16 = dequantize_mxfp4(w2_packed, w2_scale, torch.bfloat16).contiguous()

        layer.w13_weight = Parameter(w13_bf16, requires_grad=False)
        layer.w2_weight = Parameter(w2_bf16, requires_grad=False)
        # Replace scale params with empty placeholders so loaders/refs don't
        # hold on to the original FP4 scale tensors.
        layer.w13_weight_scale_inv = Parameter(
            torch.empty(0, dtype=torch.float32), requires_grad=False
        )
        layer.w2_weight_scale_inv = Parameter(
            torch.empty(0, dtype=torch.float32), requires_grad=False
        )

        layer.use_intel_amx_backend = True
        layer._native_mxfp4_cpu = True

    def _process_weights_cpu(self, layer: Module) -> None:
        kernel = torch.ops.sgl_kernel

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data

        # Weights are int8-storage of packed FP4 (two values/byte). The CPU
        # kernel checks w scalar_type == uint8, so we view as uint8 (bit-equivalent).
        w13 = w13.view(torch.uint8)
        w2 = w2.view(torch.uint8)

        if w13_scale.dtype == torch.float32:
            w13_scale = w13_scale.to(torch.float8_e8m0fnu)
            w2_scale = w2_scale.to(torch.float8_e8m0fnu)
        w13_scale = w13_scale.view(torch.uint8)
        w2_scale = w2_scale.view(torch.uint8)

        # pack weight
        w13 = kernel.convert_weight_packed(w13)
        w2 = kernel.convert_weight_packed(w2)
        w13_scale = kernel.convert_scale_packed(w13_scale)
        w2_scale = kernel.convert_scale_packed(w2_scale)

        layer.w13_weight = Parameter(w13, requires_grad=False)
        layer.w2_weight = Parameter(w2, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_scale, requires_grad=False)

        layer.use_intel_amx_backend = True

    def _register_static_scale_ones(self, layer: Module) -> None:
        device = layer.w13_weight.device
        for name in (
            "output1_scale_scalar",
            "output1_scale_gate_scalar",
            "output2_scale_scalar",
        ):
            layer.register_buffer(
                name,
                torch.ones(layer.num_local_experts, device=device, dtype=torch.float32),
                persistent=False,
            )

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        from sglang.srt.utils import use_intel_amx_backend

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        if use_intel_amx_backend(layer):
            return StandardCombineInput(
                hidden_states=self._apply_cpu(layer, hidden_states, topk_output)
            )

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale_inv
        w2_scale = layer.w2_weight_scale_inv

        intermediate_size = w2.shape[2] * 2 if w2.dtype == torch.uint8 else w2.shape[2]
        hidden_size = w13.shape[2] * 2 if w13.dtype == torch.uint8 else w13.shape[2]

        num_local_experts = layer.num_local_experts
        if w13_scale.dim() == 2:
            w13_scale = w13_scale.reshape(num_local_experts, 2 * intermediate_size, -1)
        if w2_scale.dim() == 2:
            w2_scale = w2_scale.reshape(num_local_experts, hidden_size, -1)

        if TopKOutputChecker.format_is_standard(topk_output):
            topk_ids = topk_output.topk_ids
            topk_weights = topk_output.topk_weights
        elif TopKOutputChecker.format_is_bypassed(topk_output):
            raise NotImplementedError(
                "the old code in this branch is WRONG. e.g. it does not consider HashTopK, and may miss args"
            )
        else:
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        if not envs.SGLANG_OPT_MXFP4_SKIP_DISPATCHER_MAPPING.get():
            local_expert_offset = layer.moe_ep_rank * layer.num_local_experts
            topk_ids = torch.where(
                topk_ids >= 0,
                topk_ids + local_expert_offset,
                topk_ids,
            )
        packed_topk = PackTopkIds.execute(topk_ids, topk_weights)

        precision = self.flashinfer_mxfp4_moe_precision
        if precision == "bf16":
            assert hidden_states.dtype == torch.bfloat16
            x_quant = hidden_states
            x_scale = None
            origin_dim = x_quant.shape[-1]
            if hidden_size != origin_dim:
                x_quant = torch.nn.functional.pad(
                    x_quant,
                    (0, hidden_size - origin_dim),
                    mode="constant",
                    value=0.0,
                )
        elif precision == "default":
            x_quant, x_scale = mxfp8_quantize(
                hidden_states, False, alignment=hidden_size
            )
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            )
        else:
            raise NotImplementedError(f"Unsupported mxfp4 moe precision: {precision}")

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            num_tokens = x_quant.shape[0]
            out_hidden_size = (
                x_quant.shape[-1] * 2
                if x_quant.dtype == torch.uint8
                else x_quant.shape[-1]
            )
            symm_output = torch.empty(
                num_tokens, out_hidden_size, dtype=torch.bfloat16, device=x_quant.device
            )

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B" and (
            self._gemm1_clamp_limit_tensor is not None
        ):
            deepseek_v4_moe_code_path_checker.observed += 1

        output = trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            gemm1_weights=w13,
            gemm1_weights_scale=w13_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=self._gemm1_clamp_limit_tensor,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale,
            gemm2_bias=None,
            output1_scale_scalar=(
                layer.output1_scale_scalar
                if envs.SGLANG_OPT_MXFP4_STATIC_SCALE_ONES.get()
                else torch.ones(
                    num_local_experts, device=x_quant.device, dtype=torch.float32
                )
            ),
            output1_scale_gate_scalar=(
                layer.output1_scale_gate_scalar
                if envs.SGLANG_OPT_MXFP4_STATIC_SCALE_ONES.get()
                else torch.ones(
                    num_local_experts, device=x_quant.device, dtype=torch.float32
                )
            ),
            output2_scale_scalar=(
                layer.output2_scale_scalar
                if envs.SGLANG_OPT_MXFP4_STATIC_SCALE_ONES.get()
                else torch.ones(
                    num_local_experts, device=x_quant.device, dtype=torch.float32
                )
            ),
            num_experts=layer.num_experts,
            top_k=packed_topk.shape[1],
            n_group=1,
            topk_group=1,
            intermediate_size=intermediate_size,
            local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            local_num_experts=num_local_experts,
            routed_scaling_factor=1.0,
            routing_method_type=int(RoutingMethodType.TopK),
            do_finalize=True,
            tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
            output=symm_output,
        )[0]

        if not envs.SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD.get():
            rsf = layer.moe_runner_config.routed_scaling_factor
            if rsf is not None and rsf != 1.0:
                output.mul_(rsf)

        return StandardCombineInput(hidden_states=output)

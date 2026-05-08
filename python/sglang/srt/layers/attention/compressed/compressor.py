from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, NamedTuple, Optional, Tuple, Union

import torch

from sglang.jit_kernel.deepseek_v4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_fused_norm_rope_inplace,
    triton_create_paged_compress_data,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
    quant_to_nope_fp8_rope_bf16_pack
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.layers.attention.nsa.utils import (
    assert_tensor_identical_across_cp_ranks,
)
from sglang.srt.utils import cpu_has_amx_support, is_cpu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.compressed.metadata import DeepseekV4Metadata
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v4 import Compressor, DeepseekRefRMSNorm


_is_cpu_amx_available = is_cpu() and cpu_has_amx_support()


def act_quant_pytorch(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization (PyTorch native).
    This is a pure PyTorch implementation equivalent to the Triton kernel version.
    It performs per-block FP8 quantization along the last dimension.
    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and
            its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks used for quantization.
            Default is 128.
        scale_fmt (Optional[str], optional): If not None, scales are rounded to
            powers of 2. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # FP8 e4m3fn constants
    fp8_max = 448.0
    fp8_min = -448.0

    # Flatten all dims except last
    orig_shape = x.shape
    N = x.size(-1)
    x_flat = x.view(-1, N).float()  # (M, N)
    M = x_flat.size(0)
    num_groups = N // block_size

    # Reshape into blocks: (M, num_groups, block_size)
    x_blocked = x_flat.view(M, num_groups, block_size)

    # Compute per-block absolute max -> (M, num_groups)
    amax = x_blocked.abs().amax(dim=2)

    # Clamp to avoid division by zero
    amax = amax.clamp(min=1e-4)

    # Compute scale
    round_scale = scale_fmt is not None
    if round_scale:
        # Round scale to nearest power of 2 (ceiling in log2 space)
        scale = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))
    else:
        scale = amax / fp8_max

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    # scale shape: (M, num_groups) -> broadcast to (M, num_groups, block_size)
    y = x_blocked / scale.unsqueeze(2)
    y = y.clamp(fp8_min, fp8_max)

    # Reshape output back to original shape and cast to fp8
    y = y.view(orig_shape).to(torch.float8_e4m3fn)

    # Reshape scale to match expected output shape: (*orig_shape[:-1], num_groups)
    s = scale.view(*orig_shape[:-1], num_groups)

    return y, s


def act_quant_cpu(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.sgl_kernel.act_quant_cpu(x, block_size, scale_fmt)


act_quant = act_quant_cpu if _is_cpu_amx_available else act_quant_pytorch

class FusedCompressMetadata(NamedTuple):
    write_loc: torch.Tensor
    extra_data: Optional[torch.Tensor]
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan]

    def copy_(self, other: FusedCompressMetadata) -> None:
        from .metadata import maybe_copy_inplace

        self.write_loc.copy_(other.write_loc)
        maybe_copy_inplace(self.extra_data, src=other.extra_data)
        self.plan.copy_(other.plan)


class CompressorBackend:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DeepseekV4Metadata

    def get_paged_compress_metadata(self, compress_ratio: int) -> FusedCompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        metadata = getattr(self.forward_metadata, attr_name)
        assert isinstance(metadata, FusedCompressMetadata)
        return metadata

    def forward_compress(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: DeepseekRefRMSNorm,
        freqs_cis_cache: torch.Tensor,
        rotate: bool,
        forward_batch: ForwardBatch,
        compress_ratio: int,
        is_paged: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        assert compress_ratio == 4 or compress_ratio == 128
        if is_paged:
            metadata = self.get_paged_compress_metadata(compress_ratio)
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            last_dim = 2 * head_dim * coff
            assert kv_score_buffer.shape[-1] == last_dim
            kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)
        else:
            plan = make_compressor_plan(compress_ratio, forward_batch)
            metadata = (forward_batch.req_pool_indices.to(torch.int32), None, plan)
        indices, extra_data, plan = metadata

        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape,
            indices=indices,
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            extra_data=extra_data,
        )
        compress_fused_norm_rope_inplace(
            kv_compressed,
            norm.weight,
            norm.eps,
            freqs_cis_cache,
            plan,
        )
        return rotate_activation(kv_compressed) if rotate else kv_compressed

    def forward_core_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: the concrete backend (DeepseekV4BackendRadix)
        # owns this helper. MQALayer._forward_prepare calls us before
        # attn_backend.forward(), so Raw -> Radix must happen here too
        # (e.g. 1.6T layer 0 has compress_ratio=128 and needs cX_compress_metadata).
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)
        if envs.SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY.get():
            assert_tensor_identical_across_cp_ranks(
                new_compressed_kv,
                tag=f"compressor(ratio={compressor.ratio}) layer_id={layer_id}",
                forward_batch=forward_batch,
            )
        core_metadata = self.forward_metadata.core_metadata
        out_loc = (
            core_metadata.c4_out_loc
            if compressor.ratio == 4
            else core_metadata.c128_out_loc
        )
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            #pack = quant_to_nope_fp8_rope_bf16_pack_triton(new_compressed_kv.bfloat16())
            pack = quant_to_nope_fp8_rope_bf16_pack(new_compressed_kv.bfloat16())
            token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)

    def forward_indexer_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        assert is_overlap_compress(compressor.ratio)
        # PREP_IN_CG lazy upgrade (see forward_core_compressor for rationale).
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)
        if envs.SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY.get():
            assert_tensor_identical_across_cp_ranks(
                new_compressed_kv,
                tag=f"indexer_compressor(ratio={compressor.ratio}) layer_id={layer_id}",
                forward_batch=forward_batch,
            )
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_index_k_fused(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            new_compressed_kv_fp8, new_compressed_kv_scale = act_quant(
                new_compressed_kv
            )
            token_to_kv_pool.set_index_k_scale_buffer(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                index_k=new_compressed_kv_fp8,
                index_k_scale=new_compressed_kv_scale,
            )


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def make_compressor_plan(
    compress_ratio: Literal[4, 128],
    forward_batch: ForwardBatch,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if forward_batch.forward_mode.is_decode():
        seq_lens_32 = forward_batch.seq_lens.to(torch.int32)
        return CompressorDecodePlan(compress_ratio, seq_lens_32)
    if forward_batch.forward_mode.is_prefill():
        assert not forward_batch.forward_mode.is_target_verify()
        extend_lens_list = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert extend_lens_list is not None and seq_lens_cpu is not None
        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            num_q_tokens=sum(extend_lens_list),
            seq_lens=seq_lens_cpu,
            extend_lens=torch.tensor(extend_lens_list),
            device=forward_batch.seq_lens.device,
        )
    elif forward_batch.forward_mode.is_target_verify():
        raise NotImplementedError("target verify mode to be implemented")
    else:
        raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")


from typing import Tuple
import torch


def pytorch_create_paged_compress_data(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = req_pool_indices.shape[0]
    device = req_pool_indices.device

    rid = req_pool_indices.to(torch.int32)
    seq_len = seq_lens[:batch_size].to(torch.int32)
    extend_len = extend_seq_lens[:batch_size].to(torch.int32)
    prefix_len = seq_len - extend_len

    cr = compress_ratio
    write_pos = ((seq_len - 1) // cr) * cr
    load_pos = ((prefix_len - 1) // cr) * cr
    write_overlap_pos = write_pos - cr
    load_overlap_pos = load_pos - cr

    def compute_state_loc(pos: torch.Tensor) -> torch.Tensor:
        pos = pos.clamp(min=0)
        loc = req_to_token[rid, pos].to(torch.int32)
        swa_loc = full_to_swa_index_mapping[loc].to(torch.int32)
        swa_page = swa_loc // swa_page_size
        state_loc = swa_page * ring_size + (swa_loc % ring_size)
        state_loc = state_loc // cr
        return state_loc

    v0 = compute_state_loc(load_pos)           # i == 0
    v1 = compute_state_loc(write_pos)          # i == 1
    v2 = compute_state_loc(load_overlap_pos)   # i == 2
    v3 = compute_state_loc(write_overlap_pos)  # i == 3

    out_0 = v1.clone()

    if is_overlap:
        out_1 = torch.stack([v2, v0, v3, write_pos.to(torch.int32)], dim=1)
    else:
        out_1 = v0.clone()

    return out_0, out_1

def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> FusedCompressMetadata:
    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    assert ring_size % compress_ratio == 0

    def clip_down(positions: torch.Tensor) -> torch.Tensor:
        return positions // compress_ratio * compress_ratio

    def get_raw_loc(positions: torch.Tensor) -> torch.Tensor:
        positions = positions.masked_fill(positions < 0, 0)
        loc = req_to_token[req_pool_indices, positions]
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(loc)
        swa_pages = swa_loc // swa_page_size
        state_loc = swa_pages * ring_size + swa_loc % ring_size
        return (state_loc // compress_ratio).to(torch.int32)

    is_overlap = is_overlap_compress(compress_ratio)

    if is_prefill:
        assert extend_lens is not None
        # write_loc, extra_data = triton_create_paged_compress_data(
        write_loc, extra_data = pytorch_create_paged_compress_data(
            compress_ratio=compress_ratio,
            is_overlap=is_overlap,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
        )

        plan_kwargs: dict
        if seq_lens_cpu is None:
            assert num_q_tokens is not None
            plan_kwargs = dict(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
            )
        else:
            assert extend_lens_cpu is not None
            plan_kwargs = dict(
                num_q_tokens=sum(extend_lens_cpu),
                seq_lens=torch.tensor(seq_lens_cpu),
                extend_lens=torch.tensor(extend_lens_cpu),
            )
        plan = CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            device=seq_lens.device,
            use_cuda_graph=use_prefill_cuda_graph,
            **plan_kwargs,
        )
    else:
        write_positions = clip_down(seq_lens - 1)
        write_loc = get_raw_loc(write_positions)
        if is_overlap:
            write_overlap_loc = get_raw_loc(write_positions - compress_ratio)
            extra_data = write_overlap_loc.view(-1, 1)
        else:
            extra_data = None
        plan = CompressorDecodePlan(compress_ratio, seq_lens.to(torch.int32))

    return FusedCompressMetadata(write_loc=write_loc, extra_data=extra_data, plan=plan)

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.distributed import get_tensor_model_parallel_rank

logger = logging.getLogger(__name__)
class IntelAMXAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        from sgl_kernel.cpu import decode_attention, extend_attention

        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

        self.num_head = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )

        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.decode_attention_fwd = decode_attention
        self.extend_attention_fwd = extend_attention

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        bs = forward_batch.batch_size
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                8,  # self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = None
        else:
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
        self.forward_metadata = (attn_logits, max_extend_len)

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        attn_logits, _ = self.forward_metadata

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        import copy
        q_ref = copy.deepcopy(q)
        k_ref = copy.deepcopy(forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id))
        v_ref = copy.deepcopy(forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id))
        o_ref = copy.deepcopy(o)
        k_raw_ref = copy.deepcopy(k)
        v_raw_ref = copy.deepcopy(v)
        out_cache_loc_ref = copy.deepcopy(forward_batch.out_cache_loc)
        req_to_token_ref = copy.deepcopy(forward_batch.req_to_token_pool.req_to_token)
        req_pool_indices_ref = copy.deepcopy(forward_batch.req_pool_indices)
        seq_lens_ref = copy.deepcopy(forward_batch.seq_lens)
        attn_logits_ref = copy.deepcopy(attn_logits)
        scaling_ref = copy.deepcopy(layer.scaling)
        logit_cap_ref = copy.deepcopy(layer.logit_cap)
        
        print(f"{forward_batch.out_cache_loc=}", flush=True)
        
        # k_buffer_size = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).size()
        # k_buffer_stride = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).stride()
        
        # v_buffer_size = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).size()
        # v_buffer_stride = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).stride()
        
        # req_to_token_size = forward_batch.req_to_token_pool.req_to_token.size()
        # req_to_token_stride = forward_batch.req_to_token_pool.req_to_token.stride()
        
        # logger.info(
        #     f"{q.size()=},"
        #     f"{q.stride()=},"
        #     f"{layer.tp_q_head_num=},"
        #     f"{layer.qk_head_dim=},"
        #     f"{k_buffer_size=},"
        #     f"{k_buffer_stride=},"
        #     f"{v_buffer_size=},"
        #     f"{v_buffer_stride=}"    
        #     f"{o.size()=},"
        #     f"{o.stride()=},"                
        #     f"{layer.v_head_dim=},"
        #     f"{k.size()=},"
        #     f"{k.stride()=},"
        #     f"{v.size()=},"
        #     f"{v.stride()=},"
        #     f"{forward_batch.out_cache_loc=},"
        #     f"{req_to_token_size=},"
        #     f"{req_to_token_stride=},"
        #     f"{forward_batch.req_pool_indices=},"
        #     f"{forward_batch.seq_lens=},"
        #     f"{attn_logits.size()=},"
        #     f"{attn_logits.stride()=},"
        #     f"{layer.scaling=},"
        #     f"{layer.logit_cap=},"
        # )          
        
        import time
        import os
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tp_rank = get_tensor_model_parallel_rank()
        debug_filename = f"/home/chunyuan/sglang-dev/debug-inputs/debug_inputs_{timestamp}_{tp_rank}.pt"
        torch.save({
            "q": q,
            "k": forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            "v": forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            "o": o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            "k_raw": k,
            "v_raw": v,
            "out_cache_loc": forward_batch.out_cache_loc,
            "req_to_token": forward_batch.req_to_token_pool.req_to_token,
            "req_pool_indices": forward_batch.req_pool_indices,
            "seq_lens": forward_batch.seq_lens,
            "attn_logits": attn_logits,
        }, debug_filename)        
        
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k,
            v,
            forward_batch.out_cache_loc,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            attn_logits,
            layer.scaling,
            layer.logit_cap,
        )
        logger.info(f"o contains NaN: {torch.isnan(o).any()}")
        if torch.isnan(o).any():
            logger.warning(f"NaN detected in output. Inputs saved to: {debug_filename}")
        else:
            os.remove(debug_filename)  # Optional cleanup


        # self.decode_attention_fwd(
        #     q_ref.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        #     k_ref,
        #     v_ref,
        #     o_ref.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        #     k_raw_ref,
        #     v_raw_ref,
        #     out_cache_loc_ref,
        #     req_to_token_ref,
        #     req_pool_indices_ref,
        #     seq_lens_ref,
        #     attn_logits_ref,
        #     scaling_ref,
        #     logit_cap_ref,
        # )
        
        # orig = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).data_ptr() == forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).data_ptr()
        # copied = k_ref.data_ptr() == v_ref.data_ptr()
        
        # print(f"o is: {o}, o_ref is {o_ref}", flush=True)
        
        # print(
            # f"orign ptr equals: {orig}, copied ptr equals: {copied}", flush=True
        # )
        
        # logger.info(f"")
        
        
        # cos_sim = torch.nn.functional.cosine_similarity(
        #     o.flatten(), o_ref.flatten(), dim=0
        # )
        # print("cos_sim = ", cos_sim.item(), " > 0.99: ",  cos_sim.item() > 0.99)
        # print("allclose: ", torch.allclose(o, o_ref, atol=3e-2))
        # print("comparing k_buffer: ", torch.equal(o, o_ref), "; diff sum: ", (o - o_ref).abs().sum().item())        
        
        # assert not torch.isnan(o_ref).any(), "nan in o_ref"
        assert not torch.isnan(o).any(), "nan in o"
        # assert torch.allclose(o, o_ref), "Mismatch between o and o_ref"

        return o

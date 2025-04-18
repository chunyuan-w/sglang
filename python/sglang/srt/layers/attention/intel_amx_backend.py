from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


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
        q_tensor: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        
        attn_logits, _ = self.forward_metadata # [1, 16, 8, 513]

        q_tensor = q_tensor.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim) # [[1, 16, 576]] -> [1, 9216]

        if layer.qk_head_dim != layer.v_head_dim: # 576, 512
            o = q_tensor.new_empty((q_tensor.shape[0], layer.tp_q_head_num * layer.v_head_dim)) # [1, 8192 (16 * 512)]
        else:
            o = torch.empty_like(q_tensor)

        self.decode_attention_fwd(
            q_tensor.view(-1, layer.tp_q_head_num, layer.qk_head_dim), # [1, 16, 576]
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id), # [44438539, 1, 576]
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id), # [44438539, 1, 512]
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim), # [1, 16, 512]
            k, # [1, 1, 576]
            v, # [1, 1, 512]
            forward_batch.out_cache_loc, # [1]
            forward_batch.req_to_token_pool.req_to_token, # [4097, 163844]
            forward_batch.req_pool_indices, # [1]
            forward_batch.seq_lens, # [1]
            attn_logits, # [1, 16, 8, 513]
            layer.scaling, # 0.1147213867929261
            layer.logit_cap, # 0.0
        )

        return o

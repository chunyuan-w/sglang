#pragma once

#include <ATen/ATen.h>

void decode_attention_cpu(at::Tensor& query, at::Tensor& k_cache, at::Tensor& v_cache, at::Tensor& output,
    at::Tensor& key, at::Tensor& value, at::Tensor& loc, at::Tensor& attn_logits,
    at::Tensor& req_to_token, at::Tensor& req_pool_indices, at::Tensor& seq_lens,
    double sm_scale, double logit_cap);
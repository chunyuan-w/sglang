#include "../common.h"
#include "../vec.h"
#include "../interface.h"

at::Tensor forward_absorb_cpu(
    at::Tensor& query,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    // at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& w_vc,
    double sm_scale,
    double logit_cap,
    int tp_k_head_num,
    int qk_head_dim,
    int tp_v_head_num,
    int v_head_dim,
    int tp_q_head_num,
    int num_local_heads,
    int kv_lora_rank,
    bool is_vnni,
    std::optional<at::Tensor>& scale) {
  // TODO: allocate o in cpp or in python?
  
// TODO: is the below code needed for R1?
//   if k is not None:
//     // For cross-layer sharing, kv can be None
//     assert v is not None
//     k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
//     v = v.view(-1, self.tp_v_head_num, self.v_head_dim)


    // q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

    // if layer.qk_head_dim != layer.v_head_dim:
    //     o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
    // else:
    //     o = torch.empty_like(q)
  query= query.reshape({-1, tp_q_head_num * qk_head_dim});
  at::Tensor attn_output;
  if (qk_head_dim != v_head_dim) {
      attn_output = at::empty({query.size(0), tp_q_head_num * v_head_dim}, query.options());
  } else {
      attn_output = at::empty_like(query);
  }
  auto query_3d = query.view({-1, tp_q_head_num, qk_head_dim});
  auto o_3d = attn_output.view({-1, tp_q_head_num, v_head_dim});

  decode_attention_cpu(
    query_3d,
    k_cache,
    v_cache,
    o_3d,
    key,
    value,
    loc,
    attn_logits,
    req_to_token,
    req_pool_indices,
    seq_lens,
    sm_scale,
    logit_cap);

  // attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
  attn_output = attn_output.view({-1, num_local_heads, kv_lora_rank});
  int64_t B = w_vc.sizes()[0];
  int64_t N = w_vc.sizes()[1];
  int64_t M = attn_output.sizes()[0];
  
  at::Tensor output = at::empty({M, B * N}, attn_output.options());
  at::Tensor attn_bmm_output = output.view({M, B, N}).transpose_(0, 1);
  
  attn_output = attn_output.transpose(0, 1);
  // TODO: is_vnni: set it as an arg to this OP
  bmm_cpu(attn_bmm_output, attn_output, w_vc, is_vnni, scale);

  return output;
}

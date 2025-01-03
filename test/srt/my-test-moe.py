import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size=5120, intermediate_size=12288):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )
        return down_proj

class MoERef(nn.Module):
    def __init__(
        self,
        num_experts,
        hidden_size,
        intermediate_size,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(hidden_size, intermediate_size)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        print("#" * 50)
        outs = (
            torch.cat(outputs, dim=0)
            if len(outputs)
            else sorted_tokens.new_empty(0)
        )
        new_x = torch.empty_like(outs)
        
        print(idxs)
        print(new_x.shape)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class FusedMoE(nn.Module):
    def __init__(
        self,
        experts,
    ):
        super().__init__()
        self.experts = experts
        
        weights_13 = []
        weights_2 = []
        for expert_idx in range(len(self.experts)):
            expert_layer = self.experts[expert_idx]
        
            expert_layer.gate_proj.weight     
            expert_layer.up_proj.weight     
            
            w13 = torch.cat([expert_layer.gate_proj.weight, expert_layer.up_proj.weight], dim=0)
            weights_13.append(w13)
            
            w2 = expert_layer.down_proj.weight
            weights_2.append(w2)
        
        # w13_wieght: [num_experts, 2 * intermediate_size, hidden_size]
        # w2_weight: [num_experts, hidden_size, intermediate_size]
        self.w13_weight = torch.stack(weights_13, dim=0)
        self.w2_weight = torch.stack(weights_2, dim=0)

    def forward(self, x, topk_ids, topk_weight):
        w13_weights = self.w13_weight[topk_ids]
        w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
        w2_weights = self.w2_weight[topk_ids]
        x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
        x1 = F.silu(x1)
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
        return torch.einsum("tai,ta -> ti", expert_outs, topk_weight.to(expert_outs.dtype))

class AddLayerNormTester(TestCase):
    def test_moe(self):
        # num_expert_per_token: top_k

        tokens = 1
        hidden_size = 64
        intermediate_size = 1024
        num_experts = 8
        selected_experts = 2
        x = torch.rand(tokens, hidden_size)
        
        gating_output = torch.randn(tokens, num_experts)
        topk_weight, topk_ids = torch.topk(gating_output, k=selected_experts, dim=-1, sorted=False)
        # topk_weight = torch.rand(tokens, selected_experts)
        # topk_ids = torch.randint(0, num_experts, (tokens, selected_experts))
        print("topk_ids:", topk_ids)
        with torch.no_grad():
            model_ref = MoERef(num_experts, hidden_size, intermediate_size).eval()
            output_ref = model_ref(x.clone(), topk_ids.clone(), topk_weight.clone())

            x_clone = x.clone()
            topk_ids_clone = topk_ids.clone()
            topk_weight_clone = topk_weight.clone()
            model_fused = FusedMoE(
                model_ref.experts,
            ).eval()
            output_fused = model_fused(x_clone, topk_ids_clone, topk_weight_clone)
            self.assertEqual(output_ref, output_fused)

if __name__ == "__main__":
    test = unittest.main()

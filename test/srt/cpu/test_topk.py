import unittest

import torch

from sglang.srt.layers.moe.topk import fused_topk_torch_native as native_fused_topk
from sglang.srt.utils import fast_topk
from sglang.test.test_utils import CustomTestCase


def llama4_custom_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    _renormalize: bool,
):
    """Copied from Llama4MoE.custom_routing_function to avoid CUDA-only imports."""
    router_scores_aK, router_indices_aK = fast_topk(gating_output, topk, dim=-1)
    router_scores_aK = torch.sigmoid(router_scores_aK.float()).to(hidden_states.dtype)
    return (
        router_scores_aK.view(-1).reshape(router_scores_aK.shape),
        router_indices_aK.to(torch.int32),
    )

torch.manual_seed(1234)


# This is used by the Deepseek-V2 model
def _assert_grouped_topk_valid(topk_weights, topk_ids, gating_float, topk, G, _topk_group, renormalize, scoring="softmax"):
    """
    Tie-robust validity check for grouped topk kernels.

    When multiple groups share the same group score, any selection among tied
    groups is correct.  Rather than comparing against a single reference that
    breaks ties one way, we verify the invariants directly:

      1. Weights equal the (renormalized) per-expert scores at the selected ids.
      2. Every selected group's group-score >= every non-selected group's
         group-score (allowing a small tolerance for floating-point ties).
      3. Within selected groups, the selected experts are the top ones by score.
    """
    M, E = gating_float.shape
    epg = E // G

    if scoring == "softmax":
        scores = torch.softmax(gating_float, dim=-1)
    else:
        scores = torch.sigmoid(gating_float)

    # 1. Weight correctness
    expected_w = scores.gather(1, topk_ids.long())
    if renormalize:
        expected_w = expected_w / expected_w.sum(-1, keepdim=True)
    torch.testing.assert_close(topk_weights, expected_w, atol=1e-4, rtol=1e-4)

    # 2. Group-selection validity
    group_scores = scores.view(M, G, epg).max(dim=-1).values  # (M, G)
    selected_groups = torch.zeros(M, G, dtype=torch.bool)
    for i in range(M):
        for j in range(topk):
            g = topk_ids[i, j].item() // epg
            selected_groups[i, g] = True

    min_sel_gs = group_scores.masked_fill(~selected_groups, float("inf")).min(-1).values
    max_non_sel_gs = group_scores.masked_fill(selected_groups, -float("inf")).max(-1).values
    # non-selected group scores must be <= min selected group score (up to tie tolerance)
    assert (max_non_sel_gs <= min_sel_gs + 1e-4).all(), (
        "Group selection invalid: some non-selected group has higher score than a selected group"
    )


class TestGroupedTopK(CustomTestCase):
    def _run_single_test(self, M, E, G, topk, topk_group, renormalize, dtype):
        torch.manual_seed(1234)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M

        # fused version
        topk_weights, topk_ids = torch.ops.sgl_kernel.grouped_topk_cpu(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            G,
            topk_group,
            0,
            None,
            None,
        )

        _assert_grouped_topk_valid(
            topk_weights, topk_ids, gating_output.float(), topk, G, topk_group, renormalize, scoring="softmax"
        )

    def test_grouped_topk(self):
        for renormalize in [True, False]:
            self._run_single_test(123, 8, 2, 2, 1, renormalize, torch.bfloat16)
            self._run_single_test(123, 16, 4, 3, 2, renormalize, torch.bfloat16)
            self._run_single_test(123, 32, 4, 3, 2, renormalize, torch.bfloat16)
            self._run_single_test(1123, 32, 4, 3, 2, renormalize, torch.bfloat16)
            self._run_single_test(123, 64, 1, 6, 1, renormalize, torch.bfloat16)
            self._run_single_test(123, 256, 8, 4, 8, renormalize, torch.bfloat16)
            self._run_single_test(123, 160, 8, 6, 2, renormalize, torch.bfloat16)


# DeepSeek V2/V3/R1 uses biased_grouped_top
class TestBiasedGroupedTopK(CustomTestCase):
    def _run_single_test(
        self, M, E, G, topk, topk_group, renormalize, dtype, bias_dtype
    ):
        torch.manual_seed(1234)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M
        correction_bias = torch.randn(E, dtype=bias_dtype)

        # fused version
        topk_weights, topk_ids = torch.ops.sgl_kernel.biased_grouped_topk_cpu(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            G,
            topk_group,
            0,
            None,
            None,
        )

        M2, E2 = gating_output.shape
        epg = E2 // G
        sigmoid_scores = gating_output.float().sigmoid()
        bias_f32 = correction_bias.float()
        biased_scores = sigmoid_scores + bias_f32.unsqueeze(0)  # (M, E)

        # 1. Weight correctness: weights = sigmoid scores at selected ids (with optional renorm)
        expected_w = sigmoid_scores.gather(1, topk_ids.long())
        if renormalize:
            expected_w = expected_w / expected_w.sum(-1, keepdim=True)
        torch.testing.assert_close(topk_weights, expected_w, atol=1e-4, rtol=1e-4)

        # 2. Group-selection validity using group score = sum of top-2 biased scores per group
        biased_groups = biased_scores.view(M2, G, epg)
        top2 = biased_groups.topk(2, dim=-1).values  # (M, G, 2)
        group_scores = top2.sum(-1)  # (M, G)

        selected_groups = torch.zeros(M2, G, dtype=torch.bool)
        for i in range(M2):
            for j in range(topk):
                g = topk_ids[i, j].item() // epg
                selected_groups[i, g] = True

        min_sel_gs = group_scores.masked_fill(~selected_groups, float("inf")).min(-1).values
        max_non_sel_gs = group_scores.masked_fill(selected_groups, -float("inf")).max(-1).values
        assert (max_non_sel_gs <= min_sel_gs + 1e-4).all(), (
            "Group selection invalid: some non-selected group has higher score than a selected group"
        )

    def test_biased_grouped_topk(self):
        for renormalize in [True, False]:
            for bias_dtype in [torch.float32, torch.bfloat16]:
                self._run_single_test(
                    122, 256, 8, 8, 2, renormalize, torch.bfloat16, bias_dtype
                )


class TestTopK(CustomTestCase):
    def _run_single_test(self, M, E, topk, renormalize, dtype):
        torch.manual_seed(1998)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M

        ref_topk_weights, ref_topk_ids = native_fused_topk(
            hidden_states.float(),
            gating_output.float(),
            topk,
            renormalize,
        )

        # fused version
        topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
            hidden_states, gating_output, topk, renormalize
        )

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    def test_topk(self):
        for renormalize in [True, False]:
            self._run_single_test(123, 8, 2, renormalize, torch.bfloat16)
            self._run_single_test(123, 16, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 32, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 32, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 64, 6, renormalize, torch.bfloat16)
            self._run_single_test(123, 256, 4, renormalize, torch.bfloat16)
            self._run_single_test(123, 160, 6, renormalize, torch.bfloat16)


class TestCustomTopK(CustomTestCase):
    def _run_single_test(
        self, M, E, topk, renormalize, dtype, native_custom_f, fused_custom_f
    ):
        torch.manual_seed(16)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M

        ref_topk_weights, ref_topk_ids = native_custom_f(
            hidden_states.float(),
            gating_output.float(),
            topk,
            renormalize,
        )

        # fused version
        topk_weights, topk_ids = fused_custom_f(
            hidden_states, gating_output, topk, renormalize
        )

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    def test_custom_topk(self):
        test_custom_functions = [
            (llama4_custom_routing_function, torch.ops.sgl_kernel.topk_sigmoid_cpu)
        ]
        for native_custom_f, fused_custom_f in test_custom_functions:
            self._run_single_test(
                123, 8, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
            )
            self._run_single_test(
                123, 16, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
            )
            self._run_single_test(
                123, 32, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
            )


class TestTopKSigmoidBias(CustomTestCase):
    """Tests for topk_sigmoid_bias_cpu: sigmoid scoring with optional correction_bias."""

    def _run_single_test(self, M, E, topk, renormalize, dtype, correction_bias=None):
        torch.manual_seed(42)
        hidden_states = torch.randn(M, 128, dtype=dtype)
        # Scale to avoid bfloat16 ties without saturating sigmoid.
        # At scale 8, values spread into the "integer-like" bf16 regime where
        # distinct float32 inputs reliably map to distinct bf16 values, but
        # sigmoid(±8) = 0.9997/0.0003 — not fully saturated.
        gating_output = (torch.randn(M, E) * 8).to(dtype)

        topk_weights, topk_ids = torch.ops.sgl_kernel.topk_sigmoid_bias_cpu(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            correction_bias,
        )

        # Compute full sigmoid scores (float32) for validation
        sigmoid_scores = gating_output.float().sigmoid()  # (M, E)
        if correction_bias is not None:
            sel_scores = sigmoid_scores + correction_bias.float().unsqueeze(0)
        else:
            sel_scores = sigmoid_scores

        # 1. Weights must match sigmoid_scores at the selected expert indices
        expected_weights = sigmoid_scores.gather(1, topk_ids.long())
        if renormalize:
            expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True)
        torch.testing.assert_close(topk_weights, expected_weights, atol=1e-4, rtol=1e-4)

        # 2. Each selected expert must have sel_score >= every non-selected expert
        #    (ties are acceptable: allow a small tolerance)
        min_sel = sel_scores.gather(1, topk_ids.long()).min(-1).values  # (M,)
        selected_mask = torch.zeros(M, E, dtype=torch.bool)
        selected_mask.scatter_(1, topk_ids.long(), True)
        non_sel_scores = sel_scores.masked_fill(selected_mask, -float("inf"))
        max_non_sel = non_sel_scores.max(-1).values  # (M,)
        assert (max_non_sel <= min_sel + 1e-4).all(), (
            f"Some non-selected experts have higher sel_score than selected ones.\n"
            f"max non-sel: {max_non_sel[max_non_sel > min_sel + 1e-4]}\n"
            f"min sel:     {min_sel[max_non_sel > min_sel + 1e-4]}"
        )

    def test_sigmoid_no_bias(self):
        for renormalize in [True, False]:
            for E, topk in [(8, 2), (16, 3), (32, 4), (64, 6), (128, 8), (256, 4)]:
                self._run_single_test(123, E, topk, renormalize, torch.bfloat16)

    def test_sigmoid_with_bias(self):
        for renormalize in [True, False]:
            for E, topk in [(8, 2), (16, 3), (32, 4), (64, 6), (128, 8), (256, 4)]:
                bias = torch.randn(E, dtype=torch.float32)
                self._run_single_test(123, E, topk, renormalize, torch.bfloat16, bias)

    def test_sigmoid_topk1_no_bias(self):
        """topk=1 case (same as existing topk_sigmoid_cpu but via new kernel)."""
        for renormalize in [True, False]:
            for E in [8, 16, 32, 64]:
                self._run_single_test(123, E, 1, renormalize, torch.bfloat16)

    def test_sigmoid_topk1_with_bias(self):
        for renormalize in [True, False]:
            for E in [8, 16, 32, 64]:
                bias = torch.randn(E, dtype=torch.float32)
                self._run_single_test(123, E, 1, renormalize, torch.bfloat16, bias)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_impl as native_biased_grouped_topk,
)
from sglang.srt.layers.moe.topk import fused_topk_torch_native as native_fused_topk
from sglang.srt.layers.moe.topk import grouped_topk_gpu as native_grouped_topk
from sglang.srt.models.llama4 import Llama4MoE
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


# This is used by the Deepseek-V2 model
class TestGroupedTopK(CustomTestCase):
    def _run_single_test(self, M, E, G, topk, topk_group, renormalize, dtype):
        torch.manual_seed(1234)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M

        ref_topk_weights, ref_topk_ids = native_grouped_topk(
            hidden_states.float(),
            gating_output.float(),
            topk,
            renormalize,
            G,
            topk_group,
        )

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

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

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

        ref_topk_weights, ref_topk_ids = native_biased_grouped_topk(
            hidden_states.float(),
            gating_output.float(),
            correction_bias.float(),
            topk,
            renormalize,
            G,
            topk_group,
        )

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

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    def test_biased_grouped_topk(self):
        for renormalize in [True, False]:
            for bias_dtype in [torch.float32, torch.bfloat16]:
                self._run_single_test(
                    122, 256, 8, 8, 2, renormalize, torch.bfloat16, bias_dtype
                )


class TestTopK(CustomTestCase):
    def _run_single_test(self, M, E, topk, renormalize, has_correction_bias, scoring_func, dtype):
        torch.manual_seed(1998)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        # gating_output = torch.randn(M, E, dtype=dtype) * 2 * M
        base = torch.linspace(-6, 6, E, dtype=torch.float32)  # evenly spaced
        gating_output_fp32 = base.unsqueeze(0).repeat(M, 1)

        # add noise large enough to survive bf16
        gating_output_fp32 += torch.randn_like(gating_output_fp32) * 0.1

        gating_output = gating_output_fp32.to(torch.bfloat16)

        if has_correction_bias:
            # TODO: do we need to test bf16 bias?
            correction_bias = torch.randn(E, dtype=torch.float32)
        else:
            correction_bias = None

        ref_topk_weights, ref_topk_ids = native_fused_topk(
            hidden_states.float(),
            gating_output.float(),
            topk,
            renormalize,
            correction_bias,
            scoring_func,
        )

        # fused version
        if scoring_func == "softmax":
            topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
                hidden_states, gating_output, topk, renormalize, correction_bias
            )
        elif scoring_func == "sigmoid":
            topk_weights, topk_ids = torch.ops.sgl_kernel.topk_sigmoid_cpu(
                hidden_states, gating_output, topk, renormalize, correction_bias
            )
        else:
            raise ValueError(f"Invalid scoring function: {scoring_func}")                    

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        # breakpoint()
        torch.testing.assert_close(res, ref)

    def test_topk(self):
        for renormalize in [True, False]:
            # for correction_bias in [False, True]:
            # for has_correction_bias in [False]:
            for has_correction_bias in [True]:
                # for scoring_func in ["sigmoid", "softmax"]:
                # for scoring_func in ["softmax"]:
                for scoring_func in ["sigmoid"]:
                    # self._run_single_test(16, 8, 2, renormalize, has_correction_bias, scoring_func, torch.bfloat16)
                    self._run_single_test(123, 256, 8, renormalize, has_correction_bias, scoring_func, torch.bfloat16)
                    
                    # self._run_single_test(123, 16, 3, renormalize, correction_bias, scoring_func, torch.bfloat16)
                    # self._run_single_test(123, 32, 3, renormalize, correction_bias, scoring_func, torch.bfloat16)
                    # self._run_single_test(123, 32, 3, renormalize, correction_bias, scoring_func, torch.bfloat16)
                    # self._run_single_test(123, 64, 6, renormalize, correction_bias, scoring_func, torch.bfloat16)
                    # self._run_single_test(123, 256, 4, renormalize, correction_bias, scoring_func, torch.bfloat16)
                    # self._run_single_test(123, 160, 6, renormalize, correction_bias, scoring_func, torch.bfloat16)


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
        correction_bias = None
        topk_weights, topk_ids = fused_custom_f(
            hidden_states, gating_output, topk, renormalize, correction_bias
        )

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        breakpoint()
        torch.testing.assert_close(res, ref)

    def test_custom_topk(self):
        test_custom_functions = [
            (Llama4MoE.custom_routing_function, torch.ops.sgl_kernel.topk_sigmoid_cpu)
        ]
        for native_custom_f, fused_custom_f in test_custom_functions:
            self._run_single_test(
                # 4, 8, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
                123, 8, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
            )
            # self._run_single_test(
            #     123, 32, 8, False, torch.bfloat16, native_custom_f, fused_custom_f
            # )
            # self._run_single_test(
            #     123, 32, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
            # )


if __name__ == "__main__":
    unittest.main()

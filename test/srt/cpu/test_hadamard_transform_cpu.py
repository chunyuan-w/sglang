import unittest

import torch

import sgl_kernel  # noqa: F401
from sglang.srt.layers.attention.nsa import nsa_indexer
from sglang.srt.layers.attention.nsa.nsa_indexer import _torch_hadamard_transform
from sglang.test.test_utils import CustomTestCase


class TestHadamardTransformCPU(CustomTestCase):
    def _assert_matches_torch_reference(
        self,
        shape,
        dtype: torch.dtype,
        scale: float,
        rtol: float,
        atol: float,
    ):
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=torch.float32).to(dtype).contiguous()
        actual = torch.ops.sgl_kernel.hadamard_transform_cpu(x, scale)
        expected = _torch_hadamard_transform(x, scale)

        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, dtype)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_matches_torch_reference_float32(self):
        self._assert_matches_torch_reference(
            shape=(7, 128),
            dtype=torch.float32,
            scale=128**-0.5,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_matches_torch_reference_bfloat16(self):
        self._assert_matches_torch_reference(
            shape=(2, 3, 64),
            dtype=torch.bfloat16,
            scale=64**-0.5,
            rtol=0,
            atol=0,
        )

    def test_matches_torch_reference_float16(self):
        self._assert_matches_torch_reference(
            shape=(5, 32),
            dtype=torch.float16,
            scale=0.25,
            rtol=0,
            atol=0,
        )

    def test_rotate_activation_cpu_dispatch(self):
        old_is_cpu_amx_available = nsa_indexer._is_cpu_amx_available
        try:
            nsa_indexer._is_cpu_amx_available = True
            x = torch.randn(4, 128, dtype=torch.bfloat16).contiguous()
            actual = nsa_indexer.rotate_activation(x)
            expected = _torch_hadamard_transform(x, 128**-0.5)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            nsa_indexer._is_cpu_amx_available = old_is_cpu_amx_available

    def test_rejects_non_power_of_two(self):
        x = torch.randn(4, 129, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "power of 2"):
            torch.ops.sgl_kernel.hadamard_transform_cpu(x, 1.0)


if __name__ == "__main__":
    unittest.main()

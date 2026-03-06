import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from utils import parametrize, precision

# from sglang.test.test_utils import CustomTestCase

flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func


torch.manual_seed(1234)

# result mismatch (this is the shape in model)
# N_token = 4655

# pass
# N_token = 3469

# nan
N_token = 4200

# result mismatch
# N_token = 4300
# N_token = 4400


class CustomTestCase(unittest.TestCase):
    # def _callTestMethod(self, method):
    #     max_retry = envs.SGLANG_TEST_MAX_RETRY.get()
    #     if max_retry is None:
    #         max_retry = 1 if is_in_ci() else 0
    #     retry(
    #         lambda: super(CustomTestCase, self)._callTestMethod(method),
    #         max_retry=max_retry,
    #     )

    def setUp(self):
        print(
            f"[CI Test Method] {self.__class__.__name__}.{self._testMethodName}",
            flush=True,
        )


def flash_attn_varlen_ref(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    is_causal,
    enable_gqa,
):
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    batch = len(cu_k) - 1
    
    B_T, H, D = q.shape
    T = B_T // batch
    
    # [T, H, D] -> [1, H, T, D]
    q, k, v = [x.reshape(batch, T, H, D).transpose(1, 2) for x in [q, k, v]]

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )
    # [B, H, T, D] -> [B * T, H, D]
    return out.transpose(1, 2).reshape(batch*T, H, D)


    # B, H, T, D = q.shape
    # out = torch.empty(B, H, T, v.size(-1), dtype=q.dtype)
    # for b in range(batch):
    #     start_q, end_q = cu_q[b], cu_q[b + 1]
    #     start_k, end_k = cu_k[b], cu_k[b + 1]

    #     out[:, :, start_q:end_q, :] = F.scaled_dot_product_attention(
    #         q[:, :, start_q:end_q, :],
    #         k[:, :, start_k:end_k, :],
    #         v[:, :, start_k:end_k, :],
    #         is_causal=is_causal,
    #         enable_gqa=enable_gqa,
    #     )

    # # [1, H, T, D] -> [T, H, D]
    # return out.transpose(1, 2).squeeze(0)


class TestFlashAttn(CustomTestCase):

    @parametrize(
        batch=[N_token],
        max_seqlen_q=[N_token],
        max_seqlen_k=[N_token],
        num_heads=[4],
        num_heads_kv=[4],
        head_dim=[32],  # test when D is not 32x
        head_dim_v=[32],
        is_causal=[False],
    )
    def test_flash_attn_varlen(
        self,
        batch,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_kv,
        head_dim,
        head_dim_v,
        is_causal,
    ):
        dtype = torch.bfloat16

        print("start tensor allocation")
        # random seqlens for k and kv
        # seqlens_q = torch.randint(1, max_seqlen_q, (batch,), dtype=torch.int32)
        seqlens_q = torch.full((batch,), max_seqlen_q, dtype=torch.int32)
        # seqlens_k = torch.randint(1, max_seqlen_k, (batch,), dtype=torch.int32)
        seqlens_k = torch.full((batch,), max_seqlen_k, dtype=torch.int32)
        cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_k = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
        cu_seqlens_k[1:] = torch.cumsum(seqlens_k, 0)

        sum_seqlen_q = seqlens_q.sum().item()
        sum_seqlen_k = seqlens_k.sum().item()
        q = torch.randn(sum_seqlen_q, num_heads, head_dim).to(dtype)
        k = torch.randn(sum_seqlen_k, num_heads_kv, head_dim).to(dtype)
        v = torch.randn(sum_seqlen_k, num_heads_kv, head_dim_v).to(dtype)
        print("done tensor allocation")
        out_ref = flash_attn_varlen_ref(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            is_causal=is_causal,
            enable_gqa=num_heads != num_heads_kv,
        )
        print("done ref computation")

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlens_q.max().item(),
            seqlens_k.max().item(),
            is_causal,
        )
        print("done sgl-kernel computation")
        

        atol = rtol = precision[dtype]
        if N_token == 4655:
            print("ref: ", out_ref[16777644][0][0])
            print("out: ", out[16777644][0][0])
        
            # assuming out_ref and out are your tensors
            mask = (out == 0) & (out_ref != 0)   # True where out is 0 but out_ref is not 0

            # get indices
            indices = torch.nonzero(mask, as_tuple=False)

            print(f"Number of positions where out is 0 but out_ref is not: {indices.shape[0]}")
            print("Some example indices and values:")

            for idx in indices[:10]:  # print first 10 for sanity
                i, j, k = idx.tolist()
                print(f"Index {i, j, k}: out_ref={out_ref[i,j,k]}, out={out[i,j,k]}")        
        

        if N_token == 4200:
            print("ref: ", out_ref[894600][0][0])
            print("out: ", out[894600][0][0])
            num_nans = torch.isnan(out).sum().item()
            print("Total number of NaNs in out:", num_nans)        
        
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()

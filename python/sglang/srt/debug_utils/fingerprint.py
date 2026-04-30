"""Tiny per-tensor fingerprint logger for cross-machine bisection.

Goal: emit one short line per checkpointed tensor (no full tensor dump) so two
runs (e.g. CPU vs H200) can be diffed by piping through ``diff`` or ``grep``.

Enable with::

    SGLANG_DBG_FINGERPRINT=1

By default only rank 0 prints (residual-stream tensors are replicated across
TP ranks, so this is sufficient). To enable on all ranks::

    SGLANG_DBG_FINGERPRINT_ALL_RANKS=1

Each line looks like::

    [FP r0] L23 attn_out       shape=(8, 7168) dt=torch.bfloat16 \
absmax=+1.234e+00 mean=+1.23e-04 std=+5.67e-02 norm=+1.23e+02 \
first8=[+1.234e-02, ..., -3.456e-02]

The ``first8`` slice is the most useful field: it is reduction-order
independent and a single-byte FP4-vs-FP8 weight diff would change it. Mean
/ std / norm are coarser and may differ in low-order digits between CPU and
CUDA due to reduction order; absmax is robust.
"""

from __future__ import annotations

import os

import torch


def _truthy(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on")


_ENABLED = _truthy(os.environ.get("SGLANG_DBG_FINGERPRINT", ""))
_ALL_RANKS = _truthy(os.environ.get("SGLANG_DBG_FINGERPRINT_ALL_RANKS", ""))


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def fp(tag: str, t) -> None:
    """One-line fingerprint of a tensor.

    Safe to call when disabled (returns immediately). Accepts None / non-tensor
    without raising so the call sites stay terse.
    """
    if not _ENABLED:
        return

    rank = _rank()
    if not _ALL_RANKS and rank != 0:
        return

    if t is None:
        print(f"[FP r{rank}] {tag} = None", flush=True)
        return

    if not isinstance(t, torch.Tensor):
        print(f"[FP r{rank}] {tag} = {type(t).__name__}({t!r})", flush=True)
        return

    if t.numel() == 0:
        print(
            f"[FP r{rank}] {tag} shape={tuple(t.shape)} dt={t.dtype} <empty>",
            flush=True,
        )
        return

    f = t.detach().reshape(-1).float()
    n = f.numel()
    sample_n = min(8, n)
    samp = ", ".join(f"{x:+.4e}" for x in f[:sample_n].cpu().tolist())
    absmax = f.abs().max().item()
    mean = f.mean().item()
    std = f.std().item() if n > 1 else 0.0
    norm = f.pow(2).sum().sqrt().item()

    print(
        f"[FP r{rank}] {tag:30s} shape={tuple(t.shape)} dt={t.dtype} "
        f"absmax={absmax:+.4e} mean={mean:+.4e} std={std:+.4e} "
        f"norm={norm:+.4e} first8=[{samp}]",
        flush=True,
    )

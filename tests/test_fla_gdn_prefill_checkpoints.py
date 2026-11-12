# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton/FLA GDN prefill checkpoints: the fp32 state every N tokens, laid out
as FlashInfer emits them, equal the final state of a prefill cut there."""

import pytest
import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.v1.attention.backends.gdn_attn import plan_flashinfer_checkpoints


def _inputs(T, Hg, H, K, V, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    q = F.normalize(randn(1, T, Hg, K), dim=-1).bfloat16()
    k = F.normalize(randn(1, T, Hg, K), dim=-1).bfloat16()
    v = randn(1, T, H, V).bfloat16()
    beta = torch.rand(1, T, H, generator=gen, device="cuda").sigmoid().bfloat16()
    g = F.logsigmoid(randn(1, T, H)).float()
    return q, k, v, g, beta


@pytest.mark.parametrize("every", [64, 256, 1024])
@pytest.mark.parametrize("with_h0", [False, True])
def test_checkpoints_match_cut_prefill(every, with_h0):
    # Qwen3.5 GDN head layout: 16 key heads, 32 value heads, 128 dims.
    Hg, H, K, V = 16, 32, 128, 128
    lens = [every * 3 + 17, every, 63, every * 2, 5 * 64 + 1]
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda")
    q, k, v, g, beta = _inputs(sum(lens), Hg, H, K, V)
    N = len(lens)
    h0 = (
        torch.randn(N, H, V, K, device="cuda") * 0.1
        if with_h0
        else torch.zeros(N, H, V, K, device="cuda")
    )

    offsets = [(n // every) * every for n in lens]
    every_n, cu_starts, _ = plan_flashinfer_checkpoints(offsets, lens)
    assert every_n == every
    ckpt = torch.full((cu_starts[-1], H, V, K), float("nan"), device="cuda")
    cu_starts_t = torch.tensor(cu_starts, device="cuda", dtype=torch.int64)

    o_ref, ht_ref = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=h0.clone(), output_final_state=True,
        cu_seqlens=cu,
    )
    o, ht = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=h0.clone(), output_final_state=True,
        cu_seqlens=cu, state_checkpoints=ckpt, checkpoint_cu_starts=cu_starts_t,
        checkpoint_every_n_tokens=every,
    )
    # Checkpointing changes nothing else.
    torch.testing.assert_close(o, o_ref, rtol=0, atol=0)
    torch.testing.assert_close(ht, ht_ref, rtol=0, atol=0)
    assert not ckpt.isnan().any(), "every planned row must be written"

    for i, n in enumerate(lens):
        bos = cu[i].item()
        for j in range(n // every):
            end = (j + 1) * every
            sl = slice(bos, bos + end)
            _, ht_cut = chunk_gated_delta_rule(
                q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
                initial_state=h0[i : i + 1].clone(), output_final_state=True,
                cu_seqlens=torch.tensor([0, end], device="cuda"),
            )
            # Same kernel, same chunk grid, same fp32 registers: exact.
            torch.testing.assert_close(
                ckpt[cu_starts[i] + j], ht_cut[0], rtol=0, atol=0
            )


def test_resume_from_checkpoint_matches_uncut():
    """A suffix prefilled from a checkpoint matches the uncut prefill."""
    Hg, H, K, V = 16, 32, 128, 128
    T, cut = 14_700, 14_336  # a state block hit on a 1,024-token grid
    q, k, v, g, beta = _inputs(T, Hg, H, K, V, seed=1)
    h0 = torch.zeros(1, H, V, K, device="cuda")
    ckpt = torch.empty(T // 1024, H, V, K, device="cuda")
    o_full, ht_full = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=h0.clone(), output_final_state=True,
        cu_seqlens=torch.tensor([0, T], device="cuda"), state_checkpoints=ckpt,
        checkpoint_cu_starts=torch.tensor([0, T // 1024], device="cuda"),
        checkpoint_every_n_tokens=1024,
    )
    sl = slice(cut, T)
    o_tail, ht_tail = chunk_gated_delta_rule(
        q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
        initial_state=ckpt[cut // 1024 - 1 : cut // 1024].clone(),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, T - cut], device="cuda"),
    )
    torch.testing.assert_close(o_tail, o_full[:, sl], rtol=0, atol=0)
    torch.testing.assert_close(ht_tail, ht_full, rtol=0, atol=0)

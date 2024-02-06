import math
import os

import numpy as np
import pytest
import torch

torch.manual_seed(0)
np.random.seed(0)

from .linear_attn_smallv_hd import LinearAttentionSmallVHD

NUM_TEST = 3

from typing import Tuple

BFLOAT16_MAX_TOL = 1e-2

# ----------------
# Helper Functions
# ----------------


def rel_error(x, y):
    return torch.linalg.norm(x - y) / torch.linalg.norm(y)


def make_input_qkv(
    bs: int, nh: int, sq: int, hd_qk: int, hd_v: int
) -> Tuple[torch.HalfTensor, torch.HalfTensor, torch.HalfTensor, torch.HalfTensor]:
    q = torch.randn(
        (bs, nh, sq, hd_qk), device="cuda:0", dtype=torch.bfloat16, requires_grad=False
    )
    k = torch.randn(
        (bs, nh, sq, hd_qk), device="cuda:0", dtype=torch.bfloat16, requires_grad=False
    )
    v = torch.randn(
        (bs, nh, sq, hd_v), device="cuda:0", dtype=torch.bfloat16, requires_grad=False
    )

    q = torch.nn.functional.softplus(q)
    k = torch.nn.functional.softplus(k)

    q *= 1.0 / math.sqrt(hd_qk)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    return q, k, v


def linear_ref_torch(
    q: torch.HalfTensor, k: torch.HalfTensor, v: torch.HalfTensor
) -> torch.HalfTensor:
    qk_scores = torch.matmul(q, k.transpose(-2, -1))

    L, S = q.size(-2), k.size(-2)

    temp_mask = torch.ones(1, 1, L, S, dtype=torch.bool, device=q.device).tril(
        diagonal=0
    )

    qk_scores *= temp_mask

    return torch.matmul(qk_scores, v)


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd_qk, hd_v)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(1, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd_qk in np.random.choice([32, 64, 128], size=NUM_TEST, replace=True)
        for hd_v in np.random.choice([16, 32], size=NUM_TEST, replace=True)
    ],
)
def test_attn_fwd(dims):
    # test standard causal attn (forward) (hd_qk != hd_v)
    bs, nh, sq, hd_qk, hd_v = dims

    q, k, v = make_input_qkv(bs, nh, sq, hd_qk, hd_v)

    with torch.no_grad():
        ref_out = linear_ref_torch(q, k, v)

        triton_out = LinearAttentionSmallVHD.apply(q, k, v)

    assert rel_error(triton_out, ref_out) < BFLOAT16_MAX_TOL


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd_qk, hd_v)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(1, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd_qk in np.random.choice([32, 64, 128], size=NUM_TEST, replace=True)
        for hd_v in np.random.choice([16, 32], size=NUM_TEST, replace=True)
    ],
)
def test_attn_bwd(dims):
    # test standard causal attn (backward) (hd_qk != hd_v)
    bs, nh, sq, hd_qk, hd_v = dims

    q, k, v = make_input_qkv(bs, nh, sq, hd_qk, hd_v)

    ref_out = linear_ref_torch(q, k, v)

    dy = 0.1 * torch.randn_like(ref_out)

    ref_out.backward(dy, retain_graph=True)
    dQ_torch, dK_torch, dV_torch = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None

    triton_out = LinearAttentionSmallVHD.apply(q, k, v)
    triton_out.backward(dy, retain_graph=True)
    dQ_triton, dK_triton, dV_triton = [_.grad.clone() for _ in [q, k, v]]

    assert rel_error(dQ_triton, dQ_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dK_triton, dK_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dV_triton, dV_torch) < BFLOAT16_MAX_TOL


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd_qk, hd_v)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(1, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd_qk in np.random.randint(32, 128, size=NUM_TEST)
        for hd_v in np.random.choice([16, 32], size=NUM_TEST, replace=True)
    ],
)
def test_attn_fwd_masked_hd(dims):
    # test block masking for head dim (forward) (hd_qk != hd_v)
    bs, nh, sq, hd_qk, hd_v = dims

    q, k, v = make_input_qkv(bs, nh, sq, hd_qk, hd_v)

    with torch.no_grad():
        ref_out = linear_ref_torch(q, k, v)

        triton_out = LinearAttentionSmallVHD.apply(q, k, v)

    assert rel_error(triton_out, ref_out) < BFLOAT16_MAX_TOL


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd_qk, hd_v)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(1, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd_qk in np.random.randint(32, 128, size=NUM_TEST)
        for hd_v in np.random.choice([16, 32], size=NUM_TEST, replace=True)
    ],
)
def test_attn_bwd_masked_hd(dims):
    # test block masking for head dim (backward) (hd_qk != hd_v)
    bs, nh, sq, hd_qk, hd_v = dims

    q, k, v = make_input_qkv(bs, nh, sq, hd_qk, hd_v)

    ref_out = linear_ref_torch(q, k, v)

    dy = 0.1 * torch.randn_like(ref_out)

    ref_out.backward(dy, retain_graph=True)
    dQ_torch, dK_torch, dV_torch = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None

    triton_out = LinearAttentionSmallVHD.apply(q, k, v)
    triton_out.backward(dy, retain_graph=True)
    dQ_triton, dK_triton, dV_triton = [_.grad.clone() for _ in [q, k, v]]

    assert rel_error(dQ_triton, dQ_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dK_triton, dK_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dV_triton, dV_torch) < BFLOAT16_MAX_TOL

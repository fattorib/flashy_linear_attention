import math
import os

import numpy as np
import pytest
import torch

torch.manual_seed(0)
np.random.seed(0)

from .linear_attn import LinearAttention

NUM_TEST = 3

from typing import Tuple

BFLOAT16_MAX_TOL = 1e-2

# ----------------
# Helper Functions
# ----------------


def rel_error(x, y):
    return torch.linalg.norm(x - y) / torch.linalg.norm(y)


def make_input_qkv(
    bs: int, nh: int, sq: int, hd: int
) -> Tuple[torch.HalfTensor, torch.HalfTensor, torch.HalfTensor, torch.HalfTensor]:
    shape = (bs, nh, sq, hd)

    q = torch.randn(shape, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda:0", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda:0", dtype=torch.bfloat16)

    q = torch.nn.functional.softplus(q)
    k = torch.nn.functional.softplus(k)

    q *= 1.0 / math.sqrt(hd)

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
        (b, nh, sq, hd)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(1, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd in np.random.choice([32, 64, 128], size=NUM_TEST, replace=True)
    ],
)
def test_attn_fwd(dims):
    # test standard causal attn (forward)
    bs, nh, sq, hd = dims

    q, k, v = make_input_qkv(bs, nh, sq, hd)

    with torch.no_grad():
        ref_out = linear_ref_torch(q, k, v)

        triton_out = LinearAttention.apply(q, k, v)

    assert rel_error(triton_out, ref_out) < BFLOAT16_MAX_TOL


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(1, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd in np.random.choice([32, 64, 128], size=NUM_TEST, replace=True)
    ],
)
def test_attn_bwd(dims):
    # test standard causal attn (backward)
    bs, nh, sq, hd = dims

    q, k, v = make_input_qkv(bs, nh, sq, hd)

    dy = 0.1 * torch.randn_like(q)

    ref_out = linear_ref_torch(q, k, v)

    ref_out.backward(dy, retain_graph=True)
    dQ_torch, dK_torch, dV_torch = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None

    triton_out = LinearAttention.apply(q, k, v)
    triton_out.backward(dy, retain_graph=True)
    dQ_triton, dK_triton, dV_triton = [_.grad.clone() for _ in [q, k, v]]

    assert rel_error(dQ_triton, dQ_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dK_triton, dK_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dV_triton, dV_torch) < BFLOAT16_MAX_TOL


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(1, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd in np.random.randint(16, 128, size=NUM_TEST)
    ],
)
def test_attn_fwd_masked_hd(dims):
    # test block masking for head dim (forward)
    bs, nh, sq, hd = dims

    q, k, v = make_input_qkv(bs, nh, sq, hd)

    with torch.no_grad():
        ref_out = linear_ref_torch(q, k, v)

        triton_out = LinearAttention.apply(q, k, v)

    assert rel_error(triton_out, ref_out) < BFLOAT16_MAX_TOL


@pytest.mark.parametrize(
    "dims",
    [
        (b, nh, sq, hd)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for nh in np.random.randint(1, 32, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for hd in np.random.randint(16, 128, size=NUM_TEST)
    ],
)
def test_attn_bwd_masked_hd(dims):
    # test block masking for head dim (backward)
    bs, nh, sq, hd = dims

    q, k, v = make_input_qkv(bs, nh, sq, hd)

    dy = 0.1 * torch.randn_like(q)

    ref_out = linear_ref_torch(q, k, v)

    ref_out.backward(dy, retain_graph=True)
    dQ_torch, dK_torch, dV_torch = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None

    triton_out = LinearAttention.apply(q, k, v)
    triton_out.backward(dy, retain_graph=True)
    dQ_triton, dK_triton, dV_triton = [_.grad.clone() for _ in [q, k, v]]

    assert rel_error(dQ_triton, dQ_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dK_triton, dK_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dV_triton, dV_torch) < BFLOAT16_MAX_TOL

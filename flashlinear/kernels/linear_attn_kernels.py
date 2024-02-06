"""Linear Attention kernels."""

from typing import Tuple

import torch
import triton
import triton.language as tl


# fmt: off
@triton.jit
def linear_flash_attn_fwd(
    q_ptr,k_ptr,v_ptr,o_ptr,
    qkv_stride_b,qkv_stride_h,
    qkv_stride_sq,qkv_stride_hd,
    BLOCK_HD: tl.constexpr,BLOCK_SQ: tl.constexpr,
    num_head,head_dim, context_sq,
):
#fmt: on

    q_chunk_pid = tl.program_id(axis=0)  # parallelize across sq chunks
    bh_pid = tl.program_id(axis=1)  # parallelize across batch x heads

    off_bs = (bh_pid // num_head,)
    off_h = (bh_pid % num_head,)

    bh_offset = off_bs.to(tl.int64) * qkv_stride_b + off_h.to(tl.int64) * qkv_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(q_chunk_pid * BLOCK_SQ, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    out = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    q = tl.load(q_block_ptr, boundary_check=(1,))

    max_range = q_chunk_pid * BLOCK_SQ + 1

    offs_k = tl.arange(0, BLOCK_SQ)
    offs_q = tl.arange(0, BLOCK_SQ)

    for chunk in range(0, max_range - 1, BLOCK_SQ):
        k = tl.load(
            k_block_ptr, boundary_check=(0,)
        )
        v = tl.load(
            v_block_ptr, boundary_check=(1,)
        )

        s_ij = tl.dot(q, k, allow_tf32=False)  # [BLOCK_SQ, BLOCK_SK]

        out += tl.dot(s_ij.to(tl.bfloat16), v, allow_tf32=False)

        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SQ))
        v_block_ptr = tl.advance(v_block_ptr, offsets=(BLOCK_SQ, 0))

    # final block - we reuse code here to remove conditionals from for loop
    k = tl.load(
        k_block_ptr, boundary_check=(0,)
    )
    v = tl.load(
        v_block_ptr, boundary_check=(1,)
    )
    s_ij = tl.dot(q, k, allow_tf32=False)  # [BLOCK_SQ, BLOCK_SK]
    offs = max_range - 1
    s_ij = tl.where(
        q_chunk_pid * BLOCK_SQ + offs_k[:, None] >= (offs + offs_q[None, :]),
        s_ij,
        0.0,
    )

    out += tl.dot(s_ij.to(tl.bfloat16), v, allow_tf32=False)

    out_block_ptr = tl.make_block_ptr(
        o_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(q_chunk_pid * BLOCK_SQ, 0),
    )

    tl.store(
        out_block_ptr,
        value=out.to(tl.bfloat16),
        boundary_check=(1,)
    )

# fmt: off
@triton.jit
def flash_attn_bwd(
    q_ptr,k_ptr,v_ptr,
    dO_ptr,dV_ptr,dK_ptr,dQ_ptr,
    qkv_stride_b,qkv_stride_h,
    qkv_stride_sq,qkv_stride_hd,
    BLOCK_HD: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    context_sq,head_dim, num_head,
):
# fmt: on

    kv_chunk_pid = tl.program_id(axis=0)  # parallelize across kv chunks
    bh_pid = tl.program_id(axis=1)  # parallelize across batch x heads

    off_bs = (bh_pid // num_head,)
    off_h = (bh_pid % num_head,)

    bh_offset = off_bs.to(tl.int64) * qkv_stride_b + off_h.to(tl.int64) * qkv_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        dO_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )

    dV = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)
    dK = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    k_trans = tl.load(k_block_ptr, boundary_check=(0,))
    v_trans = tl.load(v_block_ptr, boundary_check=(0,))

    max_range = context_sq
    min_range = kv_chunk_pid * BLOCK_SQ

    offs_k = tl.arange(0, BLOCK_SQ)
    offs_q = (kv_chunk_pid * BLOCK_SQ) + tl.arange(0, BLOCK_SQ)

    # loop is split into pre/post masking to remove conditional use
    for q_chunk in range(0, min_range + 1, BLOCK_SQ):
        q = tl.load(
            q_block_ptr, boundary_check=(1,)
        )
        dout = tl.load(
            dout_block_ptr, boundary_check=(1,)
        )

        S_ij = tl.dot(q, k_trans, allow_tf32=False)

        S_ij = tl.where(
            (q_chunk + offs_k[:, None]) >= (offs_q[None, :]),
            S_ij,
            0.0,
        )

        dV += tl.dot(tl.trans(S_ij.to(tl.bfloat16)), dout, allow_tf32=False)

        dS_ij = tl.dot(dout, v_trans, allow_tf32=False)

        dS_ij = tl.where(
            (q_chunk + offs_k[:, None]) >= (offs_q[None, :]),
            dS_ij,
            0.0,
        )

        dK += tl.dot(tl.trans(dS_ij.to(tl.bfloat16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SQ, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SQ, 0))

    min_range_offset = min_range + BLOCK_SQ

    for q_chunk in range(min_range_offset, max_range, BLOCK_SQ):
        q = tl.load(
            q_block_ptr, boundary_check=(1,)
        )
        dout = tl.load(
            dout_block_ptr, boundary_check=(1,)
        )

        S_ij = tl.dot(q, k_trans, allow_tf32=False)

        dV += tl.dot(tl.trans(S_ij.to(tl.bfloat16)), dout, allow_tf32=False)

        dS_ij = tl.dot(dout, v_trans, allow_tf32=False)

        dK += tl.dot(tl.trans(dS_ij.to(tl.bfloat16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SQ, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SQ, 0))

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    tl.store(
        dV_block_ptr,
        value=dV.to(tl.bfloat16), boundary_check=(1,)
    )
    tl.store(
        dK_block_ptr,
        value=dK.to(tl.bfloat16), boundary_check=(1,)
    )

    # ----------
    # compute dQ
    # ----------

    # reset block pointers
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        dO_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    q = tl.load(
        q_block_ptr, boundary_check=(1,)
    )

    dQ = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_SQ)

    max_range = kv_chunk_pid * BLOCK_SQ + 1
    final = max_range - 1

    dout = tl.load(
        dout_block_ptr, boundary_check=(1,)
    )

    for q_chunk in range(0, final, BLOCK_SQ):
        v_trans = tl.load(
            v_block_ptr, boundary_check=(0,)
        )
        k_trans = tl.load(
            k_block_ptr, boundary_check=(0,)
        )

        S_ij = tl.dot(q, k_trans, allow_tf32=False)

        dS_ij = tl.dot(dout, v_trans, allow_tf32=False)

        dQ += tl.dot(dS_ij.to(tl.bfloat16), tl.trans(k_trans), allow_tf32=False)

        v_block_ptr = tl.advance(v_block_ptr, offsets=(0, BLOCK_SQ))
        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SQ))

    v_trans = tl.load(v_block_ptr, boundary_check=(0,))
    k_trans = tl.load(k_block_ptr, boundary_check=(0,))

    S_ij = tl.dot(q, k_trans, allow_tf32=False)

    # causal masking on final block
    S_ij = tl.where(
        kv_chunk_pid * BLOCK_SQ + offs_k[:, None] >= (final + offs_k[None, :]),
        S_ij,
        0.0,
    )

    dS_ij = tl.dot(dout, v_trans, allow_tf32=False)

    dS_ij = tl.where(
        kv_chunk_pid * BLOCK_SQ + offs_k[:, None] >= (final + offs_k[None, :]),
        dS_ij,
        0.0,
    )

    dQ += tl.dot(dS_ij.to(tl.bfloat16), tl.trans(k_trans), allow_tf32=False)

    tl.store(
        dQ_block_ptr,
        dQ.to(tl.bfloat16), boundary_check=(1,)
    )


def linear_flash_wrapper_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Function wrapping causal Flash Attention kernel."""
    batch, nh, sq, hd = q.shape

    BLOCK_HD = triton.next_power_of_2(hd)
    BLOCK_SQ = 64 if BLOCK_HD <= 128 else 32
    num_warps = 4 if BLOCK_HD < 128 else 8 
    
    assert hd <= 256, "Only head_dims <= 256 are supported."
    assert (
        sq % BLOCK_SQ == 0
    ), f"Number of elements in sequence must be a multiple of {BLOCK_SQ}, {sq,BLOCK_SQ}"

    out = torch.empty_like(q)

    def grid(META):
        return (triton.cdiv(sq, META["BLOCK_SQ"]), batch * nh)

    # fmt: off
    linear_flash_attn_fwd[grid](
        q,k,v,out,
        q.stride(0),q.stride(1),q.stride(2),q.stride(3),
        BLOCK_HD=BLOCK_HD,BLOCK_SQ=BLOCK_SQ,
        num_warps=num_warps,num_stages=2,
        context_sq=sq,num_head=nh, head_dim=hd
    )
    # fmt: on
    return out



def linear_flash_wrapper_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function calling Flash Attention backward pass."""
    batch, nh, sq, hd = q.shape

    BLOCK_HD = triton.next_power_of_2(hd)
    BLOCK_SQ = 64 if BLOCK_HD <= 128 else 32

    num_warps = 4 # TODO: Find good values

    assert hd <= 256, "Only head_dims <= 256 are supported."
    assert (
        sq % BLOCK_SQ == 0
    ), f"Number of elements in sequence must be a multiple of {BLOCK_SQ}"

    dQ = torch.zeros_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)

    def grid(META):
        return (triton.cdiv(sq, META["BLOCK_SQ"]), batch * nh)

    # fmt: off
    flash_attn_bwd[grid](
        q,k,v,
        grad_output,
        dV,dK,dQ,
        q.stride(0),q.stride(1),q.stride(2),q.stride(3),
        BLOCK_HD=BLOCK_HD,BLOCK_SQ=BLOCK_SQ,
        num_warps=num_warps,num_stages=1,
        context_sq=sq,num_head=nh,head_dim=hd
    )
    #fmt: on
    return dQ, dK, dV

"""Linear Attention kernels where Queries and Keys 
have a different dimensionality than Values."""

from typing import Tuple

import torch
import triton
import triton.language as tl


# fmt: off
@triton.jit
def linear_flash_attn_fwd(
    q_ptr,k_ptr,v_ptr,o_ptr,
    qk_stride_b,qk_stride_h,qk_stride_sq,qk_stride_hd,
    v_stride_b,v_stride_h,v_stride_sq,v_stride_hd,
    BLOCK_HD: tl.constexpr,BLOCK_SQ: tl.constexpr,
    BLOCK_HD_V: tl.constexpr, BLOCK_SK: tl.constexpr,
    num_head,head_dim, context_sq,
):
#fmt: on

    q_chunk_pid = tl.program_id(axis=0)  # parallelize across sq chunks
    bh_pid = tl.program_id(axis=1)  # parallelize across batch x heads

    off_bs = (bh_pid // num_head,)
    off_h = (bh_pid % num_head,)

    bh_offset = off_bs.to(tl.int64) * qk_stride_b + off_h.to(tl.int64) * qk_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qk_stride_sq, qk_stride_hd),
        order=(1, 0),
        offsets=(q_chunk_pid * BLOCK_SQ, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SK),
        strides=(qk_stride_hd, qk_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    bh_offset = off_bs.to(tl.int64) * v_stride_b + off_h.to(tl.int64) * v_stride_h

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD_V),
        block_shape=(BLOCK_SK, BLOCK_HD_V),
        strides=(v_stride_sq, v_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    out = tl.zeros([BLOCK_SQ, BLOCK_HD_V], dtype=tl.float32)

    q = tl.load(q_block_ptr, boundary_check=(1,))

    offs_q = (q_chunk_pid * BLOCK_SQ) + tl.arange(0, BLOCK_SQ)
    offs_k = tl.arange(0, BLOCK_SK)

    for start_k in range(0, (q_chunk_pid + 1) * BLOCK_SQ, BLOCK_SK):

        start_k = tl.multiple_of(start_k, BLOCK_SK)

        k = tl.load(
            k_block_ptr, boundary_check=(0,)
        )
        v = tl.load(
            v_block_ptr, boundary_check=(1,)
        )

        s_ij = tl.dot(q, k, allow_tf32=False)  # [BLOCK_SQ, BLOCK_SK]

        s_ij = tl.where(
            offs_q[:, None] >= (start_k + offs_k[None, :]),
            s_ij,
            0.0,
        )

        out += tl.dot(s_ij.to(tl.bfloat16), v, allow_tf32=False)

        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SK))
        v_block_ptr = tl.advance(v_block_ptr, offsets=(BLOCK_SK, 0))

    

    bh_offset = off_bs.to(tl.int64) * v_stride_b + off_h.to(tl.int64) * v_stride_h

    out_block_ptr = tl.make_block_ptr(
        o_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD_V),
        block_shape=(BLOCK_SQ, BLOCK_HD_V),
        strides=(v_stride_sq, v_stride_hd),
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
    qk_stride_b,qk_stride_h,qk_stride_sq,qk_stride_hd,
    v_stride_b,v_stride_h,v_stride_sq,v_stride_hd,
    BLOCK_HD: tl.constexpr,BLOCK_HD_V: tl.constexpr,
    BLOCK_SQ: tl.constexpr,BLOCK_SK: tl.constexpr,
    context_sq,head_dim, num_head,
):
# fmt: on

    kv_chunk_pid = tl.program_id(axis=0)  # parallelize across kv chunks
    bh_pid = tl.program_id(axis=1)  # parallelize across batch x heads

    off_bs = (bh_pid // num_head,)
    off_h = (bh_pid % num_head,)

    bh_offset = off_bs.to(tl.int64) * qk_stride_b + off_h.to(tl.int64) * qk_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SK, BLOCK_HD),
        strides=(qk_stride_sq, qk_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qk_stride_hd, qk_stride_sq),
        order=(0, 1),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )
    
    bh_offset = off_bs.to(tl.int64) * v_stride_b + off_h.to(tl.int64) * v_stride_h

    dout_block_ptr = tl.make_block_ptr(
        dO_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SK, BLOCK_HD_V),
        strides=(v_stride_sq, v_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD_V, BLOCK_SQ),
        strides=(v_stride_hd, v_stride_sq),
        order=(0, 1),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )

    dV = tl.zeros([BLOCK_SQ, BLOCK_HD_V], dtype=tl.float32)
    dK = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    k_trans = tl.load(k_block_ptr, boundary_check=(0,))
    v_trans = tl.load(v_block_ptr, boundary_check=(0,))

    max_range = context_sq

    offs_k = tl.arange(0, BLOCK_SK)
    offs_q = (kv_chunk_pid * BLOCK_SQ) + tl.arange(0, BLOCK_SQ)

    for start_k in range(0, max_range, BLOCK_SK):
        q = tl.load(
            q_block_ptr, boundary_check=(1,)
        )
        dout = tl.load(
            dout_block_ptr, boundary_check=(1,)
        )

        S_ij = tl.dot(q, k_trans, allow_tf32=False)

        S_ij = tl.where(
            (start_k + offs_k[:, None]) >= (offs_q[None, :]),
            S_ij,
            0.0,
        )

        dV += tl.dot(tl.trans(S_ij.to(tl.bfloat16)), dout, allow_tf32=False)

        dS_ij = tl.dot(dout, v_trans, allow_tf32=False)

        dS_ij = tl.where(
            (start_k + offs_k[:, None]) >= (offs_q[None, :]),
            dS_ij,
            0.0,
        )

        dK += tl.dot(tl.trans(dS_ij.to(tl.bfloat16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SK, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SK, 0))

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD_V),
        block_shape=(BLOCK_SQ, BLOCK_HD_V),
        strides=(v_stride_sq, v_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    bh_offset = off_bs.to(tl.int64) * qk_stride_b + off_h.to(tl.int64) * qk_stride_h

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + bh_offset,
        shape=(context_sq, head_dim),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qk_stride_sq, qk_stride_hd),
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
        strides=(qk_stride_sq, qk_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    k_block_ptr_epi = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(head_dim, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SK),
        strides=(qk_stride_hd, qk_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    bh_offset = off_bs.to(tl.int64) * v_stride_b + off_h.to(tl.int64) * v_stride_h

    dout_block_ptr = tl.make_block_ptr(
        dO_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD_V),
        block_shape=(BLOCK_SQ, BLOCK_HD_V),
        strides=(v_stride_sq, v_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    v_block_ptr_epi = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(BLOCK_HD_V, context_sq),
        block_shape=(BLOCK_HD_V, BLOCK_SK),
        strides=(v_stride_hd, v_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    dQ = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    offs_q = kv_chunk_pid * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    offs_k = tl.arange(0, BLOCK_SK)

    dout = tl.load(
        dout_block_ptr, boundary_check=(1,)
    )

    for start_k in range(0, (kv_chunk_pid+1) * BLOCK_SQ, BLOCK_SK):
        v_trans_epi = tl.load(
            v_block_ptr_epi, boundary_check=(0,)
        )
        k_trans_epi = tl.load(
            k_block_ptr_epi, boundary_check=(0,)
        )

        dS_ij = tl.dot(dout, v_trans_epi, allow_tf32=False)

        dS_ij = tl.where(
            offs_q[:, None] >= (start_k + offs_k[None, :]),
            dS_ij,
            0.0,
        )

        dQ += tl.dot(dS_ij.to(tl.bfloat16), tl.trans(k_trans_epi), allow_tf32=False)

        v_block_ptr_epi = tl.advance(v_block_ptr_epi, offsets=(0, BLOCK_SK))
        k_block_ptr_epi = tl.advance(k_block_ptr_epi, offsets=(0, BLOCK_SK))

    tl.store(
        dQ_block_ptr,
        dQ.to(tl.bfloat16), boundary_check=(1,)
    )


def linear_flash_wrapper_fwd_smallv_hd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Function wrapping causal linear FA kernel where query/key dimension
    differs from value dimension."""

    batch, nh, sq, hd_qk = q.shape
    batch, nh, sq, hd_v = v.shape

    BLOCK_HD = triton.next_power_of_2(hd_qk)
    BLOCK_HD_V = triton.next_power_of_2(hd_v)
    BLOCK_SQ = 64 if BLOCK_HD <= 128 else 32
    BLOCK_SK = 64

    num_warps = 4 if BLOCK_HD < 128 else 8 
    
    assert triton.next_power_of_2(hd_v) == hd_v, "Expected value head dimension to be a power of 2."
    assert BLOCK_HD_V <= BLOCK_HD, "Value head dimension must be less than Query/Key head dimension."
    
    assert hd_qk <= 256, "Only head_dims <= 256 are supported."
    assert (
        sq % BLOCK_SQ == 0
    ), f"Number of elements in sequence must be a multiple of {BLOCK_SQ}, {sq,BLOCK_SQ}"

    out = torch.empty_like(v)

    def grid(META):
        return (triton.cdiv(sq, META["BLOCK_SQ"]), batch * nh)

    # fmt: off
    linear_flash_attn_fwd[grid](
        q,k,v,out,
        q.stride(0),q.stride(1),q.stride(2),q.stride(3),
        v.stride(0),v.stride(1),v.stride(2),v.stride(3),
        BLOCK_HD=BLOCK_HD,BLOCK_SQ=BLOCK_SQ,BLOCK_HD_V=BLOCK_HD_V,BLOCK_SK=BLOCK_SK,
        num_warps=num_warps,num_stages=2,
        context_sq=sq,num_head=nh, head_dim=hd_qk
    )
    # fmt: on
    return out



def linear_flash_wrapper_bwd_smallv_hd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function calling linear FA backward pass where query/key dimension
    differs from value dimension."""
    batch, nh, sq, hd_qk = q.shape
    batch, nh, sq, hd_v = v.shape

    BLOCK_HD = triton.next_power_of_2(hd_qk)
    BLOCK_HD_V = triton.next_power_of_2(hd_v)
    BLOCK_SQ = 64
    BLOCK_SK = 32

    num_warps = 4

    assert hd_qk <= 256, "Only head_dims <= 256 are supported."
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
        v.stride(0),v.stride(1),v.stride(2),v.stride(3),
        BLOCK_HD=BLOCK_HD,BLOCK_SQ=BLOCK_SQ,BLOCK_HD_V=BLOCK_HD_V,BLOCK_SK=BLOCK_SK,
        num_warps=num_warps,num_stages=1,
        context_sq=sq,num_head=nh,head_dim=hd_qk
    )
    #fmt: on
    return dQ, dK, dV

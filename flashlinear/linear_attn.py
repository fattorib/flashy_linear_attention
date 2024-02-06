"""Flashy Linear Attention in Triton."""

from typing import Any, Tuple

import torch
from torch.autograd import Function

from .kernels import linear_flash_wrapper_bwd, linear_flash_wrapper_fwd


class LinearAttention(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(
        ctx: Any, q: torch.HalfTensor, k: torch.HalfTensor, v: torch.HalfTensor
    ) -> torch.HalfTensor:
        out = linear_flash_wrapper_fwd(q, k, v)
        ctx.save_for_backward(q, k, v)

        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
        ctx: Any, grad_output: torch.HalfTensor
    ) -> Tuple[torch.HalfTensor, torch.HalfTensor, torch.HalfTensor]:
        query, key, value = ctx.saved_tensors

        gradq, gradk, gradv = linear_flash_wrapper_bwd(grad_output, query, key, value)

        return gradq, gradk, gradv

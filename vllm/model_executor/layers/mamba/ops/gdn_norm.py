# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated RMS normalization with token-strided gates and optional output scatter."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _gdn_norm_kernel(
    X,
    Z,
    W,
    B,
    Y,
    INDICES,
    num_tokens,
    stride_x_token: tl.constexpr,
    stride_x_head: tl.constexpr,
    stride_z_token: tl.constexpr,
    stride_z_head: tl.constexpr,
    stride_y_token: tl.constexpr,
    stride_y_head: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    token = (rows // NUM_HEADS).to(tl.int64)
    head = rows % NUM_HEADS
    valid = token < num_tokens
    dest = token
    if INDICES is not None:
        dest = tl.load(INDICES + token, valid, other=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.program_id(1) * GROUP_SIZE + cols
    mask = valid[:, None] & (cols[None, :] < GROUP_SIZE)
    x = tl.load(
        X
        + token[:, None] * stride_x_token
        + head[:, None] * stride_x_head
        + dims[None, :],
        mask,
        other=0.0,
    ).to(tl.float32)
    z = tl.load(
        Z
        + dest[:, None] * stride_z_token
        + head[:, None] * stride_z_head
        + dims[None, :],
        mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.sigmoid(z)
    if ACTIVATION == "silu" or ACTIVATION == "swish":
        gate = z * gate
    if not NORM_BEFORE_GATE:
        x *= gate
    variance = tl.sum(x * x, axis=1) / GROUP_SIZE
    y = x * tl.rsqrt(variance + EPS)[:, None]
    w = tl.load(W + dims, cols < GROUP_SIZE, other=0.0).to(tl.float32)
    y *= w[None, :]
    if B is not None:
        bias = tl.load(B + dims, cols < GROUP_SIZE, other=0.0).to(tl.float32)
        y += bias[None, :]
    if NORM_BEFORE_GATE:
        y *= gate
    tl.store(
        Y
        + dest[:, None] * stride_y_token
        + head[:, None] * stride_y_head
        + dims[None, :],
        y,
        mask,
    )


def gdn_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    eps: float,
    *,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    activation: str = "silu",
    token_indices: torch.Tensor | None = None,
) -> None:
    """Normalize packed attention output directly into its final token positions.

    Args:
        x: Attention output with shape [tokens, heads, head_dim].
        gate: Gate in final token order; token and head strides may be noncompact.
        weight: Per-head normalization weight.
        bias: Optional per-head normalization bias.
        out: Output in final token order. Unselected tokens remain untouched.
        eps: RMS normalization epsilon.
        group_size: Normalization group size within each head.
        norm_before_gate: Whether to apply the gate after normalization.
        activation: Gate activation: silu, swish, or sigmoid.
        token_indices: Unique destination token indices for each input token.
            Indexed output must not alias x. Without indices, x and out may
            be the same view.

    """
    assert x.ndim == gate.ndim == out.ndim == 3
    assert gate.shape == out.shape and x.shape[1:] == out.shape[1:]
    assert x.stride(-1) == gate.stride(-1) == out.stride(-1) == 1
    num_tokens, num_heads, head_dim = x.shape
    group_size = head_dim if group_size is None else group_size
    assert group_size > 0 and head_dim % group_size == 0
    assert weight.shape == (head_dim,) and weight.is_contiguous()
    assert bias is None or (bias.shape == weight.shape and bias.is_contiguous())
    assert activation in ("silu", "swish", "sigmoid")
    if token_indices is None:
        assert num_tokens == out.shape[0]
    else:
        assert token_indices.shape == (num_tokens,) and token_indices.is_contiguous()
        assert token_indices.dtype in (torch.int32, torch.int64)
    if num_tokens == 0:
        return
    _gdn_norm_kernel[(triton.cdiv(num_tokens * num_heads, 4), head_dim // group_size)](
        x,
        gate,
        weight,
        bias,
        out,
        token_indices,
        num_tokens,
        x.stride(0),
        x.stride(1),
        gate.stride(0),
        gate.stride(1),
        out.stride(0),
        out.stride(1),
        num_heads,
        group_size,
        eps,
        norm_before_gate,
        activation,
        triton.next_power_of_2(group_size),
        4,
        num_warps=4,
    )

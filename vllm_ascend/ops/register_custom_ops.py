import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.rotary_embedding import rope_forward_oot
from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.utils import enable_sp_by_pass, is_vl_model, npu_stream_switch, prefetch_stream


def _dump_flashcomm_debug(stage: str, **values) -> None:
    debug_file = os.getenv("VLLM_ASCEND_FLASHCOMM_DEBUG_FILE")
    if not debug_file:
        return
    if getattr(_EXTRA_CTX, "in_profile_run", False) or getattr(_EXTRA_CTX, "capturing", False):
        return

    rank = -1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    record = {"stage": stage, "rank": rank}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            detached = value.detach()
            flat = detached.reshape(-1)
            record[key] = {
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "sample": flat[:8].cpu().tolist(),
            }
        else:
            record[key] = value

    with open(f"{debug_file}.rank{rank}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _resolve_num_actual_tokens(attn_metadata) -> int | None:
    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
    if num_actual_tokens is not None:
        return int(num_actual_tokens)

    if isinstance(attn_metadata, dict) and attn_metadata:
        first_item = next(iter(attn_metadata.values()))
        value = getattr(first_item, "num_actual_tokens", None)
        return int(value) if value is not None else None

    if isinstance(attn_metadata, list) and attn_metadata:
        first_item = attn_metadata[0]
        if isinstance(first_item, dict) and first_item:
            value = getattr(next(iter(first_item.values())), "num_actual_tokens", None)
            return int(value) if value is not None else None

    return None


def _resolve_flashcomm_token_layout(total_tokens: int | None = None) -> tuple[int, int | None]:
    pad_size = _EXTRA_CTX.pad_size
    num_actual_tokens = None
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return pad_size, num_actual_tokens

    num_actual_tokens = _resolve_num_actual_tokens(getattr(forward_context, "attn_metadata", None))
    if num_actual_tokens is None:
        return pad_size, num_actual_tokens

    padded_num_tokens = getattr(forward_context, "num_tokens", None)
    if padded_num_tokens is None and total_tokens is not None:
        padded_num_tokens = total_tokens

    if padded_num_tokens is not None and padded_num_tokens >= num_actual_tokens:
        inferred_pad_size = int(padded_num_tokens - num_actual_tokens)
        if pad_size == 0 and inferred_pad_size > 0:
            pad_size = inferred_pad_size

    return pad_size, num_actual_tokens


def _safe_ep_world_size(is_ep_comm: bool) -> int | None:
    if not is_ep_comm:
        return None
    try:
        return get_ep_group().world_size
    except AssertionError:
        return None


def _ceil_div(num, denom: int):
    return (num + denom - 1) // denom


def _resolve_fake_pad_size() -> int:
    return max(int(getattr(_EXTRA_CTX, "pad_size", 0)), 0)


def _maybe_chunk_residual_impl(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pad_size, num_actual_tokens = _resolve_flashcomm_token_layout(residual.size(0))
        should_pad = pad_size > 0
        if num_actual_tokens is not None:
            should_pad = residual.size(0) == num_actual_tokens
        else:
            should_pad = residual.size(0) % tp_size != 0
        if should_pad:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


<<<<<<< Updated upstream
def _maybe_chunk_residual_impl(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)
    if flash_comm_v1_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            x = tensor_model_parallel_all_gather(x, 0)
            pad_size = _EXTRA_CTX.pad_size
            if pad_size > 0:
                x = x[:-pad_size]
        else:
            x = get_ep_group().all_gather(x, 0)
            if enable_sp_by_pass():  # TODO: do unpad
                return x
            # unpad
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
                offset += num_tokens_dp
            x = result

    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    flash_comm_v1_enabled = getattr(forward_context, "flash_comm_v1_enabled", False) or (
        enable_sp_by_pass() and is_ep_comm
    )

    if not flash_comm_v1_enabled or (forward_context.is_draft_model and is_vl_model() and not is_ep_comm):
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        if enable_sp_by_pass():
            return get_ep_group().reduce_scatter(x.view(-1, *x.shape[1:]), 0)
        # padding
        dp_size = get_dp_group().world_size
        num_tokens_across_dp_cpu = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        padded_x = torch.empty((dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]), device=x.device, dtype=x.dtype)
        offset = 0
=======
def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)
    _dump_flashcomm_debug(
        "maybe_all_gather_input",
        x=x,
        label=label,
        is_ep_comm=is_ep_comm,
        flash_comm_v1_enabled=flash_comm_v1_enabled,
        extra_flash_comm_v1_enabled=_EXTRA_CTX.flash_comm_v1_enabled,
        forward_flash_comm_v1_enabled=getattr(forward_context, "flash_comm_v1_enabled", None),
        enable_sp_by_pass=enable_sp_by_pass(),
        dp_metadata_is_none=forward_context.dp_metadata is None,
        num_tokens=getattr(forward_context, "num_tokens", None),
        pad_size=_EXTRA_CTX.pad_size,
        tp_world_size=get_tensor_model_parallel_world_size(),
        ep_world_size=_safe_ep_world_size(is_ep_comm),
    )
    if flash_comm_v1_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            branch = "tp_all_gather"
            x = tensor_model_parallel_all_gather(x, 0)
            if is_ep_comm:
                _, num_actual_tokens = _resolve_flashcomm_token_layout(x.size(0))
                if num_actual_tokens is not None and x.shape[0] > num_actual_tokens:
                    x = x[:num_actual_tokens]
            else:
                pad_size = _EXTRA_CTX.pad_size
                num_actual_tokens = None
                if pad_size > 0:
                    x = x[:-pad_size]
        else:
            branch = "ep_all_gather"
            x = get_ep_group().all_gather(x, 0)
            if enable_sp_by_pass():  # TODO: do unpad
                _dump_flashcomm_debug(
                    "maybe_all_gather_output",
                    x=x,
                    branch=branch,
                    enable_sp_by_pass=True,
                )
                return x
            # unpad
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
                offset += num_tokens_dp
            x = result
            num_actual_tokens = None

        _dump_flashcomm_debug(
            "maybe_all_gather_output",
            x=x,
            branch=branch,
            num_actual_tokens=num_actual_tokens,
        )

    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    flash_comm_v1_enabled = getattr(forward_context, "flash_comm_v1_enabled", False) or (
        enable_sp_by_pass() and is_ep_comm
    )
    _dump_flashcomm_debug(
        "maybe_pad_reduce_input",
        x=x,
        is_ep_comm=is_ep_comm,
        flash_comm_v1_enabled=flash_comm_v1_enabled,
        extra_flash_comm_v1_enabled=_EXTRA_CTX.flash_comm_v1_enabled,
        forward_flash_comm_v1_enabled=getattr(forward_context, "flash_comm_v1_enabled", None),
        enable_sp_by_pass=enable_sp_by_pass(),
        dp_metadata_is_none=forward_context.dp_metadata is None,
        num_tokens=getattr(forward_context, "num_tokens", None),
        pad_size=_EXTRA_CTX.pad_size,
        tp_world_size=get_tensor_model_parallel_world_size(),
        ep_world_size=_safe_ep_world_size(is_ep_comm),
    )

    if not flash_comm_v1_enabled or (forward_context.is_draft_model and is_vl_model()):
        out = tensor_model_parallel_all_reduce(x)
        _dump_flashcomm_debug("maybe_pad_reduce_output", x=out, branch="tp_all_reduce")
        return out

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        if is_ep_comm:
            _, num_actual_tokens = _resolve_flashcomm_token_layout(x.size(0))
            if num_actual_tokens is not None and x.shape[0] > num_actual_tokens:
                x = x[:num_actual_tokens]
            if num_actual_tokens is not None:
                expected_padded_tokens = (
                    (num_actual_tokens + get_tensor_model_parallel_world_size() - 1)
                    // get_tensor_model_parallel_world_size()
                    * get_tensor_model_parallel_world_size()
                )
                if x.shape[0] < expected_padded_tokens:
                    x = F.pad(x, (0, 0, 0, expected_padded_tokens - x.shape[0]))
            else:
                expected_padded_tokens = None
        else:
            pad_size = _EXTRA_CTX.pad_size
            num_actual_tokens = None
            expected_padded_tokens = None
            if pad_size > 0:
                x = F.pad(x, (0, 0, 0, pad_size))
        out = tensor_model_parallel_reduce_scatter(x, 0)
        _dump_flashcomm_debug(
            "maybe_pad_reduce_output",
            x=out,
            branch="tp_reduce_scatter",
            num_actual_tokens=num_actual_tokens,
            expected_padded_tokens=expected_padded_tokens,
        )
        return out
    else:
        if enable_sp_by_pass():
            out = get_ep_group().reduce_scatter(x.view(-1, *x.shape[1:]), 0)
            _dump_flashcomm_debug("maybe_pad_reduce_output", x=out, branch="ep_reduce_scatter_bypass")
            return out
        # padding
        dp_size = get_dp_group().world_size
        num_tokens_across_dp_cpu = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        padded_x = torch.empty((dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]), device=x.device, dtype=x.dtype)
        offset = 0
>>>>>>> Stashed changes
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            padded_x[idx, :num_tokens_dp] = x[offset : offset + num_tokens_dp]
            offset += num_tokens_dp

        out = get_ep_group().reduce_scatter(padded_x.view(-1, *x.shape[1:]), 0)
        _dump_flashcomm_debug("maybe_pad_reduce_output", x=out, branch="ep_reduce_scatter")
        return out


def _maybe_all_gather_and_maybe_unpad_fake(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)
    if flash_comm_v1_enabled and label:
        world_size = _safe_ep_world_size(is_ep_comm) or get_tensor_model_parallel_world_size()
        num_tokens = x.shape[0] * world_size - _resolve_fake_pad_size()
        return torch.empty(
            (num_tokens, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )

    return x


def _maybe_pad_and_reduce_fake(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)
    if flash_comm_v1_enabled:
        world_size = _safe_ep_world_size(is_ep_comm) or get_tensor_model_parallel_world_size()
        num_tokens = _ceil_div(x.shape[0] + _resolve_fake_pad_size(), world_size)
        return torch.empty(
            (num_tokens, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )

    return x


def _prefetch_preprocess_impl(weight: torch.Tensor, start_flag: torch.Tensor, max_weight_size: int) -> None:
    calculation_stream = torch_npu.npu.current_stream()
    weight_prefetch_stream = prefetch_stream()
    weight_prefetch_stream.wait_stream(calculation_stream)
    with npu_stream_switch(weight_prefetch_stream):
        maybe_npu_prefetch(inputs=weight, dependency=start_flag, max_size=max_weight_size)


def _prefetch_preprocess_impl_fake(weight: torch.Tensor, start_flag: torch.Tensor, max_weight_size: int) -> None:
    return


def _prefetch_postprocess_impl(stop_flag: torch.Tensor) -> None:
    calculation_stream = torch_npu.npu.current_stream()
    weight_prefetch_stream = prefetch_stream()
    calculation_stream.wait_stream(weight_prefetch_stream)


def _prefetch_postprocess_impl_fake(stop_flag: torch.Tensor) -> None:
    return


def _maybe_all_reduce_tensor_model_parallel_impl(final_hidden_states: torch.Tensor) -> torch.Tensor:
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    if (
        moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
        or _EXTRA_CTX.flash_comm_v1_enabled
    ):
        return final_hidden_states
    else:
        return tensor_model_parallel_all_reduce(final_hidden_states)


def _matmul_and_reduce_impl(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.custom_op is not None
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output = self.custom_op.matmul_and_reduce(input_parallel, bias_)

    return output


def _matmul_and_reduce_impl_fake(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    num_tokens = input_parallel.size(0)
    if _EXTRA_CTX.flash_comm_v1_enabled:
        num_tokens = _ceil_div(num_tokens + _resolve_fake_pad_size(), self.tp_size)
    output = torch.empty(
        size=(num_tokens, self.output_size_per_partition), device=input_parallel.device, dtype=input_parallel.dtype
    )

    return output


# TODO(Angazenn): The reason why we use a custom op to encapsulate npu_quantize
# is that aclnnAscendQuantV3(npu_quantize) use div_mode=False, while
# aclnnAddRmsNormQuantV2(npu_add_rms_norm_quant) use div_moe=True. We have to
# pass input_scale and input_scale_reciprocal at the same time to avoid redundant
# reciprocal calculation in fussion pass. We shall remove this once
# aclnnAddRmsNormQuantV2 supports div_moe=False.
def _quantize_impl(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _quantize_impl_fake(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _rope_forward_oot_impl_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _muls_add_impl_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="maybe_chunk_residual",
    op_func=_maybe_chunk_residual_impl,
    fake_impl=lambda x, residual: torch.empty_like(x),
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_gather_and_maybe_unpad",
    op_func=_maybe_all_gather_and_maybe_unpad_impl,
    fake_impl=_maybe_all_gather_and_maybe_unpad_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_pad_and_reduce",
    op_func=_maybe_pad_and_reduce_impl,
    fake_impl=_maybe_pad_and_reduce_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="prefetch_preprocess",
    op_func=_prefetch_preprocess_impl,
    fake_impl=_prefetch_preprocess_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="prefetch_postprocess",
    op_func=_prefetch_postprocess_impl,
    fake_impl=_prefetch_postprocess_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_reduce_tensor_model_parallel",
    op_func=_maybe_all_reduce_tensor_model_parallel_impl,
    fake_impl=lambda x: x,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="matmul_and_reduce",
    op_func=_matmul_and_reduce_impl,
    fake_impl=_matmul_and_reduce_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="quantize",
    op_func=_quantize_impl,
    fake_impl=_quantize_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_rotary_embedding",
    op_func=rope_forward_oot,
    fake_impl=_rope_forward_oot_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="muls_add",
    op_func=muls_add_triton,
    fake_impl=_muls_add_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

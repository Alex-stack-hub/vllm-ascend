# MTP Curl Original 310P Fallback

## Background

This note records the first 310P workaround for a hang observed with MTP when
`enable-prefix-caching` is enabled together with `mamba_cache_mode=align`.
Removing either `enable-prefix-caching` or `mamba_cache_mode=align` avoided the
hang in validation.

The problematic path is the Mamba align-mode postprocess path. Upstream uses a
Triton fused kernel through `postprocess_mamba_align_gpu` to update accepted
token metadata and copy Mamba state after MTP verification. 310P does not support
that Triton path, so the workaround replaced it with a CPU fallback.

## Original Approach

On 310P, monkey-patch `mamba_utils.postprocess_mamba_align_gpu` to a local
fallback and avoid launching the Triton fused postprocess kernel.

The first fallback copied a tensor of ones into
`num_accepted_tokens_cpu_tensor[:num_reqs]`. This made the next
`preprocess_mamba` step use `accept_token_bias = 0` for all active requests.

## Intended Benefit

- Avoid the 310P Triton fused postprocess launch that can hang.
- Keep the next `preprocess_mamba` path simple by making all requests start from
  accepted-token count 1.

## Risk

This does not preserve upstream semantics. Upstream only resets accepted-token
count to 1 when `src_block_idx == dest_block_idx`. For other requests it keeps
the actual accepted-token count.

Resetting every request to 1 can make the next `preprocess_mamba` compute the
wrong `accept_token_bias` when more than one draft token was accepted. That may
avoid the hang but can silently shift Mamba state copies and affect generation
correctness, especially around block boundaries or mixed batches.

## Branch And Commit

- Branch: `mtp_curl`
- Commit: `13e91f11b3 Add 310P mamba align postprocess fallback`

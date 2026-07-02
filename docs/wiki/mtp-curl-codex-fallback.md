# MTP Curl Codex 310P Fallback

## Background

This note records the revised 310P fallback for the MTP hang seen with
`enable-prefix-caching` and `mamba_cache_mode=align`.

The goal is to avoid the 310P Triton fused postprocess path while preserving the
metadata semantics that upstream `postprocess_mamba_align_gpu` provides for the
next Mamba preprocess step.

## Revised Approach

The revised fallback keeps the 310P safety property: it does not launch the
Triton fused postprocess kernel.

It also restores upstream accepted-token semantics:

1. Match the upstream function signature exactly, so the existing caller can
   invoke the fallback without a runtime keyword-argument error.
2. Read the CPU-side metadata already staged for postprocess:
   `mamba_state_idx`, `num_scheduled_tokens`, `num_computed_tokens`, and
   `num_draft_tokens`.
3. Copy the real accepted-token counts from `num_accepted_tokens_gpu` into the
   CPU tensor.
4. For each request, recompute the same block-alignment decision as upstream.
5. Reset accepted-token count to 1 only when `src_block_idx == dest_block_idx`;
   otherwise keep the actual accepted-token count.

## Why This Avoids New Risk

The fallback still avoids the hang-prone Triton kernel on 310P, but it no longer
collapses every request to accepted-token count 1. This preserves the
`accept_token_bias` used by the next `preprocess_mamba` call for requests where
multiple draft tokens were accepted.

That keeps the fix scoped to the 310P execution path and reduces the chance of
silent Mamba state drift in MTP decode, block-boundary, and mixed-batch cases.

## Validation Notes

Static checks performed locally:

- `python -m py_compile vllm_ascend/patch/worker/patch_mamba_utils.py`
- `git diff --check`

The local desktop Python environment did not include `vllm` or `torch`, so
hardware e2e validation still needs to run in the 310P test environment.

## Branch And Commit

- Branch: `mtp_curl_codex`
- Commit: `6a77b51918 Fix 310P mamba align fallback semantics`

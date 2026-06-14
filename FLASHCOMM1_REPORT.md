# FlashComm1 适配 Gemma4 系列模型验证报告

**日期**: 2026-06-12
**硬件**: Ascend NPU (A2, 4卡)
**vLLM 版本**: v0.20.2
**模型**: Gemma4-26B (MoE, 128 experts top_k=8), Gemma4-31B (Dense)

---

## 1. FlashComm1 原理概述

FlashComm1（也称 enable_sp / Sequence Parallelism）将标准 Tensor Parallel 中的 **AllReduce 操作分解为 ReduceScatter + AllGather**，每个 rank 只需计算部分 token 的 MatMul 和 RMSNorm，减少计算量和通信量。

```
标准 TP:
  [MatMul(全token)] → AllReduce → [RMSNorm(全token)]

FlashComm1:
  [MatMul(部分token)] → ReduceScatter → [RMSNorm(部分token)] → AllGather
                       ↑ tp=2: token数减半   ↑ 计算量减半   ↑ 恢复全量
```

**启用条件** (dense 模型):
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`
- `num_tokens > 1000`（运行时阈值，single/moe 模型无此限制）

---

## 2. 验证结果总览

| 模型 | 类型 | FC1 状态 | 服务启动 | 短 prompt | 长 prompt(>1000t) | 并发长 prompt | 最终结论 |
|------|------|---------|---------|----------|------------------|-------------|---------|
| **26B** | MoE | OFF | ✅ | ✅ `"Paris"` | — | — | 基线正常 |
| **26B** | MoE | ON (原版) | ❌ crash | — | — | — | 不可用 |
| **26B** | MoE | ON (适配) | ✅ | ❌ 乱码 | ❌ 乱码 | ❌ 乱码 | **不可用** |
| **31B** | Dense | OFF | ✅ | ✅ `"Paris"` | ✅ | ✅ | 基线正常 |
| **31B** | Dense | ON (原版) | ❌ crash | — | — | — | 不可用 |
| **31B** | Dense | ON (适配) | ✅ | ✅ `"Paris"` | ✅ 单请求 OK | ❌ crash | **并发不可用** |

---

## 3. Gemma4-26B (MoE) 详细分析

### 3.1 首次测试：原版 crash

**启用命令**:
```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
vllm serve /home/xty/gemma4/26B \
  --served-model-name gemma-4-26B-A4B-it \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' ...
```

**结果**: 服务启动阶段崩溃

**日志证据** — FC1 配置已确认（3个进程）:
```
(APIServer pid=123685) INFO [ascend_config.py:285]
  AscendConfig.enable_flashcomm1 falls back to
  VLLM_ASCEND_ENABLE_FLASHCOMM1 with value True.
```
> 来源: `test_log_flashcomm1_compile_fail.log`

**日志证据** — 张量维度不匹配:
```
torch._dynamo.exc.TorchRuntimeError:
  Dynamo failed to run FX node with fake tensors:
  call_function add(*(
      FakeTensor(size=((s59//2), 2816), dtype=bfloat16),   ← FC1 split 后 1024
      FakeTensor(size=(s59, 2816), dtype=bfloat16)          ← 残差仍是 2048
  ), **{}): got RuntimeError('The size of tensor a ((s59//2))
  must match the size of tensor b (s59: hint = 2048)
  at non-singleton dimension 0')
```
> 来源: `test_log_flashcomm1_compile_fail.log`

使用 `--enforce-eager` 后同样崩溃（跳过编译，直接推理时触发）:
```
RuntimeError: The size of tensor a (1024) must match
  the size of tensor b (2048) at non-singleton dimension 0
```
> 来源: `test_log_flashcomm1_eager_fail.log`

### 3.2 根因分析（26B 三个结构性不兼容）

#### 根因 1: Gemma4DecoderLayer 手动残差管理

```python
# Gemma4 原始（其他模型 Qwen3/DeepSeek 不是这样）:
residual = hidden_states                     # 全量 2048
hidden_states = self_attn(hidden_states)
hidden_states = o_proj(hidden_states)        # FC1: reduce_scatter → 1024
hidden_states = hidden_states + residual     # ❌ 1024 + 2044

# Qwen3 的方式（Norm 内部管理残差，FC1 兼容）:
hidden_states, residual = input_layernorm(x, residual)
# AscendRMSNorm.forward_oot() 内部处理 maybe_chunk_residual
```

#### 根因 2: MoE 模型 mmrs_fusion=False

```python
# ascend_forward_context.py:126 — 源码
if is_context_moe_model:
    mmrs_fusion = False   # 禁用融合 kernel，回退到非融合路径
else:
    mmrs_fusion = True    # Dense 模型可用 npu_mm_reduce_scatter_base
```

`npu_mm_reduce_scatter_base` 内部会做 all_gather → matmul → reduce_scatter，保证数学正确。非融合路径直接 reduce_scatter 但缺少 all_gather → 不同 rank 的 token 求和 → 数学错误。

#### 根因 3: 编译器 pass 无法匹配 Gemma4 Norm 操作

FC1 的 torch.compile pass 通过 FX graph 匹配重写计算图:

```
编译器期望:  npu_add_rms_norm_bias(x, r, weight, None, eps)
Gemma4 实际: npu_add_rms_norm(x, r, 1.0 + weight, eps)
             ↑ 不同算子       ↑ 不同参数签名
```

编译器 pass 无法匹配 → 图重写失败 → FC1 路径不完整。

### 3.3 26B 结论: ❌ 不可用

26B MoE 模型需要**编译器 pass 适配**（新增 `npu_add_rms_norm` pattern）和 **mmrs_fusion 评估**，单靠 worker patch 无法修复。

---

## 4. Gemma4-31B (Dense) 详细分析

### 4.1 首次测试：原版 crash（与 26B 相同根因）

**日志证据** — 张量维度不匹配:
```
(Worker_TP0 pid=229190) ERROR RuntimeError:
  The size of tensor a (1024) must match
  the size of tensor b (2048) at non-singleton dimension 0
```
> 来源: `/tmp/vllm_31b_nopatch.log`（31B + FC1 ON，零修改）

### 4.2 额外问题：31B 被误判为 MoE

31B config（`/home/xty/gemma4/31B/config.json`）包含:

```json
{
  "text_config": {
    "expert_intermediate_size": null,   // ← key 存在，value 为 null
    "num_experts": null,                // ← 同理
    "top_k_experts": null
  }
}
```

`_is_contain_expert()` 递归检查所有 key 是否含 `"expert"`，不检查 value。31B 被误判为 MoE 模型，导致被 FC1 断言拦截:

```
VllmConfig validation error:
  Assertion failed, Flash Comm v1 requires
  enable_expert_parallel=True for MoE models.
```
> 来源: `/tmp/vllm_31b_bare.log`

修复（`utils.py:976`）:
```diff
- if "expert" in str(k):
+ if "expert" in str(k) and v:   # 跳过 value 为 None/falsy 的 key
```

### 4.3 两处修复后的状态

**修复 1**: `utils.py` — `_is_contain_expert()` 跳过 null value
**修复 2**: `patch_gemma4_flashcomm.py` — 覆盖 `Gemma4DecoderLayer.forward`，手动 add 前 chunk residual

**短 prompt 验证（FC1 运行时未激活）**:
```
Test: What is the capital of France?
FC1 ON:  'Paris'   ✅
FC1 OFF: 'Paris'   ✅
```
> 来源: `test_log_fc1_31b_final.log` — 5 进程确认 FC1 启用，推理正常

### 4.4 长 prompt 单请求测试（FC1 激活，正常）

**~1400 token prompt，FC1 ON**:
```
Prompt: 855 words, ~1458 tokens (>1000 阈值)
Elapsed: 5.0s
Output: "You have provided the same introductory text on quantum mechanics..."
```
没有崩溃，输出语义正确。

### 4.5 并发长 prompt benchmark — ❌ 崩溃

**benchmark 配置**: 6 请求, 3 并发, ~1400 token/prompt, 128 max_tokens

**FC1 OFF 基线**:
```
Throughput:  15.8 tok/s
TTFT mean:   710ms
TTFT p50:    718ms
Errors:      0
```
> 来源: `/tmp/vllm_fc1_off_final.log`

**FC1 ON**:
```
WorkerProc hit exception
EngineCore encountered a fatal error
```
> 来源: `/tmp/vllm_fc1_on_final2.log`

### 4.6 崩溃根因：Flash Attention 硬件错误

```
(Worker_TP1) ERROR RuntimeError:
  npu_fusion_attention:.../opapi/FlashAttentionKernelNpuOpApi.cpp:1345
  NPU function error: call aclnnFlashAttentionVarLenScore failed,
  error code is 161001
```
> 来源: `/tmp/vllm_fc1_on_final2.log`（共 6 次重复）

**崩溃路径**:
```
Gemma4DecoderLayer.self_attn()
  → AscendAttentionBackend.forward_impl()
    → attention_v1.py:1495 — attn_state == ChunkedPrefill
      → attention_v1.py:1499 — _forward_large_head_prefill_attention()
        → attention_v1.py:1327 — torch_npu.npu_fusion_attention(
            ...
            actual_seq_qlen=attn_metadata.actual_seq_lengths_q,  ← 全量序列边界
            actual_seq_kvlen=actual_seq_lengths_kv,              ← 未随 FC1 拆分
            ...
          )
```

**根本原因**: FC1 将 `num_tokens` 按 `tp_size=2` 拆分（2048→1024），每个 rank 只持有部分 token。但 attention metadata 中的 `actual_seq_lengths_q/kv`（变长序列边界数组）是**全量构建**的（`AscendMetadataBuilder` 填充，未感知 FC1 的 token split）。传给 `npu_fusion_attention` 后，`actual_seq_qlen[-1]` 与实际的 query tensor dim=0 不一致 → NPU 硬件报错 161001。

**代码位置**: `vllm_ascend/attention/attention_v1.py:1327` 和 `vllm_ascend/attention/attention_v1.py:1294`

### 4.7 31B 结论: ⚠️ 单请求可用，并发不可用

- 短 prompt (FC1 未激活): ✅ 正常
- 长 prompt 单请求 (FC1 激活): ✅ 正常
- 长 prompt 并发 (FC1 激活): ❌ `npu_fusion_attention` 硬件错误 161001

**原因**: Attention metadata 构建链路未随 FC1 的 token split 同步调整，导致 `actual_seq_lengths_q` 与 query tensor 维度不一致。单请求能通过是因为 batch 简单（仅 1 条序列），并发时多条变长序列边界冲突触发硬件报错。

---

## 5. 代码修改清单

| 文件 | 改动 | 作用 | 状态 |
|------|------|------|------|
| `vllm_ascend/utils.py:976` | `if "expert" in str(k) and v` | 修复 31B 被误判为 MoE | ✅ 已完成 |
| `vllm_ascend/patch/worker/patch_gemma4_flashcomm.py` | 新建，覆盖 forward，add 前 chunk residual | 修复残差维度不匹配 crash | ✅ 已完成 |
| `vllm_ascend/patch/worker/__init__.py` | +1 行 import | 注册 patch | ✅ 已完成 |
| `vllm_ascend/attention/attention_v1.py:1327` | `actual_seq_lengths_q/kv` 适配 FC1 token split | 修复并发 flash attention crash | ❌ 待上游修复 |
| `vllm_ascend/compilation/passes/sequence_parallelism.py` | 新增匹配 `npu_add_rms_norm` 的 compiler pattern | 修复 26B compiler pass | ❌ 待上游修复 |
| `vllm_ascend/ascend_forward_context.py:126` | 评估 MoE 模型启用 `mmrs_fusion=True` | 修复 26B 非融合路径 | ❌ 待上游评估 |

---

## 6. 日志证据文件清单

| 文件 | 内容 |
|------|------|
| `test_log_fc1_OFF_verify.log` | 26B FC1 OFF 基线：`enable_sp=False`，输出 `"Paris"` |
| `test_log_flashcomm1_compile_fail.log` | 26B FC1 ON compile 模式：22 RuntimeErrors，残差维度不匹配 |
| `test_log_flashcomm1_eager_fail.log` | 26B FC1 ON eager 模式：8 RuntimeErrors，`1024 vs 2048` |
| `test_log_fc1_31b_final.log` | 31B FC1 ON patched：5 进程确认 `enable_flashcomm1=True`，短 prompt 输出 `"Paris"`，0 errors |
| `test_log_fc1_31b_success.log` | 31B FC1 ON patched 早期版本：推理正常 |
| `vllm_31b_nopatch.log` | 31B FC1 ON 无 patch：8 RuntimeErrors，`1024 vs 2048` |
| `vllm_fc1_off_final.log` | 31B FC1 OFF benchmark 基线：15.8 tok/s, TTFT 718ms, 0 errors |
| `vllm_fc1_on_final2.log` | 31B FC1 ON benchmark crash：6 RuntimeErrors，`npu_fusion_attention error 161001` |
| `vllm_31b_bare.log` | 31B FC1 ON 零修改：MoE 断言拦截 |
| `bench_fc1_v2.py` | Benchmark 脚本 |

---

## 7. 总结

| 维度 | 26B (MoE) | 31B (Dense) |
|------|----------|------------|
| 残差 crash | ✅ Patch 已修复 | ✅ Patch 已修复 |
| MoE 检测 bug | N/A (确实是 MoE) | ✅ `utils.py` 修复 |
| mmrs_fusion 路径 | ❌ False → 非融合路径 bug | ✅ True → 融合 kernel 正常 |
| 编译器 pass | ❌ 算子不匹配 | ✅ 匹配（若走 compile） |
| Flash Attention 并发 | N/A (前置条件未通过) | ❌ `npu_fusion_attention` 错误 161001 |
| **最终状态** | **❌ 不可用** | **⚠️ 单请求可用，并发不可用** |

FlashComm1 对 Gemma4 系列的支持需要两部分工作：
1. **已完成**: 残差 patch + MoE 检测修复（防 crash 和 config 误判）
2. **待上游**: 26B compiler pass + attention metadata FC1 适配 + mmrs_fusion MoE 评估

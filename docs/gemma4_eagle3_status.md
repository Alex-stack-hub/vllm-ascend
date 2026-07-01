# Gemma 4 31B + EAGLE3 Ascend 适配状态

**分支**: `gemma4-eagle3` | **日期**: 2026-07-01 | **vLLM 版本**: v0.20.2

---

## 改动

| Commit | 内容 |
|--------|------|
| `e32d814d` | `llm_base_proposer.py` +2: Gemma4 `image_token_id` 映射 |
| `22cdff9e` | `model_runner_v1.py`: EAGLE3 aux 时 target 跳过图模式 |

---

## 验证矩阵

| 配置 | 状态 |
|------|------|
| Target eager | ✅ PASS |
| Target graph | ✅ PASS |
| EAGLE3 eager | ✅ PASS |
| EAGLE3 (target eager + draft graph) | ✅ PASS |
| EAGLE3 (target graph + draft eager) | ❌ aux 全零 |

## 真实 target graph 根因

`EagleModelMixin._maybe_add_hidden_state` 使用 `list.append()` — **NPU graph 不捕获 Python list 操作**。

### 实验性尝试（均失败）

| 方法 | 现象 |
|------|------|
| `copy_()` 写入预分配 buffer | graph 捕获写入的是 dummy_run 时的零值，replay 无法更新 |
| `torch.stack()` + 返回 `(hs, stacked_aux)` | 同上，stacked_aux 保持零值 |
| flatten 多元素 tuple | dump_run 期间 aux list 始终为空 |
| clone() | 图不捕获 |

### Trace 证据

```
EAGER aux sums: [511.61, 4723.8, 1002.63]  (fresh every step)
GRAPH aux sums: [0.0, 0.0, 0.0]             (stale from dummy_run)
```

**结论**: `torch.npu.graph` 将 `copy_()` 视为值复制（捕获写入的值），不视为操作调用。replay 时写入捕获时的原值（零），而非实时计算结果。

### 目前方案

EAGLE3 aux 激活时 target 使用 eager（与 Qwen3 EAGLE3 一致，`enforce_eager` 在 speculative config 中）。Draft 可使用图模式。真正的 target graph 需要 NPU graph 层面的 API 支持。

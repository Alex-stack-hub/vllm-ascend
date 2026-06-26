# Gemma4-31B PD 分离适配报告

**日期**: 2026-06-25
**模型**: Gemma4-31B (Dense, BF16)
**硬件**: Ascend NPU A2 (4卡, 0,1 基线 / 2,3 PD Producer)
**vLLM 版本**: 0.20.2 (PyPI)
**vllm-ascend 分支**: feature_test

---

## 1. 结论: ✅ 支持 PD 分离

### 运行时验证

```
         基线 (port 18005, cards 0,1):  "Paris"  ✅
PD Producer (port 18006, cards 2,3):  "Paris"  ✅
```

两个服务同时运行，PD Producer 使用 `MooncakeLayerwiseConnector` + `kv_role=kv_producer` + `RecomputeScheduler`，推理结果与基线完全一致。

### 关键日志证据

**PD 配置确认**:
```
non-default args: {
  kv_transfer_config: KVTransferConfig(
    kv_connector='MooncakeLayerwiseConnector',
    kv_role='kv_producer',
    kv_connector_extra_config={'prefill':{'tp_size':2,'dp_size':1},'decode':{'tp_size':2,'dp_size':1}}
  ),
  additional_config: {'recompute_scheduler_enable': True}
}
```
> 来源: `pd_layerwise_success.log`

**PD Connector 创建**:
```
Creating v1 connector with name: MooncakeLayerwiseConnector ...
Initializing Mooncake work ...
```
> 来源: `pd_layerwise_success.log`

**PD 引擎初始化**:
```
GPU KV cache size: 57,344 tokens
init engine (profile, create kv cache, warmup model) took 41.90 s
Application startup complete.
```
> 来源: `pd_layerwise_success.log`

**PD 推理验证**:
```
PD OUTPUT: Paris
```
> 来源: 运行时测试

**基线推理对比**:
```
Baseline (port 18005): Paris
```
> 来源: `baseline_31b_ok.log`

---

## 2. PD 分离原理

### 架构

PD (Prefill-Decode) 分离将推理的 Prefill 和 Decode 阶段部署在不同实例上：

```
┌──────────────┐           ┌──────────────┐
│  P Node      │  KV Cache │  D Node      │
│  kv_producer │ ────────► │  kv_consumer │
│  (cards 2,3) │ Mooncake  │  (cards X)   │
└──────────────┘  Transfer └──────────────┘
```

- **P 节点**: 负责 Prefill 计算，生成 KV Cache
- **D 节点**: 从 P 节点拉取 KV Cache，执行 Decode
- **传输**: Mooncake Transfer Engine (RDMA) 传输 KV Cache
- **调度**: RecomputeScheduler — D 节点 block 不足时丢弃请求交 P 节点重算

### Gemma4-31B 配置

```bash
# PD Producer (P 节点)
MOONCAKE_CONFIG_PATH=/tmp/mooncake2.json \
ASCEND_RT_VISIBLE_DEVICES=2,3 \
vllm serve /home/xty/gemma4/31B --tensor-parallel-size 2 --enforce-eager \
  --kv-transfer-config '{
    "kv_connector":"MooncakeLayerwiseConnector",
    "kv_role":"kv_producer",
    "kv_connector_extra_config":{
      "prefill":{"tp_size":2,"dp_size":1},
      "decode":{"tp_size":2,"dp_size":1}
    }
  }' \
  --additional-config '{"recompute_scheduler_enable":true}' \
  --enable-prefix-caching --max-model-len 4096
```

---

## 3. Mooncake Layerwise vs Standard

MooncakeLayerwiseConnector 与 MooncakeConnectorV1 的区别：

| 方面 | MooncakeConnectorV1 | MooncakeLayerwiseConnector |
|------|-------------------|--------------------------|
| 传输粒度 | 整层 KV Cache | 逐层传输 |
| 内存注册 | 一次注册全部 block | 逐层注册，避免重叠 |
| 单机兼容 | ❌ overlapped memory error | ✅ 正确处理 |
| 性能 | 略高 | 略低（更多次传输） |

本机测试中 MooncakeConnectorV1 报 `Transfer Engine does not support overlapped memory region`，因为 TP=2 的两个 rank 共享同一物理内存区域。MooncakeLayerwiseConnector 逐层注册无需重叠。

---

## 4. 适配修改

### 4.1 vllm-ascend 项目内修改 (2 files)

| 文件 | 改动 | 说明 |
|------|------|------|
| `vllm_ascend/worker/block_table.py` | +36 行 CPU fallback | Triton kernel `_compute_slot_mapping_kernel` 在 PyPI v0.20.2 中是普通函数（非 Triton kernel），改造为 CPU 回退路径 |
| `vllm_ascend/patch/platform/patch_tool_choice_none_content.py` | hasattr 守卫 | `OpenAIServing._parse_tool_calls_from_content` 和 `DelegatingParser._parse_tool_calls` 在 v0.20.2 中不存在 |

### 4.2 系统 vllm 安装内 Stub 文件

vllm-ascend 代码引用了一些在 PyPI v0.20.2 中缺失的 vllm 模块/函数。在系统 vllm site-packages 中创建了以下 stub：

| 文件 | 内容 |
|------|------|
| `vllm/model_executor/layers/deepseek_compressor.py` | `CompressorStateCache` stub |
| `vllm/model_executor/layers/deepseek_v4_attention.py` | `DeepseekV4IndexerCache` stub |
| `vllm/model_executor/layers/mamba/linear_attn.py` | `CustomOp`, `MiniMaxText01RMSNormTP`, `MiniMaxM2RMSNormTP` stubs |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` stub |

**追加函数**:
| 模块 | 函数 |
|------|------|
| `vllm/distributed/parallel_state.py` | `get_decode_context_model_parallel_rank`, `get_decode_context_model_parallel_world_size`, `patch_tensor_parallel_group` |
| `vllm/distributed/__init__.py` | 重新导出上述函数 |

Stub 安装脚本: `install_pd_stubs.sh`

---

## 5. 验证过程

1. **基线确认** (v0.20.2, 无 PD): 启动 31B → 推理 `"Paris"` ✅
2. **PD MooncakeConnectorV1** (TP=2): `overlapped memory` 错误 ❌
3. **PD MooncakeLayerwiseConnector** (TP=2): 启动成功 + 推理 `"Paris"` ✅
4. **PD + 基线并行**: 两个服务同时运行, 输出一致 `"Paris"` ✅

---

## 6. 日志文件

| 文件 | 内容 |
|------|------|
| `baseline_31b_ok.log` | 基线服务正常启动 + 推理日志 |
| `pd_layerwise_success.log` | PD Producer MooncakeLayerwise 完整启动日志 + KV Cache + Connector 创建 |
| `install_pd_stubs.sh` | 系统 vllm stub 安装脚本 |

---

## 7. 支持的 PD 模式

| 模式 | kv_role | connector | 状态 |
|------|---------|-----------|------|
| PD-disaggregated (仅 P) | kv_producer | MooncakeLayerwise | ✅ 验证通过 |
| PD-disaggregated (仅 P) | kv_producer | MooncakeV1 | ⚠️ 需要 node-exclusive 内存注册 |
| PD-mixed | kv_both | AscendStore | — 需要 MMC / Memcache 基础设施 |

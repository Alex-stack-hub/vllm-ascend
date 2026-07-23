---
name: adapt-vllm-ascend-model
description: 面向 vLLM Ascend 的端到端模型适配流程。适用于分析新模型结构、实现 A2/A3/A5 设备路径、适配注意力/MoE/KV Cache/量化/MTP、定位 eager 与 ACLGraph 精度差异、建立性能基线和 CI、准备可合入主干的 PR，以及编写模型适配 Wiki。
---

# vLLM Ascend 模型适配

## 目标

交付语义正确、设备差异清晰、图模式稳定、性能可测、代码可维护且有持续测试的模型支持。服务能够启动不等于适配完成。

## 必须遵循的流程

### 1. 固定证据基线

修改代码前记录：

```text
vLLM 与 vLLM Ascend SHA
模型仓库与 revision
设备/SOC、驱动、CANN、torch、torch-npu
TP/DP/EP/PP 配置
图模式与 capture sizes
量化格式
完整启动命令与数据集
```

检查当前工作树和上游模型实现。当前代码及原始产物是权威证据，历史 PR 只能作为参考。

### 2. 分析模型语义

按本文的“模型结构分析清单”执行，并输出：

- 模型结构与配置矩阵；
- Layer pattern 和 tensor shape；
- Attention mask、scale、RoPE 与 Cache 所有权；
- Dense/MoE 结构和激活函数；
- Multimodal、PLE、KV Sharing、speculative state；
- Checkpoint 到 runtime module 的权重映射；
- 后端必须保持的语义不变量。

这些不变量明确前，不要开始写设备后端代码。

### 3. 建立设备能力矩阵

对所有关键 shape 和执行阶段分别记录：

| 维度 | 示例 |
|---|---|
| 设备 | A2、A3、A5 |
| 阶段 | profile、prefill、chunked prefill、decode |
| 后端 | FIA、PA、Fusion Attention、Triton/自定义算子 |
| Shape | head dim、KV heads、dtype、block size |
| 执行 | eager、compile、full graph、piecewise |

已有抽象允许时，将设备算子选择放入 DeviceAdaptor 或 device utils。模型语义和通用 attention 流程不直接感知设备型号。

### 4. 梳理端到端执行链

跟踪：

```text
model forward
 -> Q/K/V 或 expert routing
 -> Cache 读写
 -> Backend 选择
 -> Graph capture
 -> Graph 参数更新
 -> Replay
 -> Output/logits
```

图模式必须证明：

```text
capture backend == replay backend
capture 参数类型 == replay update 类型
metadata owner == 模型语义中的 owner
workspace >= 所有 replay 算子的需求
mask/scale/state 与 eager 语义一致
```

Graph 启动或精度异常时按本文的“Ascend 图模式排障”执行。

### 5. 实现最小但语义完整的修改

优先：

- 复用当前仓库的 helper 和抽象；
- 使用显式参数类型，不依赖 tuple 长度；
- 使用明确模型语义，不依赖运行时 signature 反射；
- 将 SOC 差异放入 DeviceAdaptor；
- 在 capture 阶段计算缓存，避免 replay 动态工作；
- 量化例外严格绑定模型配置；
- 同时测试修改路径与未修改路径。

禁止：

- 机械复制旧 PR；
- 使用 `len(param)` 判断 backend；
- 能显式记录 layer identity 时仍从任意数字字符串推导；
- 让所有模型承担目标模型的运行时成本；
- 在正式修复中静默回退 eager；
- 仅依据最终答案统计宣称根因。

### 6. 分层验证

按本文的“验证与性能”依次执行：

1. Import/config/权重映射 UT；
2. Eager 启动与确定性 smoke；
3. Graph 启动与确定性 smoke；
4. 执行阶段和 shape 专项测试；
5. Dense/MoE、producer/target 等控制组；
6. 小数据集确认方向；
7. 全量精度数据集；
8. 输出长度、重复率、no-answer；
9. 固定 workload 性能；
10. CI 接入。

每次根因实验只改变一个主要变量。

### 7. 将精度作为一等要求

Eager 正常、graph 异常时：

1. 固定为确定性采样；
2. 找到第一个分叉 token；
3. 找到第一个分叉 layer；
4. 先比较 attention output，再比较 router；
5. 对 MoE 比较 logits、top-k、dispatch、combine；
6. 检查 padding、active token、cache owner、mask 与 sparse mode；
7. 仅将 fallback backend 用作隔离实验；
8. 用全量数据确认。

结论必须标记：

- **已确认**：存在直接 A/B、代码或运行时证据；
- **已定位范围**：已隔离到子系统，但未证明具体字段；
- **待验证假设**：机制合理，尚缺实验。

### 8. 量化与 MTP

量化需要检查：

- 合法缺失或共享权重；
- Packed module mapping；
- Checkpoint prefix；
- 激活函数；
- Cache format 与自定义算子构建；
- Graph 支持；
- 全量精度，而不只是启动。

MTP 需要检查：

- Draft/target layer 和 cache 对齐；
- Q-only 或共享 KV 语义；
- 每个 draft step 的 graph metadata；
- 各位置 acceptance；
- 平均提交 token；
- Draft、verify、sync、graph update 耗时；
- MTP on/off 的精度与性能。

不能将 acceptance 直接写成加速倍数：

$$
\mathrm{Speedup}
\approx
\frac{\mathbb{E}[\mathrm{progress}]\cdot T_{\mathrm{baseline}}}
{T_{\mathrm{draft}}+T_{\mathrm{verify}}+T_{\mathrm{sync}}}
$$

### 9. 建立性能基线

每项结果必须包含：

```text
设备与软件版本
模型/量化 checkpoint
并行与 graph mode
输入/输出长度
并发与请求数
预热和重复次数
TTFT、TPOT、吞吐、HBM
精度门禁
原始产物路径
```

功能观测和性能基线必须分开表述。

### 10. 增加长期看护

至少包括：

- Helper 和分支选择 UT；
- PR 启动/短 graph generation smoke；
- 代表性 Dense/MoE nightly 精度；
- Loop 和输出长度指标；
- 性能敏感特性的周期基线。

引用绿色 CI 前，先确认该测试实际覆盖了什么。

### 11. 生成可上游交付物

准备：

- 最小且带 sign-off 的提交；
- 聚焦实现的 PR 描述；
- 准确的测试命令和结果；
- 已知限制；
- 使用本文的“模型适配 Wiki 模板”编写的适配 Wiki。

Wiki 保持紧凑结构：

```text
README 总览
模型结构分析
A2/A3/A5 适配过程
特性接入情况
量化适配
精度、性能基线与调优
```

对外发布的适配文档不能只罗列结论。以分享者视角组织每个关键问题：

```text
背景：当时要实现什么，为什么需要
现象：在哪个设备、模型和模式下出现什么报错或精度变化
假设：根据哪些控制组形成哪些候选原因
验证：每次只改变什么变量，观测到什么证据
修复：为什么选择当前方案，没有选择哪些粗粒度 fallback
收益：正确性、性能或维护性得到什么改善
边界：哪些结论已确认，哪些仍待验证
```

先用通俗语言解释原理，再给公式、参数和代码路径。让读者能够复用分析方法，而不是只能记住最终改动。

## 完成门禁

满足以下证据后才能宣称适配完成：

- 模型语义和 checkpoint mapping 正确；
- 必需的设备/阶段路径可用；
- Eager 与 graph 精度达到门禁；
- Dense/MoE 或等价结构变体有覆盖；
- 量化/MTP 结论不超出测试范围；
- 性能数据可复现；
- PR 与 nightly CI 已接入；
- 已知限制清晰；
- 文档可追溯到原始证据。

## 模型结构分析清单

### 配置

- 模型类型与变体
- Layer 数、hidden size、intermediate size
- Attention heads、KV heads、head dimension
- Layer type pattern 和 sliding window
- 各 layer type 的 RoPE 参数
- Attention scale、soft cap、mask
- Dense/MoE、专家数、Top-k、激活函数
- Shared KV、Shared Expert、PLE、YOCO
- Multimodal encoder 与 token 扩展
- 最大上下文与 Cache dtype

### Tensor 语义

每个阶段记录：

```text
TP 前后的 Q/K/V shape
Cache layout 与读写 owner
Position/RoPE owner
Mask 与 sparse mode
Output shape 与 buffer owner
```

MoE 额外记录：

```text
Router input/logits
Top-k ids/weights
Dispatch token 顺序
Tokens per expert
Expert activation
Combine 顺序和权重
```

### 权重映射

- Runtime module path
- Checkpoint tensor path
- Packed/fused module mapping
- 模型语义允许共享或缺失的权重
- TP/EP shard 规则
- Quant description mapping
- Tied embedding/lm_head

### 控制组

- Dense vs MoE
- Sliding vs global
- Producer vs KV-sharing target
- Quantized vs BF16
- Text vs multimodal
- Draft vs target

### 输出

1. 配置表；
2. 架构图；
3. 分阶段 shape 表；
4. Cache 所有权图；
5. 后端不变量；
6. 验证矩阵。

## Ascend 图模式排障

### Capture/Replay 快照

每个算子记录：

```text
Rank
阶段：capture/update/replay
Layer name 与语义 owner
Attention type
Head size 与 KV heads
Num tokens 与 graph bucket
Backend
Workspace bytes
Block table/seq lengths shape
Mask 是否存在及 shape
Sparse mode
Input/output norm 或 checksum
```

MoE 增加：

```text
Active/padded token 数
Top-k ids/weights checksum
Expanded row index 范围/checksum
Tokens per expert
Dispatch/combine backend
```

### 常见问题

#### Workspace 不匹配

检查同一 token bucket 是否存在不同 layer shape。Capture 时缓存最大 workspace；除非框架契约要求，否则不要在 replay 动态重算。

#### Tuple unpack 不匹配

将其视为 backend identity 错配。使用显式 graph param 类型，不使用 tuple 长度。

#### Eager 正常、graph 异常

检查 layer identity、metadata owner、cache write、mask/sparse mode、padding、active token、graph input copy 和通信 metadata。

#### Dense 正常、MoE 异常

先比较 router 前的 hidden state，再比较 router logits、top-k、dispatch、expert output 和 combine。Attention 漂移可能被离散路由放大。

#### 短 prompt 异常

扫描 token length 和 graph bucket，检查 padding 比例、最小 block table、position、prompt template 和首个 decode transition。

### 调试日志

使用环境变量开关和输出次数上限。打印摘要，不打印大 tensor。定位完成后从发布代码删除。

## 验证与性能

### 精度矩阵

| 维度 | 最小覆盖 |
|---|---|
| 执行 | Eager、目标 graph mode |
| 采样 | Greedy 定位、推荐采样评测 |
| 阶段 | Prefill、Chunked Prefill、Decode |
| 结构 | 代表性 Dense/MoE/Shared-KV |
| Batch | 1 和目标 batch |
| Prompt | 短、中、长；支持时加入 multimodal |
| 精度 | BF16 和支持的量化格式 |

采集：

```text
Accuracy
No-answer 数
Completion token 平均/中位/最大值
Max-token 触顶数
重复率
Latency 分布
Server error
```

### 性能基线

报告：

- TTFT p50/p90/p99
- TPOT
- Request/token throughput
- 单用户 decode rate
- 设备显存
- Graph capture/compile 时间
- 预热和重复次数

模型 revision、输入/输出长度、并发、graph mode 或并行不同时，不得直接比较结果。

### MTP

记录：

```text
各 draft position acceptance
平均 accepted/committed token
Draft time
Target verification time
KV synchronization time
Graph update time
MTP on/off 的 TTFT、TPOT、吞吐和精度
```

扫描 $k$；只有新增 accepted progress 高于 draft 与 verification 额外成本时，更大的 $k$ 才有价值。

### CI 分层

```text
UT -> PR 启动/graph smoke -> Nightly 精度 -> 周期性能
```

PR smoke 不是精度证据；Nightly 精度也不是性能基线。

## 模型适配 Wiki 模板

### README 总览

- 范围与模型变体
- 设备和特性矩阵
- 关键精度与性能结果
- 五篇主体文档链接

### 01 模型结构分析

- 配置矩阵
- Attention/MoE/Cache 语义
- Multimodal 或 speculative 结构
- Tensor shape 与不变量

### 02 A2/A3/A5 适配过程

- 设备能力矩阵
- Prefill/Decode backend
- Graph capture/replay
- 问题、根因与修复
- 关键 PR 和代码边界

### 03 特性接入情况

- 运行时特性和原理
- 支持/证据矩阵
- CI 覆盖和限制

### 04 量化适配

- Checkpoint mapping
- 激活函数与 Cache format
- 验证结果和支持范围

### 05 精度、性能基线与调优

- 精度回归和根因证据
- 固定 workload 性能
- MTP acceptance 与端到端收益
- 调优顺序
- 原始报告和 CI 链接

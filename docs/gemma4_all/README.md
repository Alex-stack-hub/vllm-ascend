# Gemma4 on Ascend 适配总览

> 适配设备：Atlas A2、Atlas A3、Ascend 950（A5）
>
> 模型范围：Gemma4 31B Dense、26B-A4B MoE、E2B、E4B
>
> 内容范围：模型结构、Attention、ACLGraph、MoE、KV Cache、量化、运行时特性、精度、性能与 MTP

## 1. 这套文档做了什么

这套文档整理了 Gemma4 在 vLLM Ascend 上的完整适配过程。我们不仅记录最终实现，还把适配过程中遇到的设备能力差异、图模式错误、精度回归和量化问题一起保留下来。

主要完成了六类工作：

| 工作 | 具体内容 |
|---|---|
| 模型结构分析 | 梳理交错 Attention、K=V、Dense/MoE、PLE、YOCO 和 KV Sharing |
| A5 适配 | 使用原生 FIA 执行 256/512-head Attention，修复 workspace 和 replay metadata |
| A2/A3 适配 | 为 FIA TND 不支持的 512-head 设计 Decode PA 和 Prefill fallback |
| 图模式适配 | 处理 Mixed PA/FIA、显式 graph param、layer identity、mask 和 workspace |
| 量化与特性 | 适配 ModelSlim W8A8、GELU-TANH、C8 Cache、Prefix Cache、EP 等能力 |
| 验证与看护 | 建立 eager/graph、Dense/MoE、精度、性能、PR E2E 和 Nightly 基线 |

最终目标是让同一套模型语义在不同设备、不同执行模式和不同优化组合下保持一致，而不是要求所有设备使用完全相同的算子。

## 2. 为什么需要这些适配

Gemma4 不是一个只需要注册模型名称就能运行的常规模型。它同时包含多种会影响后端执行的结构：

1. Sliding Attention 与 Global Attention 交错出现。
2. Sliding head dimension 为 256，Global head dimension 为 512。
3. 31B 使用 Dense FFN，26B-A4B 使用 128 experts、Top-8 的 MoE。
4. Global layer 使用 `attention_k_eq_v`，checkpoint 不一定有独立 `v_proj`。
5. E2B/E4B 使用 PLE、YOCO 和跨层 KV Sharing。
6. 多模态输入会改变 Prefill token 数、Graph bucket 和 KV Cache 压力。
7. A2/A3 与 A5 的 FIA 能力不同，不能直接共用 512-head 执行方案。

这些差异会沿着整条推理链传播：

```mermaid
flowchart LR
    A["模型结构"] --> B["Q/K/V 与 Router"]
    B --> C["Attention / MoE 算子"]
    C --> D["KV Cache"]
    D --> E["Graph Capture"]
    E --> F["参数更新与 Replay"]
    F --> G["Logits 与生成结果"]
```

因此，适配需要同时关注模型语义、设备算子能力和图执行状态，任何一层理解错误都可能表现为启动失败、精度下降或重复输出。

## 3. 我们遇到了哪些主要问题

| 设备/场景 | 现象 | 问题范围 | 最终处理 |
|---|---|---|---|
| A5 Graph | FIA workspace 传入约 55 MB，实际需要约 104 MB | 256/512 交错层共享 token bucket | Capture 阶段缓存 bucket 内最大 workspace |
| A5 Graph | Eager 正常，Graph 出现重复、乱码或特殊符号 | Replay layer metadata 与执行层不一致 | Capture 时显式记录 layer/cache owner |
| A2/A3 | FIA TND 无法执行 512-head Global Attention | 设备算子能力缺口 | Decode 使用 PA，Prefill 使用 large-head fallback |
| A2/A3 Graph | `expected 21, got 9` | PA Capture 与 FIA Update 类型错配 | 使用显式 `PagedAttentionGraphParam` |
| A2/A3 RoPE | UB 超出约 8,192 bits | Large head 的局部 tile 过大 | `head_dim>=256` 时减小 head tile |
| A5 MoE Graph | GPQA-D 约 73% 降到约 57% | MoE Graph dispatch/combine 组合路径 | 使用 ALLGATHER 做控制组并收敛问题范围 |
| A2 Nightly | GPQA-D 约 70% 降到约 47%，重复率上升 | FIA Decode mask/sparse-mode 语义变化 | 恢复回归前的完整参数语义 |
| W8A8 | 缺少 `v_proj` 或输出精度异常 | K=V、专家路径和激活函数不匹配 | 窄范围权重映射与 GELU-TANH 分支 |

详细报错、分析证据和方案取舍见 [A2/A3/A5 适配过程](02_A2_A3_A5适配过程.md)、[量化适配](04_量化适配.md) 和 [精度、性能基线与调优](05_精度性能基线与调优.md)。

## 4. 我们如何组织适配工作

适配按由简单到复杂的顺序推进：

```text
读取模型配置和实现
  -> 明确 layer、shape、mask、activation 和 cache owner
  -> 建立 A2/A3/A5 设备能力矩阵
  -> 先验证 eager，再验证 ACLGraph
  -> 先验证 Dense，再增加 MoE 动态路由
  -> 再接入量化、运行时特性和 MTP
  -> 最后建立精度、性能和 CI 基线
```

这个顺序让每一步都有控制组：

- A5 原生 FIA 可以帮助确认模型 Attention 语义；
- A2/A3 只处理不支持的 512-head shape；
- 31B Dense 用于观察公共 Attention 和 Graph；
- 26B MoE 用于观察 Router、Expert 和通信；
- Eager 用于对照 Graph Capture/Replay；
- ALLGATHER 用于对照 MC2/ALLTOALL。

## 5. 详细文档

建议首次阅读时按顺序进行：

| 文档 | 主要回答的问题 |
|---|---|
| [01 模型结构分析](01_模型结构分析.md) | Gemma4 的结构有什么特殊之处，这些结构怎样影响后端设计 |
| [02 A2/A3/A5 适配过程](02_A2_A3_A5适配过程.md) | 不同设备遇到了什么错误，如何分析并设计对应执行路径 |
| [03 特性接入情况](03_特性接入情况.md) | 接入了哪些运行时特性，它们的原理、收益和验证方式是什么 |
| [04 量化适配](04_量化适配.md) | W8A8 为什么需要单独处理 K=V、MoE 激活和 Cache |
| [05 精度、性能基线与调优](05_精度性能基线与调优.md) | 如何定位 Graph 精度问题，当前精度和性能结果如何 |
| [Gemma4 模型适配 Skill](Gemma4模型适配_SKILL.md) | 如何把本次方法复用于后续 vLLM Ascend 模型适配 |

如果已经遇到具体问题，可以直接查阅：

| 问题 | 对应章节 |
|---|---|
| A5 FIA workspace mismatch | 02 的 A5 适配 |
| A2/A3 512-head 不支持 | 02 的 A2/A3 适配 |
| Graph 参数 9/21 解包错误 | 02 的 Mixed PA/FIA |
| Eager 正常、Graph 重复或乱码 | 02 的 layer-aware replay、05 的精度分析 |
| MoE Graph 掉点 | 05 的 A5 MoE Graph 精度问题 |
| W8A8 缺 `v_proj` | 04 的 K=V 权重加载 |
| W8A8 能启动但精度异常 | 04 的 GELU-TANH 与专家路径 |
| MTP 是否有性能收益 | 05 的 MTP 收益分析 |

## 6. 当前适配结果

### 6.1 支持情况

| 能力 | 31B Dense | 26B-A4B MoE | E2B | E4B |
|---|---|---|---|---|
| A2/A3 Eager | 已支持 | 已支持 | 基础路径已支持 | 基础路径已支持 |
| A2/A3 ACLGraph | 已支持 | 已支持 | KV Sharing 已接入，需持续看护精度 | KV Sharing 已接入，需持续看护精度 |
| A5 Eager | 已支持 | 已支持 | 基础路径已支持 | 基础路径已支持 |
| A5 ACLGraph | 已支持 | 可使用；MC2 组合路径仍需专项验证 | Graph 精度待完整闭环 | Graph 精度待完整闭环 |
| ModelSlim W8A8 | 已适配 | 已适配 | 待 checkpoint 级验证 | 待 checkpoint 级验证 |
| MTP | 已有阶段性实现和 acceptance 数据 | 已有阶段性实现和 acceptance 数据 | 待专项验证 | 待专项验证 |
| Nightly GPQA-D | 已覆盖 | 已覆盖 | 尚未建立同等级基线 | 尚未建立同等级基线 |

### 6.2 关键精度结果

| 场景 | 问题结果 | 修复或控制组 | 说明 |
|---|---:|---:|---|
| A5 26B MoE Graph | 约 57% | Graph + ALLGATHER 约 73% | 问题集中在 MC2 Graph 组合路径 |
| A2 FIA Decode Mask | 46.97% | 恢复旧语义后 72.22% | Mask/sparse-mode 是 PR 级回归原因 |
| A2 Loop Rate | 27.3% | 2.5% | 重复和 max-token 触顶明显恢复 |

![Gemma4 精度恢复](assets/accuracy_recovery.png)

### 6.3 关键性能结果

以下数据来自 Ascend 910B3、vLLM 0.23.0、TP1、PIECEWISE 和对应 W8A8 checkpoint：

| 模型/权重 | 文本 TPOT | 图片 TPOT | Decode Rate |
|---|---:|---:|---:|
| 31B Dense W8A8 | 37.1 ms | 36.0 ms | 26.93–27.77 tok/s |
| 26B MoE Only-Experts W8A8 | 22.2 ms | 22.9 ms | 43.76–44.94 tok/s |

![W8A8 性能基线](assets/w8a8_performance_baseline.png)

这些结果只能在对应设备、权重、并行和 workload 下比较，不能简单外推到其他设备或高并发 EP 场景。

## 7. 设计原则与当前边界

本次适配沉淀了以下原则：

1. 模型语义决定合法的后端优化边界。
2. 设备差异尽量放在 DeviceAdaptor 和设备工具层。
3. Capture backend、Replay backend 和 Graph Param 类型必须一致。
4. 当前执行 layer、metadata owner 和 KV producer 必须明确区分。
5. Workspace 必须覆盖同一 bucket 内所有交错层需求。
6. 量化只能改变表示和 kernel，不能改变激活函数或权重语义。
7. Smoke 用于快速发现错误，全量精度用于确认结果，性能需要固定 workload。

目前仍需继续完善的部分包括：

- A5 MoE MC2 Graph 内部动态字段的算子级根因；
- E2B/E4B Graph 的完整精度基线；
- MTP 在不同设备、不同 `k` 下的端到端收益；
- 更多量化 checkpoint、TP/EP 和长上下文组合。

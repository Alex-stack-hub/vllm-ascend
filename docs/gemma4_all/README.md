# Gemma4 on Ascend 适配总览

> 适配设备：Atlas A2、Atlas A3、Ascend 950（A5）
>
> 模型范围：Gemma4 31B Dense、26B-A4B MoE、E2B、E4B
>
> 内容范围：模型结构、Attention、ACLGraph、MoE、KV Cache、量化、运行时特性、精度、性能与 MTP

## 1. 背景：为什么要做 Gemma4 on Ascend

Gemma4 是 Google 推出的新一代开放模型系列，覆盖 Dense、MoE 和多模态形态。它既能处理文本，也面向图像、音频等输入，并具备推理、指令遵循、结构化输出和工具调用能力。对推理系统而言，Gemma4 的价值不仅在于单个模型分数，更在于它把多种有代表性的模型结构放在了同一个系列里：

- 31B Dense 适合验证通用 Attention、长上下文和稳定的计算密集型负载；
- 26B-A4B MoE 以较低的单 Token 激活参数量换取更大的模型容量，适合研究吞吐、专家并行和通信优化；
- E2B/E4B 引入 PLE、YOCO 和跨层 KV Sharing，代表了通过复用状态降低推理成本的新方向；
- 多模态能力使同一服务可以覆盖知识问答、文档与图片理解、内容分析、智能助手等应用。

把 Gemma4 适配到 [昇腾](https://www.hiascend.com/) 平台，首先是为了提供一条可部署、可验证、可持续维护的开放模型推理路径。政务与公共服务、医疗、教育、金融、能源、交通、制造等国计民生和行业项目，通常同时关心数据本地化、软硬件可控、长周期运维以及国产算力上的性能。只有模型能够稳定运行、精度有基线、性能可解释、CI 能持续看护，这些场景才具备从实验走向业务的基础。

另一方面，vLLM 已经形成了广泛使用的服务接口和开发者习惯。Gemma4 在 vLLM Ascend 上获得主干级支持后，国内外开发者可以沿用熟悉的 OpenAI-compatible API、并行配置和评测工具验证昇腾设备。这会降低模型和应用迁移成本，也为昇腾生态面向全球开发者和海外项目提供更容易理解、复现和接入的技术入口。

本次工作的对外使用入口已经沉淀到 [vLLM Ascend Gemma4 官方部署教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Gemma4.html)。这套文档在官方教程之外，进一步记录了模型结构、设备差异、问题定位和方案设计，帮助后续适配者理解“为什么这样实现”。

## 2. 这套文档做了什么

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

## 3. 为什么需要这些适配

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

## 4. 我们遇到了哪些主要问题

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

## 5. 我们如何组织适配工作

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

## 6. 相关 PR

下面按能力域列出已经合入主干、并直接参与 Gemma4 适配或持续看护的 PR。它们共同完成了设备执行、Graph、量化、KV Sharing、文档和 CI 闭环。

| PR | 主要内容 | 与本适配的关系 |
|---|---|---|
| [#10643](https://github.com/vllm-project/vllm-ascend/pull/10643) | A5 Gemma4 Graph 支持 | 引入 Layer-Aware Replay、FIA 最大 Workspace 和 GELU 相关适配 |
| [#11091](https://github.com/vllm-project/vllm-ascend/pull/11091) | A2/A3 512-Head 支持 | Global Decode 使用 PA，Prefill 使用 Large-Head Fallback，并将设备差异放入 DeviceAdaptor |
| [#11055](https://github.com/vllm-project/vllm-ascend/pull/11055) | Gemma4 官方部署教程 | 增加部署、功能验证、精度、性能、调优和 FAQ，并更新支持矩阵 |
| [#11536](https://github.com/vllm-project/vllm-ascend/pull/11536) | A2/A3 Mixed PA/FIA Graph | 使用显式 PA Graph Param，解决 9/21 参数错配，并处理 Large-Head RoPE UB |
| [#11575](https://github.com/vllm-project/vllm-ascend/pull/11575) | Gemma4 量化适配 | 处理 Global K=V 缺少 `v_proj` 和 MoE Expert Prefix |
| [#11266](https://github.com/vllm-project/vllm-ascend/pull/11266) | FIA Decode Mask 行为调整 | 后续 Nightly 精度回归的引入点，用于建立问题时间线和修复边界 |
| [#11732](https://github.com/vllm-project/vllm-ascend/pull/11732) | 量化 MoE 激活语义修复 | 让量化 Expert 使用模型配置的 GELU，而不是固定走 SwiGLU |
| [#11791](https://github.com/vllm-project/vllm-ascend/pull/11791) | E2B/E4B KV Sharing | 保证 Target Layer 读取 Producer KV，同时跳过 Cache Write |
| [#11798](https://github.com/vllm-project/vllm-ascend/pull/11798) | 文档国际化同步 | 将包含 Gemma4 教程在内的模型文档同步到多语言文档体系 |
| [#11856](https://github.com/vllm-project/vllm-ascend/pull/11856) | Gemma4 Nightly | 增加 31B Dense、26B MoE 的 Eager、ACLGraph 和 GPQA-D 持续看护 |
| [#11899](https://github.com/vllm-project/vllm-ascend/pull/11899) | PA 路径重构 | 曾移除 PA；其影响帮助确认 A2/A3 512-Head Decode 仍依赖 PA Fallback |
| [#12228](https://github.com/vllm-project/vllm-ascend/pull/12228) | 恢复 PA 相关能力 | 保证 A2/A3 512-Head Decode Fallback 路径仍然可达 |
| [#12391](https://github.com/vllm-project/vllm-ascend/pull/12391) | Gemma4 MoE PR E2E | 增加 A3 双卡、TP2+EP、FULL_DECODE_ONLY 的启动与短生成看护 |
| [#12605](https://github.com/vllm-project/vllm-ascend/pull/12605) | FIA Decode Mask 精度恢复 | 恢复回归前的 Mask/Sparse-Mode 语义，使 A2 MoE GPQA-D 和重复率恢复 |

另外两条尚未合入、但对早期设计分析有参考价值的 PR 是 [#9222](https://github.com/vllm-project/vllm-ascend/pull/9222) 和 [#11399](https://github.com/vllm-project/vllm-ascend/pull/11399)。前者覆盖早期 Attention、MoE、RoPE、Runner 和 KV Sharing 方案，后者探索 Gemma4 MTP。阅读它们时需要结合所基于的主干版本；最终行为以当前主干和已合入 PR 为准。

量化权重导出位于 msModelSlim 仓库，对应的三条变更也属于完整适配的一部分：

| MR | 主要内容 |
|---|---|
| [msModelSlim !676](https://gitcode.com/Ascend/msmodelslim/pull/676) | 增加 26B-A4B MoE W8A8，拆分 3D 专家权重并逐专家导出 |
| [msModelSlim !718](https://gitcode.com/Ascend/msmodelslim/pull/718/diffs) | 增加 31B Dense Adapter、逐层量化和 tied `lm_head` 导出处理 |
| [msModelSlim !719](https://gitcode.com/Ascend/msmodelslim/pull/719/diffs) | 增加 Dense 一键量化配置、支持矩阵和用户文档 |

## 7. 报告与对外资料

| 资料 | 内容 |
|---|---|
| [vLLM Ascend Gemma4 官方部署教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Gemma4.html) | 面向使用者的安装、部署、精度、性能和调优入口 |
| [特性验证报告总览](https://github.com/0moyi0-2024/vllm-ascend_tp/blob/gemma4_performence_glm52/gemma4_perf_artifacts/README.md) | 汇总特性验证、Graph 模式和 W8A8 性能产物 |
| [Eager、PIECEWISE 与 FULL_DECODE_ONLY 综合报告](https://github.com/0moyi0-2024/vllm-ascend_tp/blob/gemma4_performence_glm52/gemma4_perf_artifacts/feature_graphmode_fdo_20260723/reports/gemma4_feature_graphmode_consolidated.md) | 31B Dense 与 26B MoE、5 组运行时特性、42 个组合用例 |
| [FULL_DECODE_ONLY 特性报告](https://github.com/0moyi0-2024/vllm-ascend_tp/blob/gemma4_performence_glm52/gemma4_perf_artifacts/feature_graphmode_fdo_20260723/reports/feature_graphmode_fdo_report.md) | TP4 Graph Capture、服务请求和特性组合验证 |
| [PIECEWISE 特性报告](https://github.com/0moyi0-2024/vllm-ascend_tp/blob/gemma4_performence_glm52/gemma4_perf_artifacts/feature_graphmode_piecewise_20260723/reports/feature_graphmode_piecewise_report.md) | TP2 Piecewise Graph 的功能与轻量性能观测 |
| [31B Dense W8A8 性能报告](https://github.com/0moyi0-2024/vllm-ascend_tp/blob/gemma4_performence_glm52/gemma4_perf_artifacts/dense_w8a8_baseline_20260713/reports/quant_tp1_piecewise_baseline.md) | Dense 量化文本/图片基线 |
| [26B MoE W8A8 性能报告](https://github.com/0moyi0-2024/vllm-ascend_tp/blob/gemma4_performence_glm52/gemma4_perf_artifacts/moe_w8a8_baseline_20260713/reports/quant_moe_tp1_piecewise_baseline.md) | MoE Only-Experts 量化文本/图片基线 |

综合特性报告记录的测试版本中，31B Dense 和 26B MoE 分别在 Eager TP2、PIECEWISE TP2、FULL_DECODE_ONLY TP4 下完成 7/7 场景验证，共 42 个组合全部拉起并完成请求。报告中的结论受设备、版本、TP 和测试输入约束，应作为可复现证据使用，而不是脱离环境外推为所有配置的性能承诺。

## 8. 详细文档

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

## 9. 当前适配结果

### 9.1 支持情况

| 能力 | 31B Dense | 26B-A4B MoE | E2B | E4B |
|---|---|---|---|---|
| A2/A3 Eager | 已支持 | 已支持 | 基础路径已支持 | 基础路径已支持 |
| A2/A3 ACLGraph | 已支持 | 已支持 | KV Sharing 已接入，需持续看护精度 | KV Sharing 已接入，需持续看护精度 |
| A5 Eager | 已支持 | 已支持 | 基础路径已支持 | 基础路径已支持 |
| A5 ACLGraph | 已支持 | 可使用；MC2 组合路径仍需专项验证 | Graph 精度待完整闭环 | Graph 精度待完整闭环 |
| ModelSlim W8A8 | 已适配 | 已适配 | 待 checkpoint 级验证 | 待 checkpoint 级验证 |
| MTP | 已有阶段性实现和 acceptance 数据 | 已有阶段性实现和 acceptance 数据 | 待专项验证 | 待专项验证 |
| Nightly GPQA-D | 已覆盖 | 已覆盖 | 尚未建立同等级基线 | 尚未建立同等级基线 |

### 9.2 关键精度结果

| 场景 | 问题结果 | 修复或控制组 | 说明 |
|---|---:|---:|---|
| A5 26B MoE Graph | 约 57% | Graph + ALLGATHER 约 73% | 问题集中在 MC2 Graph 组合路径 |
| A2 FIA Decode Mask | 46.97% | 恢复旧语义后 72.22% | Mask/sparse-mode 是 PR 级回归原因 |
| A2 Loop Rate | 27.3% | 2.5% | 重复和 max-token 触顶明显恢复 |

![Gemma4 精度恢复](assets/accuracy_recovery.png)

### 9.3 关键性能结果

以下数据来自 Ascend 910B3、vLLM 0.23.0、TP1、PIECEWISE 和对应 W8A8 checkpoint：

| 模型/权重 | 文本 TPOT | 图片 TPOT | Decode Rate |
|---|---:|---:|---:|
| 31B Dense W8A8 | 37.1 ms | 36.0 ms | 26.93–27.77 tok/s |
| 26B MoE Only-Experts W8A8 | 22.2 ms | 22.9 ms | 43.76–44.94 tok/s |

![W8A8 性能基线](assets/w8a8_performance_baseline.png)

这些结果只能在对应设备、权重、并行和 workload 下比较，不能简单外推到其他设备或高并发 EP 场景。

## 10. 设计原则与当前边界

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

# Gemma 4 31B + EAGLE3 Ascend 适配状态（供 Codex Review）

**分支**: `gemma4-eagle3` | **HEAD**: `7fccd8f6`
**日期**: 2026-06-30 | **vLLM 版本**: v0.20.2

---

## 当前代码改动

### 1. Gemma4 image_token_id 映射 (e32d814d)
`llm_base_proposer.py` +2行: `image_token_id` -> `image_token_index`

### 2. ACL graph nested output flatten (524fa0df, 7fccd8f6)
**根因**: NPU graph capture (`torch.npu.graph`) 不处理嵌套输出。
Gemma4 返回 `(hidden_states, [aux0, aux1, aux2])` 时，list 元素不被 NPU graph 注册为输出，replay 后为 stale 值。

**修复**:
- `acl_graph.py`: capture 时 flatten `(hs, [aux0,aux1,aux2])` -> `(hs, aux0, aux1, aux2)`
- `model_runner_v1.py`: 解包兼容 `len==2` (eager) 和 `len>2` (graph flat)
- `utils.py`: `weak_ref_tensors` 递归 list/tuple

### 3. Draft graph padding tail zero-fill (08c764b5, 8d135b58)
防御性补丁，单独未能修复 graph 乱码。

---

## 验证矩阵

| 配置 | commit | 状态 | 备注 |
|------|--------|------|------|
| Target eager | e32d814d | ✅ | |
| Target graph | e32d814d | ✅ | |
| EAGLE3 eager | e32d814d | ✅ | greedy token 100% 一致 |
| EAGLE3 graph (T-graph + D-eager) | 7fccd8f6 | 🔄 待验证 | 需 NPU 环境 |

---

## 待验证命令

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# EAGLE3 graph (target FULL_DECODE_ONLY + draft eager)
python3 -c "
from vllm import LLM,SamplingParams
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('/home/xty/gemma4/31B')
p=t.apply_chat_template([{'role':'user','content':'Explain AI briefly.'}],tokenize=False,add_generation_prompt=True)
llm=LLM(model='/home/xty/gemma4/31B',tensor_parallel_size=4,max_model_len=512,gpu_memory_utilization=0.6,max_num_seqs=1,
        compilation_config={'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[4]},
        speculative_config={'method':'eagle3','model':'/home/xty/eagle3/gemma-4-31b-it-eagle3','num_speculative_tokens':3,'draft_tensor_parallel_size':1,'enforce_eager':True})
sp=SamplingParams(temperature=0,max_tokens=32)
out=llm.generate([p],sp)
expected=[3834,1061,33055,236764,5213,118870,28243,568,12553,62902,563,506,5596,529,496,5194,653,5464,531,62611,3246,14020,531,2121,9395,532,8974,4078,236761,108,40725,529]
print(f'MATCH:{\"PASS\" if out[0].outputs[0].token_ids==expected else \"FAIL\"}')
print(f'TEXT:{out[0].outputs[0].text[:80]!r}')
del llm
"
```

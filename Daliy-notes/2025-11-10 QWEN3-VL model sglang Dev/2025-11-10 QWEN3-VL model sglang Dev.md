### MHA backend for vision block attn
slgang中的VisionAttention没有支持aiter的attn backend，默认运行会报错，针对这个修复已经提交了，且merged到了主线了
[[[Enable Aiter Attention for VL model by Yuechguo · Pull Request #12699 · sgl-project/sglang](https://github.com/sgl-project/sglang/pull/12699)]]
![](./Pasted%20image%2020251110100655.png)

### fused_moe layer issue in EP mode
EP mode下，fused_moe layer加载模型的checkpoint的时候，针对非local rank的expert id的checkpoint处理有问题，需要修复
![](./Pasted%20image%2020251110101130.png)

### Padding issue for PTPC aiter op
![](./Pasted%20image%2020251110102027.png)

### weight load as ptpc 
EP mode 下，Qwen3-VL-235B-A22B-Instruct 模型的hf checkpoint 模式的 moe权重部分的expert的gate weight和up_proj weight是融合在一起的， 但是Qwen3-VL-235B-A22B-Instruct-FP8-dynamic（PTPC量化）模型的hf checkpoint 模式的 moe权重部分是分离的。导致这连个模型加载的时候走的path是不一样，然后两者有区别，需要统一修改
修改的PR ： [[BugFix] weight load bug when checkpoint expert.gate and exepert.up_proj are not fused by Yuechguo · Pull Request #13113 · sgl-project/sglang](https://github.com/sgl-project/sglang/pull/13113)

### PTPC量化的moe 算子layer
sglang中的PTPC量化的fused_moe path 会走到CompressedTensorsW8A8Fp8MoEMethod，改方法的实现有几个问题：
1.QuantizationStrategy.CHANNEL模式下且启用aiter的话，才需要shuffle_weight,
2.moe_runner_config中的apply_router_weight_on_input应该always 是false的，apply_router_weight_on_input通常会在topk 提前实现
3.rocm_fused_experts_tkw1 只支持topk == 1

PR ：[[BugFix] Accuracy and function Issue when run ptpc quant model by Yuechguo · Pull Request #13157 · sgl-project/sglang](https://github.com/sgl-project/sglang/pull/13157)

Llama-4-Maverick-17B-128E-Instruct-FP8
启动命令：
```
SGLANG_USE_AITER=1 \
USE_ROCM_AITER_ROPE_BACKEND=0 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m sglang.launch_server \
        --model-path /pretrained_model/meta/Llama-4-Maverick-17B-128E-Instruct-FP8 \
        --served-model-name  Llama-4-Maverick-17B-128E-Instruct-FP8 \
        --attention-backend aiter \
        --host 127.0.0.1 \
        --port 28000 \
        --tp-size 4 --ep-size 4 --trust-remote-code \
        --chunked-prefill-size 32768 \
        --max-prefill-tokens 32768 \
        --context-len 32768 \
        --max-running-requests 512 \
        --mem-fraction-static 0.85 \
```
精度测试：
```
python3 /data/testhome/sglang/benchmark/gsm8k/bench_sglang.py --num-shots 5 --num-questions 100 --port 28000

Accuracy: 0.950
Invalid: 0.000
Latency: 10.075 s
Output throughput: 962.065 token/s
```

Qwen3-VL-235B-A22B-Instruct-FP8-dynamic
启动命令：
```
SGLANG_USE_AITER=1 \
USE_ROCM_AITER_ROPE_BACKEND=0 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
SGLANG_VLM_CACHE_SIZE_MB=4096 \
python3 -m sglang.launch_server \
        --model-path /pretrained_model/qwen/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/ \
        --served-model-name  Qwen3-VL-235B-A22B-Instruct-FP8-dynamic \
        --attention-backend aiter \
        --host 127.0.0.1 \
        --port 28000 \
        --tp-size 4 --ep-size 4 --trust-remote-code \
        --chunked-prefill-size 32768 \
        --max-prefill-tokens 32768 \
        --context-len 32768 \
        --max-running-requests 512 \
        --mem-fraction-static 0.85 \
        --mm-attention-backend aiter_attn \
```
文本精度测试：
```
python3 /data/testhome/sglang/benchmark/gsm8k/bench_sglang.py --num-shots 5 --num-questions 100 --port 28000

Accuracy: 0.980
Invalid: 0.000
Latency: 22.778 s
Output throughput: 606.273 token/s
```

多模态精度测试：
```
python3 /data/testhome/sglang/benchmark/mmmu/bench_sglang.py --port 28000 --concurrency 1

{'Accounting': {'acc': 0.533, 'num': 30},
 'Agriculture': {'acc': 0.567, 'num': 30},
 'Architecture_and_Engineering': {'acc': 0.3, 'num': 30},
 'Art': {'acc': 0.767, 'num': 30},
 'Art_Theory': {'acc': 0.933, 'num': 30},
 'Basic_Medical_Science': {'acc': 0.733, 'num': 30},
 'Biology': {'acc': 0.6, 'num': 30},
 'Chemistry': {'acc': 0.6, 'num': 30},
 'Clinical_Medicine': {'acc': 0.8, 'num': 30},
 'Computer_Science': {'acc': 0.733, 'num': 30},
 'Design': {'acc': 0.9, 'num': 30},
 'Diagnostics_and_Laboratory_Medicine': {'acc': 0.433, 'num': 30},
 'Economics': {'acc': 0.733, 'num': 30},
 'Electronics': {'acc': 0.2, 'num': 30},
 'Energy_and_Power': {'acc': 0.4, 'num': 30},
 'Finance': {'acc': 0.433, 'num': 30},
 'Geography': {'acc': 0.567, 'num': 30},
 'History': {'acc': 0.8, 'num': 30},
 'Literature': {'acc': 0.933, 'num': 30},
 'Manage': {'acc': 0.567, 'num': 30},
 'Marketing': {'acc': 0.6, 'num': 30},
 'Materials': {'acc': 0.333, 'num': 30},
 'Math': {'acc': 0.4, 'num': 30},
 'Mechanical_Engineering': {'acc': 0.367, 'num': 30},
 'Music': {'acc': 0.433, 'num': 30},
 'Overall': {'acc': 0.61, 'num': 900},
 'Overall-Art and Design': {'acc': 0.758, 'num': 120},
 'Overall-Business': {'acc': 0.573, 'num': 150},
 'Overall-Health and Medicine': {'acc': 0.7, 'num': 150},
 'Overall-Humanities and Social Science': {'acc': 0.792, 'num': 120},
 'Overall-Science': {'acc': 0.567, 'num': 150},
 'Overall-Tech and Engineering': {'acc': 0.414, 'num': 210},
 'Pharmacy': {'acc': 0.767, 'num': 30},
 'Physics': {'acc': 0.667, 'num': 30},
 'Psychology': {'acc': 0.733, 'num': 30},
 'Public_Health': {'acc': 0.767, 'num': 30},
 'Sociology': {'acc': 0.7, 'num': 30}}
eval out saved to ./val_sglang.json
Overall accuracy: 0.61
```
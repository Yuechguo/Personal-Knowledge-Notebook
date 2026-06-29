### source code
sglang社区主线已经按照upstream的方式集成了aiter的基础算子

### AITER enable in sglang main
```
Aiter-Attn ： MHA/MLA，max_dim_size=256, MLA max_heads=128, enable attention_backend='aiter' 

Aiter rms_norm/fused_add_rms_norm : enable SGLANG_USE_AITER = 1

Aiter Moe(asm_moe/ck_moe_2stages) : 

Aiter rotary_embedding(rope) :  enable SGLANG_USE_AITER=1 only in Deepseek model

Aiter fp8 ， 默认block quant，gemm_a8w8_blockscale_CK， Activation，ck_moe_2stages， shuffle_w
eight ：enable SGLANG_USE_AITER=1

```

### Enhanced feature integration
```
upstream fp8-PTPC ：接入aiter的fp8-PTPC的功能

DEEPEP ：
1.rocm-shmen 已经具备了，但是random hang（16卡两node），暂时无进展。
2.hipfy nv-shmem 在 BF3L网卡所有mode运行都OK，IBRC/IBGDA, 对应性能只有 DS3 1/3 H20， QWEN3 1/2 H20。
3.阿里自研网卡EIC 所有的mode都不行, 高通cx7网卡只在normal mode可以（IBRC）。

TBO ：SGLANG框架具备，DEEPEP 具备后，就OK

EPLB：
1.sglang主线具备基础功能，适配AMD需要做修改，依赖 EP-MOE并行，sglang main启动碰到EP的问题https://github.com/sgl-project/sglang/issues/7055
2.内部集成版本实际测试效果没有明显的uplift

DeepGEMM：目前是Aitrer GEEM算子替代
```

![](./Pasted%20image%2020250618160850.png)

### offline 量化的模型的精度（from niuxinjun team）
![](./Pasted%20image%2020250619145220.png)

### build mooncake-transfer-engine
```
apt install libgflags-dev libgoogle-glog-dev libjsoncpp-dev
```

### build weight
```
# with shuffle weight the CK kernel do not need to cache weight on LDS
# and get better performance
weight_shuffled = shuffle_weight(weight.t(), layout=(16,16))

from sglang.srt.utils import (
    get_bool_env_var,
    is_hip,
)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_hip:
    from aiter.ops.shuffle import shuffle_weight
```
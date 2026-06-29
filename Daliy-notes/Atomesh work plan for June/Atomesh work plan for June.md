### Overall Objective for June
1. **Benchmarking DeepSeek-V4-Pro on InferenceX**
2. **Atomesh code base functionality is fully implemented and production-ready for public release.**
### Task breakdown
#### Model related tasks
##### Non-MTP issues
1.**DeepSeek-V4-Pro performance alignment with NVIDIA-vLLM under low-concurrency, small-scale (1P1D) conditions. -- P0** 
ETA 21/06/2026, Depends on 2 TW machines and 2 B200 machines(1P1D enable).
@guanbao for B200 trace
@wanzhen MI355 trace with 1 node conc=1 for 8192/1, 1/1024, breakdown, performance alignment, feature requirements for ATOM
@qianyun @xinjian for performance alignment

2.**EP(ep with dispatch and combine in mori-EP bf16 -> MXFP4 ) enable and TBO enable  with performance -- P0**
ATOM EP OK, TBO OK
ETA 28/06/2026,  Depends on 4 TW machines
@lilong @jiaoliang
@yuechguo interface for process track
@zhangling EP and DPA performance result

3.**Sglang-ATOM for DeepSeek-V4-Pro -- P2**
ETA 28/06/2026
@yuhua  enable functional E2E
@zhiwei for DSV4 attn alignment

##### MTP issues
1.**DeepSeek-V4-Pro MTP enable with TP/DPA/EP mode -- P1**
ATOM DPA TP MTP is OK.
ETA 28/06/2026,  Depends on 2 TW machines
@yajie, some step: breakdown for this weekend, long time for performance result.

3.**Sglang-ATOM MTP support for DeepSeek-V4-Pro -- P2**
pause for june.

#### Atomesh core development
##### Code base develop and validation
 1.**unified CI/CD for atomesh -- P0**
 Accuracy validation and performance validation for atomesh-atom standalone. 
 ETA 09/06/2026 @wanzhen
 Mocker CI test for atomesh itself performance.
 ETA 10/06/2026 @wanzhen
 Daily or regular benchmark runs for xPyD
 Ensure at least 32/48 dedicated cards are available every weekend for benchmark tasks.
 @yajie 
 
2.**worker pool update -- P1**
ETA 14/06/2026 
@yuechguo

3.**Policy validation on Atomesh and Implementation with mocker test -- P2**
prefix hash and cache aware routing.
benchmark for performance uplift validation
inferenceX agentic benchmark validation

4.**refactor on policy implementation with new worker_pool storage and new smart routers -- P2**
ETA 28/06/2026 
@yuechguo

##### Joint development with MORI-UMP or LMCache in ATOM
1.**Atomesh and mori-UMP co-design for kv-cache aware routing scheme -- P2**
Design of store index synchronization method between atomesh and mori-UMP - P1
routing policy implementation with unified kv-cache index
performance validation for kv-cache aware routing policy

2.**Atomesh and ATOM-LMCache for kv-cache aware routing. -- P1**
LMcache enable in ATOM
Atomesh support kv-cache index get/put from LMCache
Atomesh enable kv-cache routing policy

3.**Atomesh and Mooncake-kv cache pool for kv aware routing.**
@yuechguo tech research @ 16/06/2026
## Background
Alibaba ads online system will trigger model switch every 10mins or 15mins. what happens when model switch:
1.Tensorflow create a new session and do warmup, there is many different input shape in TF warmup which will trigger xla re-complie. The shape size was pre-load by config file which is collected though experience of Alibaba engineer
2.when the new session is ready, and the old session has done handing of the requests in the task queue. The service process immediately switch the task queue to new session and destroy the old session.

Bad thing happened:
The serving model in online system suffers crash or soft-hang, or something latency explosion. Some model like AIDA may suffer 30 times crash in every 24 hours for 140 GPUs.

## Re-produce in AMD HJBOG env
In AMD env, we can partly reproduce by Ads benchmark.
``` shell
bench_model_config {
   name: "ad_rtp_target_prerank_predict_uni_all"
   frozen_graph: "ad_rtp_target_prerank_predict_uni_all_split_parted_search_new.pbtxt"
   runmeta: "BlazeXlaOp_runmeta"
   meta_graph: "meta_graph"
   config_proto: "config_proto"
   run_options: "run_options"
   predictor_num: 16
   qps: -1
   #batch_size: 2
   switch_interval: 120 ->> // set 120 seconds for every model switch time
   dense_input_to_add:"RapidEmbeddingReduceOpV2"
}
bench_thread_count: 16
duration: 28800000
#max_queue_size: 80
```
Modify `switch_interval` to a small value in order to trigger model switch more frequently.

### Result
In observation of the running situation for a period of time.
`1.Latency explosion`
![](./Pasted%20image%2020250220103055.png)
![](./Pasted%20image%2020250220103104.png)
latency P99 and P999 get bigger and performance drops.

`2.Abnormal memory usage`
![](./Pasted%20image%2020250220103132.png)
![](./Pasted%20image%2020250220103141.png)
![](./Pasted%20image%2020250220143648.png)

### XLA autotuning
set xla autotuning flag：
``` shell
XLA_FLAGS="--xla_gpu_autotune_level=0"

0 : mean close autotune
```

### Test in hipblaslt 
with hipblaslt log :
``` shell
HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=./hipblaslt_%i.log
```
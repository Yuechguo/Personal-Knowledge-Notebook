## Prefix kv-cache routing without kv-cache sync between atomesh and workers
![](./Pasted%20image%2020260624230707.png)
We don't need to care about whether it's a PD worker, nor do we need to worry about the specific implementation of KV-cache transfer between p worker and d worker.

cache aware/affinity routing
- Routes by prefix affinity: It matches the incoming `request_text` against historical request prefixes and prefers the worker associated with the longest / strongest prefix match.
- Does not query the real KV cache: It uses an approximate radix tree built from request history to infer which worker is likely to have useful cache.
- Maintains one tree per model: Requests are grouped by `model_id`, so cache affinity is tracked separately for each model.
- Updates continuously after routing: After a worker is selected, the policy inserts `request_text` into the tree under that worker URL.
- Falls back to load balancing when needed: If worker load is imbalanced, it prioritizes the least-loaded worker over cache affinity.
- Still updates the tree during load balancing: Even when routing by load, the selected worker is recorded in the cache tree for future requests.
- Uses eviction to bound memory growth: Old leaf nodes can be evicted with an approximate LRU strategy.
- Works best for repeated or similar prompts: Shared prefixes increase the chance of routing related requests to the same worker, improving `potential KV cache reuse`.

![](./Pasted%20image%2020260624230908.png)

init status:
```
model: ds-v4-pro
Tree
└── root
    tenants: { w1, w2, w3 }
    children: empty

first request with text : "hello world"
root
└── "hello world"   tenants: { w1 }

first request with text : "hello rust"
root
└── "hello"        { w1 }
    ├── "world"    { w1 }
    └── "rust"     { w1 }

request with text : "hi there"
root
├── "hello"        { w1 }
│   ├── "world"    { w1 }
│   └── "rust"     { w1 }
└── "hi there"     { w2 }

request with text : "hello world again"
root
├── "hello "              { w1 }
│   ├── "world"           { w1 }
│   │   └── " again"      { w1 }
│   └── "rust"            { w1 }
└── "hi there"            { w2 }

request with text : "hello gpu", request routing with min load balance w2
root
├── "hello "              { w1, w2 }
│   ├── "world"           { w1 }
│   │   └── " again"      { w1 }
│   ├── "rust"            { w1 }
│   └── "gpu"             { w2 }
└── "hi there"            { w2 }
```

### Task list
1.atomesh ci (P0, wanzhen)
2.aotmesh + atom MTP/Eagle support (P0 yajie)
3.atomesh + atom + lmcache（standalone mode） enablement and testing （P0，code ready, xingjian）
4.benchmark get from SA agentic or we design and develop one？（P0, yuechguo)
5.atom + lmcache with n+1 layer kv-cache preload like sglang Hicache or per-forward pre-load（P0）
6.atomesh metric support request prefix cache match stats（P1, wanzhen）
7.atomesh optimal tree search for prefix cache aware routing（p1, yuechguo）
8.atom modeling perf optimal, DSV4-PRO, kimi, minmax M3. (zhangling, qianyun)
9.atom support for PD KV-cache transfer under heterogeneous TP/DPA, for optimal performance practices.（P1）

## Prefix kv-cache routing with kv-cache sync for best policy 
### LMCache p2p sharing kv pool
Core understanding:
- The Coordinator is only responsible for node discovery, registration, and heartbeats. It does not transfer KV data.
- Each vLLM instance only connects to the local LMCache Server.
- Each LMCache Server creates a `P2PL2Adapter` for every other live peer, so the topology is close to a full mesh.
- During a local L1 cache miss, the lookup control plane sends `lookup-and-lock` requests to all live peer adapters.
- The lookup control plane uses LMCache MessageQueue RPC, implemented with ZMQ and msgpack.
- The peer that receives a lookup request only checks its local L1 cache and locks matched objects; it does not recursively query other peers.
- Lookup responses return metadata such as hit bitmap and remote L1 `offset/size`, not KV tensors.
- The actual KV data path uses NIXL / RDMA read. The requesting LMCache Server pulls KV data directly from the remote peer’s L1 memory into its own local L1 memory.
- KV data is not transferred through the Coordinator or through the ZMQ RPC channel.
- Although lookup is sent to multiple peers, the actual KV load is only performed from the peer selected by the prefetch policy, usually the first matching adapter by index.
- After the RDMA read finishes, the requester sends an unlock RPC so the remote peer can release the read locks.
- This design keeps metadata/control traffic separate from high-throughput KV data transfer.
![](./Pasted%20image%2020260625001247.png)
In PD mode, the KV-pool does not support P2P sharing

Atomesh no need to care about kv-cache，just routing out request in load balance policy.
**As the number of peers increases, the number of P2P connections grows exponentially.**
**The routing is not optimal, resulting in unnecessary KV migration overhead.**

### Mooncake store
Mooncake Store has two main roles: Master and Client.

Master is the control-plane service. It manages cluster metadata, tracks registered clients and their storage segments, allocates space for new objects, records where each object replica is located, handles leases, eviction, quota, replication tasks, offload/promotion, and client liveness. The master does not transfer object data itself. It only tells clients where data should be written or read.

Client is both a data-plane participant and an API endpoint for applications. A client can expose local memory, SSD, or other storage as a segment in the distributed cache pool. It also issues `put`, `get`, `remove`, and batch operations on behalf of the application. During reads and writes, clients ask the master for metadata, then transfer data directly to or from other clients through the Transfer Engine.
```
Master:
  manages metadata, allocation, replicas, leases, eviction, and liveness

Client:
  contributes storage, serves data, and performs direct data transfer

put:
  client -> master PutStart
  master -> returns target replica locations
  client -> writes data directly to target client/storage
  client -> master PutEnd

get:
  client -> master GetReplicaList
  master -> returns available replicas
  client -> reads data directly from the selected replica
```
![](./Pasted%20image%2020260625130027.png)

### Atomesh with distributed kv store
####  No keep kv-cache index in Atomesh and support PD with LMCache
![](./Pasted%20image%2020260625004503.png)
ATOMesh replaces lmcache's coordinator role, and each lmcache server registers with ATOMesh.
ATOMesh acts as the master/peer role and maintains connections with each lmcache server to discover prefix-cache hits for prompts.
![](./Pasted%20image%2020260625122618.png)
In PD mode, a request is dispatched to a decode node only when the KV cache fully matches on that node.

atomesh routing step:
```
1.lookup all LMcache server for each prompt of requests
2.set prefix kv cache priority: 0-HBM, 1-RAM, 2-SSD, 3-in other node RAM
2.apply in routing policy
```

**mget hole problem**
**As the number of nodes increases, the network fan-out of each ATOMesh mget request also increases, and the latency from individual nodes leads to overall performance degradation.**
####  Keep kv-cache index in Atomesh and support PD with LMCache
Atomesh keep kv-cache by cache-aware policy but support more store level (L0-HBM, L1-RAM, L2-SSD), Initially, everything defaults to the HBM level.

Whenever an lmcache server performs HBM->RAM or RAM->SSD offload, it updates ATOMesh. ATOMesh then performs lazy updates(via P2P MQ) to the index levels in the tree structure.

When KV-cache is completely evicted, it also notifies(via P2P MQ) ATOMesh to perform eviction on the tree.

#### Minimum integration approach for Atomesh and Mooncake Store
**Atomesh does not need to concern itself with the specific distribution of KV indices. Before sending each request to a node, it first queries the Mooncake Conductor, and then determines the routing path based on the response.**

#### Atomesh Keep kv-cache index with Mooncake Store
**Atomesh maintains a global KV-cache index, replacing the Conductor's role in this regard. KV-cache index updates from all nodes are sent to Atomesh via event notifications, and Atomesh maintains a complete, unified KV-cache index.**

### Atomesh Out-of-order execution of requests
**Atomesh's out-of-order execution: within each worker's request queue, requests are not processed strictly in the order they arrive. Instead, for each request's multiple execution steps, the scheduler prioritizes requests whose steps do not require pre-loading or pre-transfer operations.**
![](./Pasted%20image%2020260625120254.png)

## Benchmark for kv-cache routing
### SA agentic benchmark
client side dataset:
[semianalysisai/cc-traces-weka-062126-256k · Datasets at Hugging Face](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k")
[semianalysisai/cc-traces-weka-062126 · Datasets at Hugging Face](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126 "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126")
client benchmark tool:
**AIperf**

### AIperf from nvidia ai-dynamo
[ai-dynamo/aiperf: AIPerf is a comprehensive benchmarking tool that measures the performance of generative AI models served by your preferred inference solution.](https://github.com/ai-dynamo/aiperf)
AIPerf is a comprehensive benchmarking tool that measures the performance of generative AI models served by your preferred inference solution. It provides detailed metrics using a command line display as well as extensive benchmark performance reports.

#### Workloads and Data
- [Trace Benchmarking](https://github.com/ai-dynamo/aiperf/blob/main/docs/benchmark-modes/trace-replay.md) - Deterministic workload replay
- [Bailian Traces](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/bailian-trace.md) - Bailian production trace replay
- [BurstGPT Traces](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/burst-gpt-trace.md) - BurstGPT real-world bursty traffic trace replay
- [SageMaker Data Capture](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/sagemaker-data-capture.md) - Replay production traffic from SageMaker endpoints
- [Custom Prompt Benchmarking](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/custom-prompt-benchmarking.md) - Send exact prompts as-is
- [Custom Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/custom-dataset.md) - Custom dataset formats
- [Inline Datasets](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/inline-datasets.md) - Embed records directly in the YAML config (single_turn, multi_turn, multi-pool random_pool, traces)
- [ShareGPT Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/sharegpt.md) - Profile with ShareGPT dataset
- [AIMO Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/aimo.md) - Profile with AIMO math reasoning datasets (NuminaMath-TIR, NuminaMath-CoT, NuminaMath-1.5, AIME)
- [MMStar Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/mmstar.md) - Profile vision language models with MMStar visual QA benchmark
- [MMVU Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/mmvu.md) - Profile video language models with MMVU expert-level video understanding benchmark
- [VisionArena Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/vision-arena.md) - Profile with real-world vision conversations from Chatbot Arena
- [LLaVA-OneVision Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/llava-onevision.md) - Profile with diverse multimodal instruction-following data
- [SPEED-Bench Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/speed-bench.md) - Profile speculative decoding with SPEED-Bench
- [InstructCoder Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/instruct-coder.md) - Profile with InstructCoder code generation dataset
- [SpecBench Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/spec-bench.md) - Profile with SpecBench speculative decoding dataset
- [Blazedit Dataset](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/blazedit.md) - Profile with Blazedit code editing dataset
- [ASR Datasets](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/asr.md) - Profile ASR models with LibriSpeech, VoxPopuli, GigaSpeech, AMI, and SPGISpeech
- [Synthetic Dataset Generation](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/synthetic-dataset.md) - Generate synthetic datasets
- [Agentic Code Generator](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/agentic-code-generator.md) - Generate multi-turn coding-agent traces for KV cache benchmarking
- [Fixed Schedule](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/fixed-schedule.md) - Precise timestamp-based execution
- [Time-based Benchmarking](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/time-based-benchmarking.md) - Duration-based testing
- [Sequence Distributions](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/sequence-distributions.md) - Mixed ISL/OSL pairings
- [Prefix Synthesis](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/prefix-synthesis.md) - Prefix data synthesis for KV cache testing
- [Reproducibility](https://github.com/ai-dynamo/aiperf/blob/main/docs/reproducibility.md) - Deterministic datasets with `--random-seed`
- [Template Endpoint](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/template-endpoint.md) - Custom Jinja2 request templates
- [Multi-Turn Conversations](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/multi-turn.md) - Multi-turn conversation benchmarking
- [Raw Payload Replay](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/raw-payload-replay.md) - Verbatim JSONL payload replay (single file or directory)
- [Inputs JSON Replay](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/inputs-json-replay.md) - Verbatim multi-turn replay of AIPerf inputs.json artifacts
- [Local Tokenizer](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/local-tokenizer.md) - Use local tokenizers without HuggingFace

### LMBenchmark from LMCache
[LMCache/LMBenchmark: Systematic and comprehensive benchmarks for LLM systems.](https://github.com/LMCache/LMBenchmark)
he workload simulated in these benchmarks is a multi-round QA (question answering) task with multiple users interacting with an LLM engine concurrently.
#### Available Bench
1. **ShareGPT Benchmark**
    - Replays real-world conversations from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
    - Default QPS: 1.34
2. **Short Input, Short Output (Synthetic)**
    - System prompt: 0 tokens
    - Chat history: 256 tokens
    - Answer length: 20 tokens
    - Default QPS: 15
3. **Long Input, Short Output (Synthetic)**
    - System prompt: 1000 tokens
    - Chat history: 20000 tokens
    - Answer length: 100 tokens
    - Default QPS: 0.1

### Mooncake traces
https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
he Open Source Trace dataset is a privacy-preserving LLM serving workload trace released by Mooncake. It is designed for simulating KV cache behavior, prefix-cache reuse, and storage performance.
Each record is one request, stored as one JSON object per line:
```
{
"timestamp": 27482,
"input_length": 6955,
"output_length": 52,
"hash_ids": [46, 47, 48, 49, 50, 51]
}
```
The fields are:
- `timestamp`: Request arrival time.
- `input_length`: Number of input tokens in the request.
- `output_length`: Number of generated output tokens.
- `hash_ids`: Remapped KV-cache block identifiers. Each ID represents a cache block, typically a 512-token block.
The dataset does not contain raw prompts, user text, or original token IDs. Instead, it keeps only the timing, token lengths, and anonymized block-hash sequence needed to replay cache access patterns.
It can be used to evaluate:
- KV cache hit ratio
- Prefix cache reuse
- Read/write behavior of KV storage
- Mooncake Store performance
- CPU/SSD offloading behavior
- Cache-aware scheduling simulations

### SCBench
[SCBench: A KV Cache-Centric Analysis of Long-Context Methods](https://hqjiang.com/scbench.html)
[MInference/scbench at main · microsoft/MInference](https://github.com/microsoft/MInference/tree/main/scbench)
CBench, or SharedContextBench, is a benchmark dataset for evaluating long-context LLM systems under multi-turn and multi-request scenarios.
Its main purpose is to test how well a model or serving system can reuse shared context and manage the KV cache lifecycle, including generation, compression, retrieval, loading, and reuse.
The dataset format is standardized as:
```
{
  "id": "random id",
  "context": "long shared context, such as code, documents, or many-shot examples",
  "multi_turns": [
    {
      "input": "question",
      "answer": "reference answer"
    }
  ]
}
```
SCBench covers tasks such as key-value lookup, prefix/suffix string retrieval, variable tracing, repository QA, long-document QA, many-shot learning, summarization, and mixed multi-task settings.
It supports two main evaluation modes:
- Multi-turn mode: multiple turns share the same long context within one session.
- Multi-request mode: multiple independent requests share the same context but ask different questions.
SCBench is used to evaluate the accuracy and efficiency of long-context methods and KV-cache reuse strategies in realistic shared-context workloads.
# SemiAnalysis Agentic Task Run Up

## Agentic Datasets

- [semianalysisai/cc-traces-weka-062126-256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)
- [semianalysisai/cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)

## Tooling

- The tool used is `aiperf`, specifically the SemiAnalysis fork.
- GitHub tree: [SemiAnalysisAI/aiperf@97ae7e898e8cce4070f779f2e0dc04f1d9bb5feb](https://github.com/SemiAnalysisAI/aiperf/tree/97ae7e898e8cce4070f779f2e0dc04f1d9bb5feb)
- The current SemiAnalysis fork supports the SA datasets `semianalysisai/cc-traces-weka-062126-256k` and `semianalysisai/cc-traces-weka-062126` through automatic download and testing only. It does not support parsing these datasets through an `input` argument, so the two datasets above do not need to be downloaded separately. Each model still needs tokenizer-specific data preprocessing before running the benchmark.

```bash
git clone https://github.com/SemiAnalysisAI/aiperf.git
cd aiperf
git checkout 97ae7e898e8cce4070f779f2e0dc04f1d9bb5feb

# Install the tool from the checked-out SemiAnalysis fork.
python -m pip install -e .
```

- Successful SA agentic runner: [SemiAnalysisAI/InferenceX actions run 28313061392](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28313061392)

```bash
/tmp/inferencex-agentic-31313/venv/bin/aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url http://localhost:8892 \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model Qwen/Qwen3.5-27B-FP8 \
  --concurrency 16 \
  --benchmark-duration 1800 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --agentic-cache-warmup-duration 600 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --tokenizer-trust-remote-code \
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --output-artifact-dir /workspace/results/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126_256K
```

## Agentic Test Task

The test flow is to start two ATOM workers first, then start ATOMesh without worker information, manually add the workers through `curl`, and finally launch the SA agentic task.

```bash
# 1. Start two ATOM workers, each with one GPU.
./run_mi355_atom.sh 8891 true 0
./run_mi355_atom.sh 8892 true 1

# 2. Start ATOMesh. Do not pass worker information at launch time.
./run_atomesh.sh 30000 true

# 3. Manually add workers to ATOMesh.
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"http://localhost:8891","model":"Qwen/Qwen3.5-27B-FP8"}'

curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"http://localhost:8892","model":"Qwen/Qwen3.5-27B-FP8"}'

# 4. Start the SA agentic task against ATOMesh.
./run_aiperf_sa_agentic.sh 127.0.0.1 30000 path/to/Qwen3.5-27B-FP8 semianalysis_cc_traces_weka_062126_256K
```

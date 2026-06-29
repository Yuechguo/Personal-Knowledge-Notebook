# SGLang-ATOM 基建搭建指南

本文档基于 `docker/Dockerfile` 和 `.github/` 下的 workflow、脚本与 benchmark 配置整理，目标是让没有参与过 SGLang-ATOM 基建的人，可以仿照现有实现从零搭建一套 Docker 发布、CI、nightly 测试和 benchmark/dashboard 流程。

文档中提到的 “ATOM 原生镜像” 指只包含 ATOM、Aiter、MORI、RCCL、Triton 等基础运行环境的 `rocm/atom-dev` 镜像；“vLLM OOT 镜像” 指在 ATOM 原生镜像上继续构建 vLLM ROCm OOT 的镜像；“SGLang 镜像” 指在 ATOM 原生镜像上继续安装 SGLang 的镜像。

## 1. 整体基建结构

SGLang-ATOM 的基建可以拆成五层：

1. Docker 构建层：`docker/Dockerfile` 通过多阶段构建生成三类镜像：ATOM 原生镜像、vLLM OOT 镜像、SGLang 镜像。
2. Docker 发布层：`.github/workflows/docker-release.yaml` 每天定时构建、验证并推送 Docker Hub 镜像。
3. PR/Main CI 层：`pre-checks.yaml` 做轻量静态检查，`atom-test.yaml`、`atom-vllm-test.yaml`、`atom-sglang-test.yaml` 做 GPU 准确率检查。
4. Nightly 层：`atom-test.yaml` 的 schedule 跑原生 ATOM 准确率全集，`atom-vllm-accuracy-validation.yaml` 和 `atom-sglang-accuracy-validation.yaml` 跑 vLLM/SGLang 适配的 nightly accuracy。
5. Benchmark 层：`atom-benchmark.yaml` 定时跑原生 ATOM 性能，`atom-vllm-benchmark.yaml` 和 `atom-sglang-benchmark.yaml` 手动跑 vLLM/SGLang 性能，并把数据推到 `gh-pages` 分支的 dashboard。

要仿照这套基建，核心不是单独复制某一个 workflow，而是同时准备好以下基础设施：

- Docker Hub 仓库，例如 `rocm/atom-dev`。
- GitHub Secrets：至少需要 `DOCKER_USERNAME`、`DOCKER_PASSWORD`、`AMD_HF_TOKEN`，部分 workflow 还依赖默认 `GITHUB_TOKEN`。
- 自托管 GPU runner 标签，例如 `linux-atom-mi35x-1`、`linux-atom-mi35x-4`、`linux-atom-mi35x-8`、`atom-mi355-8gpu.predownload`、`build-only-atom`。
- Runner 上的 GPU 设备挂载能力：workflow 会使用 `/dev/kfd`、`/dev/dri`，如果存在 `/etc/podinfo/gha-render-devices`，会从该文件读取 render device 参数。
- 可选但强烈建议的模型缓存目录：`/models`、`/it-share/models` 或 `/data/models`。CI 和 benchmark 会把 Hugging Face 模型下载到这些共享目录，避免每次重复下载。
- `gh-pages` 分支和 `benchmark-dashboard/` 目录，用于发布 accuracy 和 benchmark 数据。

## 2. 如何写 SGLang-ATOM Dockerfile

现有 Dockerfile 的关键路径是 `docker/Dockerfile`。它不是一个单目标镜像，而是一个多目标、多阶段 Dockerfile，包含三个最终可发布 target：

- `atom_image`：ATOM 原生镜像。
- `atom_oot`：vLLM OOT 镜像，基于已构建好的 ATOM 镜像。
- `atom_sglang`：SGLang 镜像，基于已构建好的 ATOM 镜像。

### 2.1 基础 ARG 设计

Dockerfile 开头定义了基础镜像和 GPU 架构：

```dockerfile
ARG BASE_IMAGE="rocm/pytorch:latest"
ARG OOT_BASE_IMAGE="rocm/atom-dev:latest"
ARG SGLANG_BASE_IMAGE="rocm/atom-dev:latest"
ARG GPU_ARCH="gfx942;gfx950"
```

仿照搭建时建议保持这类 ARG，而不是把版本写死在 workflow 里。这样 nightly 发布、手动发布、临时调试都可以通过 `--build-arg` 覆盖：

- `BASE_IMAGE` 控制 ATOM 原生镜像的基础 ROCm/PyTorch 环境。
- `OOT_BASE_IMAGE` 控制 vLLM OOT 镜像基于哪个 ATOM 镜像继续构建。
- `SGLANG_BASE_IMAGE` 控制 SGLang 镜像基于哪个 ATOM 镜像继续构建。
- `GPU_ARCH` 控制 MORI/RCCL/Aiter 等组件编译的 GPU arch。

### 2.2 ATOM 原生镜像：并行构建再合并

`atom_image` 的结构是：

1. `base`：从 `BASE_IMAGE` 出发，安装 git、cmake、OpenMPI、ibverbs、locale 等基础编译依赖，并设置 `GPU_ARCH_LIST`、`PYTORCH_ROCM_ARCH`。
2. `build_mori`：克隆并安装 `ROCm/mori`，默认 ref 是 `v1.1.0`。
3. `build_rccl`：克隆并编译 `ROCm/rccl`，默认 ref 是 `29e1567b95e28823b0beb1a988adc587bfab5b4f`。
4. `build_triton`：卸载基础镜像里的 Triton，然后从 `ROCm/triton` 的 `release/internal/3.5.x` 分支编译安装。
5. `build_aiter`：克隆并开发模式安装 Aiter，支持 `AITER_REPO`、`AITER_COMMIT`、`PREBUILD_KERNELS`、`MAX_JOBS`。
6. `atom_image`：拷贝或安装前面几个 stage 的产物，再 clone `ATOM_REPO`、checkout `ATOM_COMMIT`、`pip install -e .`。

这个设计的好处是 BuildKit 可以并行构建 MORI、RCCL、Triton、Aiter，最终再在 `atom_image` 合并产物。仿照实现时建议保留这个思路：

- 大型依赖独立成 stage，减少互相污染。
- 最终镜像只做产物合并和 ATOM 轻量安装。
- 对经常变化的 ATOM 源码使用 `CACHEBUST`，只失效最后一层，避免每次都重编 RCCL/Triton/Aiter。

### 2.3 vLLM OOT 镜像：保留 ATOM 基础环境

`atom_oot` 基于 `OOT_BASE_IMAGE`，默认是 `rocm/atom-dev:latest`。它做的事情包括：

- 克隆 `VLLM_REPO`，checkout 固定 `VLLM_COMMIT`。
- 安装 vLLM ROCm 构建依赖。
- 构建 vLLM wheel 并安装。
- 安装 `lm-eval[api]`、`fastsafetensors` 等可选运行依赖。
- 打上 `LABEL com.rocm.atom.vllm_commit="${VLLM_COMMIT}"`，用于 CI 判断预构建镜像是否匹配当前 vLLM commit。

这里有一个重要细节：OOT 构建会先备份 ATOM base 里实际 `import triton` 得到的 Triton 包和对应 dist-info，vLLM 安装结束后再恢复该 Triton。原因是 vLLM/OOT 的依赖安装可能改动 Triton 元数据，而 ATOM base 里经过验证的 Triton 版本才是运行时基准。仿照搭建时，如果你的上层框架会改动底层编译器或 runtime 包，也应显式备份和恢复关键包，避免 CI 里出现“pip show 和 import 不一致”或 runtime 行为漂移。

### 2.4 SGLang 镜像：局部覆盖 SGLang 所需依赖

`atom_sglang` 基于 `SGLANG_BASE_IMAGE`，默认也是 `rocm/atom-dev:latest`。它做的事情包括：

- 克隆 `SGLANG_REPO`，checkout `SGLANG_REF`，默认 `v0.5.10`。
- 编译 `sgl-kernel` 的 ROCm kernel。
- 安装 SGLang Python 包和 runtime 依赖。
- 单独安装 `triton==SGLANG_TRITON_VERSION`，默认 `3.6.0`。
- 打上 `LABEL com.rocm.atom.sglang_ref="${SGLANG_REF}"`。

SGLang 镜像和 vLLM OOT 镜像对 Triton 的处理不同：OOT 尽量恢复 ATOM base 的 Triton；SGLang 镜像则在最终阶段安装 SGLang 验证过的 Triton 3.6.0，因为 DeepSeek-R1 的 SGLang serving 在旧 Triton 上会遇到编译路径问题。仿照搭建时，建议把“基础镜像通用依赖”和“某个上层框架专属依赖”分开处理，避免为了 SGLang 影响原生 ATOM 或 vLLM OOT 镜像。

### 2.5 建议的构建命令

原生 ATOM 镜像：

```bash
DOCKER_BUILDKIT=1 docker build --pull --network=host \
  -t atom_release:ci \
  --target atom_image \
  --build-arg BASE_IMAGE=rocm/pytorch:latest \
  --build-arg GPU_ARCH="gfx942;gfx950" \
  --build-arg ATOM_REPO=https://github.com/ROCm/ATOM.git \
  --build-arg ATOM_COMMIT=HEAD \
  --build-arg AITER_REPO=https://github.com/ROCm/aiter.git \
  --build-arg AITER_COMMIT=HEAD \
  --build-arg RCCL_REPO=https://github.com/ROCm/rccl.git \
  --build-arg RCCL_BRANCH=29e1567b95e28823b0beb1a988adc587bfab5b4f \
  --build-arg CACHEBUST="$(date +%s)" \
  -f docker/Dockerfile .
```

vLLM OOT 镜像：

```bash
docker build --network=host \
  -t atom_oot_release:ci \
  --target atom_oot \
  --build-arg OOT_BASE_IMAGE=atom_release:ci \
  --build-arg MAX_JOBS=64 \
  --build-arg VLLM_COMMIT=2a69949bdadf0e8942b7a1619b229cb475beef20 \
  --build-arg INSTALL_LM_EVAL=1 \
  --build-arg INSTALL_FASTSAFETENSORS=1 \
  -f docker/Dockerfile .
```

SGLang 镜像：

```bash
docker build --network=host \
  -t atom_sglang_release:ci \
  --target atom_sglang \
  --build-arg SGLANG_BASE_IMAGE=atom_release:ci \
  --build-arg GPU_ARCH="gfx942;gfx950" \
  --build-arg SGLANG_REPO=https://github.com/sgl-project/sglang.git \
  --build-arg SGLANG_REF=v0.5.10 \
  -f docker/Dockerfile .
```

## 3. Docker 镜像如何发布

Docker 发布由 `.github/workflows/docker-release.yaml` 完成，workflow 名称是 `Nightly Docker Release`。

### 3.1 触发机制和发布时间

发布有两种触发方式：

- 定时触发：`cron: '0 14 * * *'`，即每天 UTC 14:00，北京时间 22:00。
- 手动触发：`workflow_dispatch`，可以指定 base image、ATOM/Aiter/RCCL/SGLang 仓库和 ref、runner、是否跳过测试、是否只发布 OOT 或 SGLang 镜像等。

默认定时发布会构建并发布三类镜像：

- ATOM 原生镜像。
- vLLM OOT 镜像。
- SGLang 镜像。

### 3.2 发布流程

发布流程是：

1. 设置 `HF_TOKEN`，checkout 仓库。
2. 使用 `DOCKER_USERNAME` 和 `DOCKER_PASSWORD` 登录 Docker Hub。
3. 构建 `atom_release:ci`，target 为 `atom_image`。
4. 如果没有设置 `skip_tests` 且不是 `only_release_oot` / `only_release_sglang`，启动容器跑一次 `atom.examples.simple_inference`，并和 golden output 做 diff。
5. 推送 ATOM 原生镜像 tag。
6. 基于刚构建的 `atom_release:ci` 构建 `atom_oot`，推送 vLLM OOT tag。
7. 基于刚构建的 `atom_release:ci` 构建 `atom_sglang`，推送 SGLang tag。
8. 清理本地容器和镜像，释放 runner 磁盘空间。

### 3.3 镜像 tag 规则

ATOM 原生镜像：

- `rocm/atom-dev:latest`
- `rocm/atom-dev:nightly_YYYYMMDDHHMM`

vLLM OOT 镜像：

- `rocm/atom-dev:vllm-v${VLLM_VERSION}-nightly_YYYYMMDD`
- `rocm/atom-dev:vllm-latest`

SGLang 镜像：

- `rocm/atom-dev:sglang-v${SGLANG_VERSION}-nightly_YYYYMMDD`
- `rocm/atom-dev:sglang-latest`

当前 workflow 中固定的版本元数据包括：

- `VLLM_COMMIT=2a69949bdadf0e8942b7a1619b229cb475beef20`
- `VLLM_VERSION=0.19.0`
- `SGLANG_REF=v0.5.10`
- `SGLANG_VERSION=0.5.10`

需要注意的是，Dockerfile 里也有默认 `VLLM_COMMIT`、`SGLANG_REF`，但 nightly 发布实际使用的是 `docker-release.yaml` 里的 env 值。维护时要以 workflow 为发布事实来源，并同步更新 Dockerfile 默认值，避免本地构建和 nightly 构建产生混淆。

## 4. CI 搭建方式

SGLang-ATOM 的 CI 分成轻量 pre-check 和 GPU case 两层。轻量检查先完成并上传 signal artifact，GPU workflow 再通过 `check_signal.sh` 读取该 signal，避免静态检查失败时继续占用 GPU runner。

### 4.1 Pre Checkin

文件：`.github/workflows/pre-checks.yaml`

触发条件：

- `push` 到 `main`
- 指向 `main` 的 `pull_request`，类型包括 `opened`、`synchronize`、`reopened`、`ready_for_review`
- `workflow_dispatch`
- 定时 `cron: '0 22 * * *'`，北京时间 06:00

忽略路径：

- `**/*.md`
- `docs/**`
- `LICENSE`
- `.gitignore`

覆盖内容：

- Black 格式检查。
- Ruff 检查，并通过 reviewdog 在 PR diff 上评论。
- 成功或失败后都会上传 `checks-signal-${{ github.sha }}` artifact，内容为 `success` 或 `failure`。

新增轻量 CI 时，建议放到 `pre-checks.yaml`，并把新 job 加入 `upload-success-artifact` 和 `upload-failure-artifact` 的 `needs`，这样 GPU workflow 可以继续复用同一个 signal。

### 4.2 ATOM 原生准确率 CI

文件：`.github/workflows/atom-test.yaml`

触发条件：

- `push` 到 `main`
- 指向 `main` 的 PR，类型包括 `opened`、`synchronize`、`reopened`、`ready_for_review`
- 定时 `cron: '0 16 * * *'`，北京时间 00:00
- 手动 `workflow_dispatch`

执行逻辑：

1. `check-signal` 检查 pre-check signal。手动触发时跳过该 gate。
2. `download_aiter_wheel` 优先从 S3 manifest 下载最新 Aiter main wheel，失败时从 `ROCm/aiter` workflow artifact 回退下载。
3. `load-test-models` 读取 `.github/benchmark/models_accuracy.json`，根据事件类型过滤模型。
4. GPU runner 启动 `rocm/atom-dev:latest` 或已解析到 digest 的 immutable 镜像。
5. 容器内安装刚下载的 Aiter wheel、安装当前 ATOM 源码、下载模型。
6. 通过 `.github/scripts/atom_test.sh launch` 启动服务，通过 `.github/scripts/atom_test.sh accuracy` 跑 GSM8K accuracy。
7. 用 `models_accuracy.json` 中的 `accuracy_threshold` 判断是否通过。
8. 在 `main` 的 `push` 或 `schedule` 场景，汇总 accuracy artifact 并通过 dashboard action 写入 `gh-pages`。

`models_accuracy.json` 的 `test_level` 控制 case 覆盖范围：

- PR：只跑 `test_level=pr`。
- Push 到 main：跑 `pr` 和 `main`。
- Schedule 或手动：跑 `pr`、`main`、`nightly` 全集。

当前原生 ATOM accuracy 覆盖的模型包括 Llama、DeepSeek、gpt-oss、Qwen、Kimi、GLM、MiniMax 等，具体以 `.github/benchmark/models_accuracy.json` 为准。

新增原生 ATOM accuracy case 的步骤：

1. 在 `.github/benchmark/models_accuracy.json` 添加一个对象。
2. 必填或常用字段包括 `model_name`、`model_path`、`extraArgs`、`env_vars`、`runner`、`test_level`、`accuracy_threshold`。
3. 如果 dashboard 需要显示基线，可补充 `accuracy_baseline`、`accuracy_baseline_model`、`_baseline_note`。
4. 根据模型大小选择合适 runner，例如单卡、四卡、八卡。
5. 如果使用了新的 runner label，同时更新 `.github/runner-config.yml`，便于 dashboard 展示 GPU 架构和卡数。

### 4.3 vLLM OOT PR CI

文件：`.github/workflows/atom-vllm-test.yaml`

触发条件：

- 指向 `main` 的 PR，类型包括 `opened`、`synchronize`、`reopened`、`ready_for_review`、`closed`。
- 实际 GPU job 会过滤掉 `closed` 和 draft PR。
- 忽略 `*.md`、`docs/**`、`LICENSE`、`.gitignore`。

执行逻辑：

1. 检查 pre-check signal。
2. 下载最新 Aiter wheel。
3. 尝试拉取 `rocm/atom-dev:vllm-latest`。
4. 检查镜像 label `com.rocm.atom.vllm_commit` 是否等于 workflow 期望的 `VLLM_COMMIT`。
5. 如果 label 匹配，走 fast path：在预构建 OOT 镜像上叠加当前 PR 的 ATOM 和 Aiter wheel。
6. 如果 label 不匹配或拉取失败，走 full path：从 `rocm/atom-dev:latest` 开始重建 `atom_oot`。
7. 启动容器，下载模型，执行 `.github/scripts/atom_oot_test.sh accuracy ci`。
8. 检查 GSM8K 的 `exact_match,flexible-extract` 是否超过 case 阈值。

当前 PR CI case 直接写在 workflow 的 `matrix.include` 中，包括：

- DeepSeek-R1-FP8 TP8
- gpt-oss-120b TP1
- Kimi-K2-Thinking-MXFP4 TP4
- Qwen3.5-35B-A3B-FP8 TP2

新增 vLLM OOT PR case 的步骤：

1. 编辑 `.github/workflows/atom-vllm-test.yaml` 的 `matrix.include`。
2. 增加 `display_name`、`model_name`、`model_path`、`extra_args`、`env_vars`、`accuracy_test_threshold`、`runner`。
3. 如果该 case 需要特殊 few-shot，可增加或使用 `lm_eval_num_fewshot`。
4. 确保所选 runner 有足够 GPU 数和显存。

### 4.4 SGLang PR CI

文件：`.github/workflows/atom-sglang-test.yaml`

触发条件和整体逻辑与 vLLM OOT PR CI 类似。差异在于：

- 预构建镜像是 `rocm/atom-dev:sglang-latest`。
- 校验 label 是 `com.rocm.atom.sglang_ref`。
- full path 会重建 `atom_sglang` target。
- 启动和准确率测试通过 `.github/scripts/atom_sglang_test.sh accuracy`。

当前 SGLang PR CI 覆盖一个 case：

- DeepSeek-R1-FP8 TP4

新增 SGLang PR case 的步骤：

1. 编辑 `.github/workflows/atom-sglang-test.yaml` 的 `matrix.include`。
2. 增加 `model_name`、`model_path`、`extra_args`、`env_vars`、`accuracy_test_threshold`、`runner`。
3. 确认 `SGLANG_REF` 和发布镜像的 `sglang-latest` label 一致，否则 CI 会转为 full rebuild。

## 5. Nightly 测试搭建方式

这里的 nightly 包括 nightly accuracy 和 nightly Docker/benchmark。它们都通过 GitHub Actions schedule 触发，时间统一写在各 workflow 的 cron 中。

### 5.1 Nightly 时间安排

- Docker 发布：`.github/workflows/docker-release.yaml`，UTC 14:00，北京时间 22:00。
- 原生 ATOM accuracy：`.github/workflows/atom-test.yaml`，UTC 16:00，北京时间 00:00。
- 原生 ATOM benchmark：`.github/workflows/atom-benchmark.yaml`，UTC 17:00，北京时间 01:00。
- vLLM OOT nightly accuracy：`.github/workflows/atom-vllm-accuracy-validation.yaml`，UTC 18:00，北京时间 02:00。
- SGLang nightly accuracy：`.github/workflows/atom-sglang-accuracy-validation.yaml`，UTC 18:00，北京时间 02:00。
- Pre-check 定时检查：`.github/workflows/pre-checks.yaml`，UTC 22:00，北京时间 06:00。

这个顺序大致形成了“晚上先发布镜像，凌晨再基于发布镜像跑 accuracy 和 benchmark”的节奏。仿照搭建时，应避免把 Docker 发布和大规模 GPU 测试安排在同一时间点，减少 runner 和镜像 tag 竞争。

### 5.2 原生 ATOM nightly accuracy

原生 ATOM nightly 使用 `atom-test.yaml` 的 schedule。和 PR/main CI 使用同一套执行逻辑，区别是 `load-test-models` 会把 `models_accuracy.json` 中 `pr`、`main`、`nightly` 全部纳入矩阵。

新增原生 ATOM nightly case 的方式与新增 CI case 相同，只需要把 `test_level` 设置为 `nightly`。

### 5.3 vLLM OOT nightly accuracy

文件：`.github/workflows/atom-vllm-accuracy-validation.yaml`

触发条件：

- 定时 `cron: '0 18 * * *'`，北京时间 02:00。
- 手动 `workflow_dispatch`，可以通过多个 boolean input 选择模型，也可以选择是否上传 accuracy 到 dashboard。

执行逻辑：

1. `prepare-oot-image` 在 `build-only-atom` runner 上准备验证镜像。
2. schedule 场景会解析 `rocm/atom-dev:vllm-latest` 到 immutable digest，优先使用已发布镜像。
3. 手动非 main 分支场景会从当前分支重建临时 OOT 镜像，并推送到 `rocm/atom-dev:vllm-full-manual-${run_id}-${sha}`。
4. workflow 内嵌 Python 列表生成 model matrix。schedule 会启用所有模型；手动触发只启用勾选的模型。
5. 每个模型启动容器、下载模型、运行 `.github/scripts/atom_oot_test.sh accuracy full`。
6. 结果 artifact 名称为 `accuracy-${{ matrix.model_name }}`，schedule 或手动勾选 `upload_accuracy_to_dashboard` 时上传到 dashboard。

当前 vLLM OOT nightly accuracy 覆盖的模型包括：

- Qwen3-235B-A22B-Instruct-2507-FP8 TP8+EP8
- Qwen3-Next-80B-A3B-Instruct-FP8 TP4
- Qwen3.5-397B-A17B-FP8 TP8
- Qwen3.5-397B-A17B TP8
- Qwen3.5-397B-A17B-MXFP4 TP4
- Meta-Llama-3.1-405B-Instruct-FP8 TP8
- Llama-3.1-8B-Instruct TP1
- Kimi-K2-Thinking-MXFP4 TP8
- Kimi-K2.5-MXFP4 TP8
- DeepSeek-R1-FP8 TP8
- DeepSeek-R1-0528-MXFP4 TP8
- gpt-oss-120b TP1/TP2
- GLM-5.1-FP8 TP8

新增 vLLM OOT nightly case 的步骤：

1. 在 `workflow_dispatch.inputs` 中增加一个 boolean input，例如 `run_new_model_tp4`。
2. 在 `prepare-oot-image` 的 env 中把 input 映射成环境变量。
3. 在内嵌 Python `models` 列表中增加对象，字段包括 `toggle_env`、`model_name`、`model_path`、`extra_args`、`accuracy_test_threshold`、`env_vars`、`runner`，需要时可加 `lm_eval_num_fewshot`。
4. 如果要让 dashboard 知道基线，在 `.github/benchmark/oot_models_accuracy.json` 中增加对应条目。

### 5.4 SGLang nightly accuracy

文件：`.github/workflows/atom-sglang-accuracy-validation.yaml`

触发条件：

- 定时 `cron: '0 18 * * *'`，北京时间 02:00。
- 手动 `workflow_dispatch`，可以选择模型和是否上传 dashboard。

执行逻辑与 vLLM OOT nightly 类似，主要差异是：

- 使用 `rocm/atom-dev:sglang-latest`。
- main 或 schedule 优先解析已发布 SGLang 镜像。
- 非 main 手动触发会重建并推送临时 SGLang 镜像。
- 测试脚本是 `.github/scripts/atom_sglang_test.sh accuracy`。
- dashboard backend 标记为 `ATOM-SGLang`。

当前覆盖一个 nightly case：

- DeepSeek-R1-FP8 TP4

新增 SGLang nightly case 的步骤：

1. 在 `workflow_dispatch.inputs` 中增加 boolean input。
2. 在 `prepare-sglang-image` 的 env 中映射该 input。
3. 在内嵌 Python `models` 列表中增加模型对象。
4. 在 `.github/benchmark/sglang_models_accuracy.json` 中增加同名 accuracy 配置，供 dashboard 转换脚本使用。

## 6. Benchmark 搭建方式

SGLang-ATOM 有三套 benchmark：

- 原生 ATOM benchmark：定时和手动均支持。
- vLLM OOT benchmark：仅手动。
- SGLang benchmark：仅手动。

三者共同模式是：

1. 读取模型配置 JSON。
2. 读取或解析 benchmark 参数组合。
3. 构建 matrix：模型 x 输入输出长度 x 并发。
4. 在 GPU runner 启动容器和服务端。
5. 使用 benchmark client 发送随机数据请求。
6. 生成 JSON 结果并注入 GPU、ROCm、Docker image、源码 ref 等元数据。
7. 上传 artifact。
8. 汇总结果，必要时和 baseline 比较。
9. 转换成 `benchmark-action/github-action-benchmark@v1` 可识别的格式。
10. 推送到 `gh-pages` 的 `benchmark-dashboard/data.js`。

### 6.1 原生 ATOM benchmark

文件：`.github/workflows/atom-benchmark.yaml`

触发条件：

- 定时 `cron: '0 17 * * *'`，北京时间 01:00。
- 手动 `workflow_dispatch`。

模型配置：

- `.github/benchmark/models.json`

参数配置：

- schedule 使用 `.github/benchmark/nightly_params.json`。
- 手动触发使用 `param_lists` input，格式为 `input_length,output_length,concurrency,random_range_ratio`，多组用分号分隔。

当前 nightly 参数包括：

- ISL 1024 / OSL 1024，并发 4、8、16、32、64、128、256。
- ISL 8192 / OSL 1024，并发 4、8、16、32、64、128、256。

执行流程：

1. `parse-param-lists` 解析参数矩阵。
2. `load-models` 读取 `.github/benchmark/models.json`。schedule 跑全部模型，手动触发按 workflow input 的 boolean 开关过滤。
3. `benchmark` job 在模型指定 runner 上启动 `rocm/atom-dev:latest` 或手动指定的 image。
4. `.github/scripts/atom_test.sh launch` 启动 ATOM server。
5. `.github/scripts/atom_test.sh benchmark` 运行 benchmark。
6. 输出 `${RESULT_FILENAME}.json`，注入 GPU 名称、显存、ROCm 版本、Docker image、display name。
7. 上传 `benchmark-${RESULT_FILENAME}` artifact。
8. `summarize-benchmark-result` 下载所有结果，用 `.github/scripts/summarize.py` 汇总并和上一次 schedule run 的 artifact 做回归比较。
9. `.github/scripts/plugin_benchmark_to_dashboard.py` 转成 dashboard input。
10. `benchmark-action/github-action-benchmark@v1` 写出 `benchmark-dashboard/data.js`，但 `auto-push: false`。
11. workflow 手动 checkout `gh-pages`，复制自定义 dashboard 页面 `.github/dashboard/index.html` 和 logo，生成 `models_map.js`，commit 并 push。

性能回归处理：

- 如果 `summarize.py` 检测到 regression，会上传 `regression-report`。
- `regression-rerun` 根据 `.github/scripts/regression_rerun.py` 生成回归复跑矩阵，并开启 profiler。
- `profiler-analysis` 会解析 profiler trace，生成分析 artifact。
- 如果仍有回归，会通过 GitHub issue 记录 regression 信息。

新增原生 ATOM benchmark case 的步骤：

1. 在 `.github/benchmark/models.json` 增加模型对象。
2. 字段包括 `display`、`path`、`prefix`、`args`、`bench_args`、`suffix`、`runner`、`env_vars`。
3. 如果希望手动触发页面有独立开关，需要在 `atom-benchmark.yaml` 的 `workflow_dispatch.inputs` 增加 boolean input，input 名称通常与 `prefix` 对应。
4. 如果是 nightly 固定跑的模型，只加到 `models.json` 就会被 schedule 自动纳入。
5. 如需新增 ISL/OSL/concurrency 组合，编辑 `.github/benchmark/nightly_params.json`。
6. 如果低并发或特定参数对某个模型不适用，可在 matrix `exclude` 或模型级参数过滤中加规则。

### 6.2 vLLM OOT benchmark

文件：`.github/workflows/atom-vllm-benchmark.yaml`

触发条件：

- 仅 `workflow_dispatch`。

主要 input：

- 多个模型 boolean 开关。
- `oot_image`：可手动指定 OOT 镜像。
- `publish_to_dashboard`：是否上传 dashboard，默认 true。
- `param_lists`：benchmark 参数组合。

模型配置：

- `.github/benchmark/oot_benchmark_models.json`

执行流程：

1. `resolve-atom-source` 判断镜像来源。
2. 如果手动提供 `oot_image`，直接使用该镜像。
3. 如果在 main 分支且未提供镜像，解析 `rocm/atom-dev:vllm-latest`，并优先 pin 到 digest 或匹配到同 digest 的 nightly tag。
4. 如果不是 main 分支且未提供镜像，构建并推送临时 OOT benchmark 镜像，tag 形如 `rocm/atom-dev:oot-benchmark-${run_id}-${attempt}-${sha}-${vllm_version}-${vllm_commit}`。
5. `parse-param-lists` 解析参数，并限制 concurrency 只能是 4、8、16、32、64、128、256。
6. `load-models` 按 boolean input 从 `oot_benchmark_models.json` 过滤模型。
7. `build-benchmark-matrix` 做模型与参数组合，并根据 `supported_input_output_pairs`、`excluded_input_output_pairs` 过滤不支持的 ISL/OSL。
8. GPU job 启动 OOT 容器，clone `https://github.com/kimbochen/bench_serving.git`，使用 `benchmark_serving.py --backend=vllm` 压测。
9. 结果 JSON 会被注入 `benchmark_backend=ATOM-vLLM`、tensor parallel size、ATOM source、vLLM commit/version、镜像来源等元数据。
10. 汇总 job 用 `plugin_benchmark_summary.py` 生成 summary，用 `plugin_benchmark_regression.py` 对比 baseline。
11. dashboard 上传使用 `plugin_benchmark_to_dashboard.py` 和 `github-action-benchmark`，最后只把 `benchmark-dashboard/data.js` 推回 `gh-pages`。

新增 vLLM OOT benchmark case 的步骤：

1. 在 `.github/benchmark/oot_benchmark_models.json` 增加模型对象。
2. 字段包括 `display`、`dashboard_model`、`source_path`、`path`、`prefix`、`extra_args`、`bench_args`、`runner`、`env_vars`。
3. 如模型只支持部分输入输出长度，增加 `supported_input_output_pairs`；如只需排除少数组合，增加 `excluded_input_output_pairs`。
4. 在 `atom-vllm-benchmark.yaml` 的 `workflow_dispatch.inputs` 增加 boolean 开关。
5. 在 `load-models` job 的 env 和 jq filter 中把新 input 与新 `prefix` 对应起来。
6. 在后面的 “Check if model is enabled” case 语句中增加同样的 prefix 映射，保持矩阵 job 和手动开关一致。

### 6.3 SGLang benchmark

文件：`.github/workflows/atom-sglang-benchmark.yaml`

触发条件：

- 仅 `workflow_dispatch`。

主要 input：

- DeepSeek FP8/FP4 TP8/TP4 四个模型开关。
- `sglang_image`：可手动指定 SGLang 镜像。
- `publish_to_dashboard`：是否上传 dashboard，默认 true。
- `param_lists`：benchmark 参数组合。

模型配置：

- `.github/benchmark/sglang_benchmark_models.json`

执行流程：

1. `resolve-atom-source` 判断镜像来源。
2. main 分支默认使用 `rocm/atom-dev:sglang-latest`。
3. 非 main 且未指定镜像时，构建并推送临时 SGLang benchmark 镜像，tag 形如 `rocm/atom-dev:sglang-benchmark-${run_id}-${attempt}-${sha}-${sglang_version}-${sglang_ref}`。
4. `load-models` 按 input 从 `sglang_benchmark_models.json` 过滤模型。
5. GPU job 启动 SGLang 容器，clone `bench_serving`，先通过 `.github/scripts/atom_sglang_test.sh launch` 启动服务，再使用 `benchmark_serving.py --backend=sglang` 压测。
6. 结果 JSON 注入 `benchmark_backend=ATOM-SGLang`、SGLang ref/version、镜像来源、ATOM source 等元数据。
7. 汇总和 dashboard 发布方式与 vLLM OOT benchmark 类似，只更新 `benchmark-dashboard/data.js`。

新增 SGLang benchmark case 的步骤：

1. 在 `.github/benchmark/sglang_benchmark_models.json` 增加模型对象。
2. 在 `atom-sglang-benchmark.yaml` 的 `workflow_dispatch.inputs` 增加 boolean 开关。
3. 在 `load-models` job 的 env 和 jq filter 中加入新 `prefix`。
4. 在 “Check if model is enabled” case 语句中加入新 `prefix`。
5. 如需要限制参数组合，使用 `supported_input_output_pairs` 或 `excluded_input_output_pairs`。

## 7. Dashboard 数据推送方式

SGLang-ATOM 的 accuracy 和 benchmark 都复用了 `benchmark-action/github-action-benchmark@v1`，目标分支是 `gh-pages`，数据目录是 `benchmark-dashboard`。

### 7.1 Accuracy dashboard

原生 ATOM accuracy：

- workflow：`atom-test.yaml`
- 触发上传条件：`main` 分支上的 `push` 或 `schedule`
- 转换脚本：`.github/scripts/accuracy_to_dashboard.py`
- 模型元数据：`.github/benchmark/models_accuracy.json`
- backend 默认是原生 ATOM
- `github-action-benchmark` 使用 `auto-push: true`

vLLM OOT accuracy：

- workflow：`atom-vllm-accuracy-validation.yaml`
- 上传条件：schedule，或手动触发且勾选 `upload_accuracy_to_dashboard`
- 转换脚本：`.github/scripts/accuracy_to_dashboard.py`
- 模型元数据：`.github/benchmark/oot_models_accuracy.json`
- backend：`ATOM-vLLM`
- `auto-push: true`

SGLang accuracy：

- workflow：`atom-sglang-accuracy-validation.yaml`
- 上传条件：schedule，或手动触发且勾选 `upload_accuracy_to_dashboard`
- 转换脚本：`.github/scripts/accuracy_to_dashboard.py`
- 模型元数据：`.github/benchmark/sglang_models_accuracy.json`
- backend：`ATOM-SGLang`
- `auto-push: true`

### 7.2 Benchmark dashboard

原生 ATOM benchmark：

- 转换脚本：`.github/scripts/plugin_benchmark_to_dashboard.py`
- `github-action-benchmark` 配置：`tool: customBiggerIsBetter`、`auto-push: false`
- workflow 后续手动 checkout `gh-pages`，复制 `.github/dashboard/index.html`、logo 和 `models_map.js`，再 commit/push 整个 `benchmark-dashboard/`。

vLLM OOT benchmark 和 SGLang benchmark：

- 转换脚本同样是 `.github/scripts/plugin_benchmark_to_dashboard.py`
- `github-action-benchmark` 生成 `benchmark-dashboard/data.js`
- workflow 随后 checkout `gh-pages`，只更新 `benchmark-dashboard/data.js`，避免覆盖原生 ATOM 自定义 dashboard 页面。

仿照搭建时，建议把“数据转换”和“前端页面发布”分开：

- 数据转换脚本只负责把 benchmark JSON 转成 action 需要的格式。
- `github-action-benchmark` 只负责维护 `data.js`。
- 自定义页面、logo、模型 display map 由单独步骤合并到 `gh-pages`。

## 8. 从零搭建一套类似基建的建议步骤

1. 准备 Dockerfile。先实现 `atom_image`，确认基础依赖、Aiter、Triton、RCCL、MORI 和 ATOM 都能构建；再增加 `atom_oot` 和 `atom_sglang` 这类上层框架 target。
2. 准备 Docker 发布 workflow。先手动 `workflow_dispatch` 跑通构建和推送，再开启 schedule。发布时固定版本变量，并给上层镜像打 label，供 CI 判断复用还是重建。
3. 准备 runner 和 secrets。GPU runner 要能访问 `/dev/kfd`、`/dev/dri`，Docker/Podman 可用，Hugging Face token 可用，Docker Hub 可登录。
4. 准备模型缓存。大模型 CI 没有共享缓存会非常慢，建议在 runner 上挂 `/models` 或统一路径，并在下载脚本里加 lock。
5. 先搭 pre-check。Black/Ruff 这类轻量检查先跑，通过 artifact signal 控制 GPU workflow 是否继续。
6. 搭原生 ATOM CI。把模型 case 放到 JSON，workflow 根据事件类型过滤 `pr`、`main`、`nightly`。
7. 搭上层框架 PR CI。vLLM/SGLang 这类上层框架要优先复用 nightly 发布镜像，通过 label 判断版本是否匹配；不匹配时再 full rebuild。
8. 搭 nightly accuracy。nightly 适合放更大、更慢、更接近真实部署的模型矩阵；手动触发时加 checkbox，方便只跑某一个模型。
9. 搭 benchmark。先把单模型、单参数跑通，再引入模型 JSON、参数矩阵、artifact 汇总、baseline 比较和 dashboard。
10. 搭 dashboard。先用 `github-action-benchmark` 写数据，再按需要替换自定义 `index.html`，注意不同 benchmark workflow 不要互相覆盖页面资源。
11. 建立新增 case 的规范。明确每类 case 应改哪个 JSON、哪个 workflow input、哪个 jq/Python 过滤逻辑，以及是否要更新 runner metadata。

## 9. 关键文件索引

- Dockerfile：`docker/Dockerfile`
- Docker 发布：`.github/workflows/docker-release.yaml`
- 轻量检查：`.github/workflows/pre-checks.yaml`
- pre-check signal 脚本：`.github/scripts/check_signal.sh`
- 原生 ATOM accuracy：`.github/workflows/atom-test.yaml`
- 原生 ATOM accuracy case：`.github/benchmark/models_accuracy.json`
- vLLM OOT PR CI：`.github/workflows/atom-vllm-test.yaml`
- SGLang PR CI：`.github/workflows/atom-sglang-test.yaml`
- vLLM OOT nightly accuracy：`.github/workflows/atom-vllm-accuracy-validation.yaml`
- SGLang nightly accuracy：`.github/workflows/atom-sglang-accuracy-validation.yaml`
- 原生 ATOM benchmark：`.github/workflows/atom-benchmark.yaml`
- 原生 ATOM benchmark case：`.github/benchmark/models.json`
- 原生 ATOM nightly benchmark 参数：`.github/benchmark/nightly_params.json`
- vLLM OOT benchmark：`.github/workflows/atom-vllm-benchmark.yaml`
- vLLM OOT benchmark case：`.github/benchmark/oot_benchmark_models.json`
- SGLang benchmark：`.github/workflows/atom-sglang-benchmark.yaml`
- SGLang benchmark case：`.github/benchmark/sglang_benchmark_models.json`
- vLLM OOT accuracy dashboard 元数据：`.github/benchmark/oot_models_accuracy.json`
- SGLang accuracy dashboard 元数据：`.github/benchmark/sglang_models_accuracy.json`
- 镜像 digest 解析：`.github/scripts/resolve_atom_image.py`
- accuracy dashboard 转换：`.github/scripts/accuracy_to_dashboard.py`
- benchmark dashboard 转换：`.github/scripts/plugin_benchmark_to_dashboard.py`
- benchmark summary：`.github/scripts/plugin_benchmark_summary.py`
- benchmark regression：`.github/scripts/plugin_benchmark_regression.py`
- 原生 ATOM 测试脚本：`.github/scripts/atom_test.sh`
- vLLM OOT 测试脚本：`.github/scripts/atom_oot_test.sh`
- SGLang 测试脚本：`.github/scripts/atom_sglang_test.sh`
- 模型下载锁脚本：`.github/scripts/download_model_with_lock.sh`
- 自定义 dashboard 页面：`.github/dashboard/index.html`
- runner 元数据：`.github/runner-config.yml`


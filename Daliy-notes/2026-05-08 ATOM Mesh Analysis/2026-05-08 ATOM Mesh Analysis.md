# ATOM Mesh Analysis
## Overview
`atom/mesh` is a Rust inference gateway for ATOM serving deployments. Its main job is to accept OpenAI-compatible and SGLang-compatible HTTP requests, choose suitable backend workers, and forward or execute the request through either HTTP proxying or a Rust-native gRPC pipeline.
The folder is organized as an independent Rust crate:

- `Cargo.toml` defines the `atom-mesh` package, `mesh` library crate, and `atom-mesh` binary.
- `src/main.rs` is the CLI entry point. It parses launch options such as worker URLs, routing policy, PD mode, gRPC mode, tokenizer paths, retry/circuit-breaker settings, metrics, logging, and rate limiting.
- `src/server.rs` builds and starts the Axum HTTP server, initializes shared context, registers workers, starts background health/load tasks, and wires HTTP routes.
- `src/lib.rs` exposes the main modules: `config`, `core`, `middleware`, `observability`, `policies`, `routers`, and `server`.

At a high level, Mesh is both a control plane and a data plane:

- Control plane: worker registration, worker updates/removal, tokenizer registration, health checks, load monitoring, cache flush, metrics, and admin endpoints.
- Data plane: routing `/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, tokenization, detokenization, and parser endpoints to backend inference workers.

## Main Capabilities
### Inference Gateway
Mesh exposes a front-door HTTP API and routes model requests to backend workers. The main public/protected API surface is assembled in `src/server.rs`:

- `POST /generate`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `GET/POST/DELETE /v1/responses/{response_id}`
- `GET /v1/models`
- `GET /health`, `/readiness`, `/liveness`
- `POST /v1/tokenize`, `POST /v1/detokenize`
- `POST /parse/reasoning`, `POST /parse/function_call`
- Worker APIs: `POST /workers`, `GET /workers`, `GET/PUT/DELETE /workers/{worker_id}`

The server routes request handlers through a shared `RouterTrait` abstraction in `src/routers/mod.rs`. This lets the HTTP server call the same logical methods regardless of backend transport or routing mode:

- `route_generate`
- `route_chat`
- `route_completion`
- `route_responses`
- `get_models`
- `get_model_info`
- `get_server_info`

### Regular and PD Routing Modes
Mesh supports two deployment topologies through `RoutingMode` in `src/config/types.rs`:

- `Regular`: one worker pool; each request selects one backend worker.
- `PrefillDecode`: separate prefill and decode worker pools; the router selects one worker from each pool and coordinates a disaggregated request.

PD mode is intended for prefill/decode disaggregated LLM serving. For HTTP PD routing, `src/routers/http/pd_router.rs` handles backend-specific metadata. It contains SGLang-style PD behavior and vLLM/Mooncake support through cached prefill bootstrap data from each prefill worker's `/query` endpoint.

For gRPC PD routing, `src/routers/grpc/pd_router.rs` delegates to the same gRPC request pipeline as regular mode but configures it for dual dispatch.

### HTTP and gRPC Backends
Router creation is centralized in `src/routers/factory.rs`. The selected implementation depends on two config dimensions:

| Connection mode | Routing mode | Router implementation |
| --- | --- | --- |
| HTTP | Regular | `src/routers/http/router.rs` |
| HTTP | PrefillDecode | `src/routers/http/pd_router.rs` |
| gRPC | Regular | `src/routers/grpc/router.rs` |
| gRPC | PrefillDecode | `src/routers/grpc/pd_router.rs` |

HTTP mode primarily behaves as a proxy. It serializes the incoming request, forwards selected headers, streams or buffers backend responses, and preserves response headers where appropriate.

gRPC mode is more native to Mesh. It tokenizes and prepares requests inside Rust, selects workers, acquires gRPC clients, builds worker requests, executes them, and post-processes streaming or non-streaming responses.

## Request Flow
### Startup Flow
`src/server.rs::startup` is the main runtime bootstrap:

1. Initialize logging and Prometheus metrics.
2. Build `AppContext` from `RouterConfig`.
3. Create a `JobQueue` and typed workflow engines.
4. Optionally submit a startup tokenizer job from `--tokenizer-path` or `--model-path`.
5. Submit worker initialization jobs based on `worker_urls` or PD `prefill`/`decode` config.
6. Wait until expected workers are registered or startup timeout is reached.
7. Create `RouterManager`, which creates the concrete router through `RouterFactory`.
8. Start worker health checking unless disabled.
9. Start load monitoring for policies that need load data.
10. Start concurrency queue processor if rate limiting with queue is configured.
11. Build the Axum app and serve it with graceful shutdown.

### HTTP Request Flow
For normal HTTP routing:

1. Axum route in `src/server.rs` parses a typed request.
2. Handler calls `state.router.route_*`.
3. `RouterManager` delegates to the default concrete router.
4. `src/routers/http/router.rs` extracts routing text, picks a worker using `PolicyRegistry`, and forwards the request to the mapped backend endpoint.
5. `RetryExecutor` retries retryable responses such as 408, 429, 500, 502, 503, and 504.
6. Worker load and circuit-breaker state are updated through worker guards and outcome recording.

HTTP PD routing follows the same outer flow but uses `src/routers/http/pd_router.rs` to select prefill and decode workers, inject PD metadata, and handle backend-specific prefill/decode coordination.

### gRPC Request Flow
gRPC mode is organized around `RequestPipeline` in `src/routers/grpc/pipeline.rs`.

Regular gRPC mode uses these stages:

1. `PreparationStage`: normalize request data and tokenize input.
2. `WorkerSelectionStage`: select one regular gRPC worker by model, health, circuit breaker, and routing policy.
3. `ClientAcquisitionStage`: acquire or create the worker gRPC client.
4. `RequestBuildingStage`: build the gRPC request.
5. `DispatchMetadataStage`: attach dispatch metadata.
6. `RequestExecutionStage`: execute a single-worker request.
7. `ResponseProcessingStage`: convert backend output into OpenAI/SGLang-compatible responses, including reasoning and tool-call parsing.

PD gRPC mode uses the same pipeline shape, but `WorkerSelectionStage` selects a prefill/decode pair and `RequestExecutionStage` runs dual dispatch.

## Core Modules
### `src/config`
Configuration types and builder logic. Important concepts include:

- `RouterConfig`: top-level runtime config.
- `RoutingMode`: regular versus prefill/decode.
- `ConnectionMode`: HTTP versus gRPC.
- `PolicyConfig`: random, round-robin, cache-aware, power-of-two, and prefix-hash policies.
- `RetryConfig`, `CircuitBreakerConfig`, `HealthCheckConfig`, and tokenizer cache settings.

The CLI in `src/main.rs` maps command-line flags into these config structures.

### `src/app_context.rs`

`AppContext` is the shared dependency container. It owns or references:

- `reqwest::Client`
- router config
- rate limiter
- tokenizer registry
- reasoning and tool parser factories
- worker registry
- policy registry
- response and conversation storage
- load monitor
- job queue and workflow engines
- in-flight request tracker

This object is passed into router creation and request handling so the hot path does not repeatedly rebuild shared services.

### `src/core`

Core runtime primitives:

- `worker.rs`: worker trait, regular worker implementation, DP-aware worker behavior, load tracking, worker metadata, and health checks.
- `worker_registry.rs`: central registry with model-based indexing, worker-type indexing, connection-mode indexing, URL-to-ID lookup, and per-model consistent hash rings.
- `worker_manager.rs`: fan-out utilities for cache flush, load collection, and engine metrics.
- `job_queue.rs` and `steps/`: async worker/tokenizer registration workflows.
- `retry.rs`: retry executor with exponential backoff and jitter.
- `circuit_breaker.rs`: lock-free per-worker circuit breaker using atomic state.
- `token_bucket.rs`: rate limiting primitive.
- `metrics_aggregator.rs`: aggregation of worker Prometheus metrics.

### `src/routers`

Router implementations and helper endpoints:

- `mod.rs`: `RouterTrait`, the common interface for all router implementations.
- `factory.rs`: concrete router selection by routing and connection mode.
- `router_manager.rs`: single-router manager that delegates requests to the configured router.
- `http/router.rs`: regular HTTP proxy router.
- `http/pd_router.rs`: HTTP PD router for prefill/decode deployments.
- `grpc/router.rs`: regular gRPC router.
- `grpc/pd_router.rs`: gRPC PD router.
- `grpc/pipeline.rs`: staged request execution pipeline.
- `grpc/common/stages`: reusable pipeline stages for preparation, selection, client acquisition, metadata, and execution.
- `grpc/regular`: regular gRPC request and response processing.
- `conversations`, `parse`, `tokenize`: non-routing API handlers for Responses API state, parser utilities, and tokenizer operations.

### `src/policies`

Load balancing policy implementations. Each policy implements `LoadBalancingPolicy`:

- `random`: uniform selection among healthy workers.
- `round_robin`: sequential selection with counters.
- `cache_aware`: prefix-tree based routing to improve KV cache reuse.
- `power_of_two`: samples two workers and chooses the less loaded one.
- `prefix_hash`: hashes a token prefix onto a per-model consistent hash ring and walks the ring when workers are overloaded.

The policy layer receives request text, token IDs, headers, and optional hash-ring snapshots through `SelectWorkerInfo`.

### `src/middleware.rs`

HTTP middleware and Tower layers:

- Request ID generation and propagation.
- Structured request/response logging.
- HTTP metrics layer and in-flight tracking.
- Concurrency limiting with a token bucket.
- Optional queueing when concurrency tokens are exhausted.
- `TokenGuardBody`, which returns concurrency tokens only after a streaming response body is consumed or dropped.

### `src/observability`

Metrics, logging, events, and in-flight tracking:

- `metrics.rs` defines Prometheus metrics for HTTP, router, inference, worker pools, circuit breakers, retries, worker health, and routing decisions.
- `logging.rs` initializes structured tracing and optional file logging.
- `inflight_tracker.rs` samples active request age buckets.
- `events.rs` provides structured event helpers for routing and worker actions.

## Routing and Worker Selection
Workers are represented through the `Worker` trait in `src/core/worker.rs`. A worker carries:

- URL and optional API key.
- Type: regular, prefill, or decode.
- Connection mode: HTTP or gRPC.
- Runtime metadata, model ID, labels, priority, and cost.
- Health state and circuit breaker.
- Current load and routing-key load.

`WorkerRegistry` keeps multiple indexes so request-time selection stays cheap:

- all workers by generated ID;
- model ID to immutable worker snapshot;
- type to worker IDs;
- connection mode to worker IDs;
- URL to worker ID;
- per-model consistent hash ring.

The gRPC worker selection stage filters by model, worker type, connection mode, health, and circuit-breaker availability before invoking the configured policy.

## Reliability Behavior

Mesh includes several reliability mechanisms:

- Retries: `RetryExecutor` retries retryable HTTP status codes with exponential backoff, max backoff, and jitter.
- Circuit breakers: each worker has an atomic closed/open/half-open state machine. Failed outcomes can open the circuit; timeout transitions to half-open; successful probes close it again.
- Health checks: background health checks periodically update worker health unless disabled.
- Rate limiting: protected inference routes use a token bucket when `max_concurrent_requests` is enabled.
- Queueing: if configured, requests that cannot acquire a concurrency token can wait in a bounded queue until timeout.
- RAII load guards: worker load counters are decremented when response bodies complete or are dropped, which matters for streaming and client disconnects.

## Observability
Mesh exports Prometheus metrics by default on the configured metrics host/port. Metrics cover:

- HTTP request counts, response counts, latency, active connections, and rate limiting.
- Router request counts, latency, errors, upstream responses, and gRPC stage duration.
- gRPC inference metrics such as TTFT, TPOT, token counts, and generation duration.
- Worker pool size, worker selection, health, circuit-breaker state/transitions, retries, and backoff.

The metrics code interns dynamic labels such as model IDs and worker URLs to reduce allocation overhead in hot paths.

## Tests and Scripts
`tests/README.md` describes the integration test layout. The tests are grouped around:

- API endpoint behavior.
- Routing policies and PD topologies.
- Reliability features such as retries, rate limiting, circuit breakers, and worker failures.
- Security/auth behavior.
- Protocol/spec compatibility.
- Standalone load guard, in-flight tracker, and metrics aggregation tests.

The `scripts/` folder contains deployment and benchmark helpers for standalone, prefill/decode, vLLM, SGLang, and Slurm-based launches.

## Notes and Caveats
- `atom/mesh/README.md` is already a user-facing quickstart. This document is a code-structure analysis with more emphasis on internal responsibilities and request flow.
- The config model includes `api_key`, and workers can store/forward API keys. When relying on router-level authentication, verify the currently wired middleware path because the visible `build_app` route construction primarily shows concurrency limiting, request ID, body limit, logging, and metrics layers.
- The codebase has both HTTP and gRPC implementations for regular and PD modes. When changing routing behavior, update the corresponding mode-specific implementation and sweep the equivalent mode to avoid drift.
- gRPC mode depends on tokenizer, reasoning-parser, and tool-parser components being available in `AppContext`; startup configuration should be checked before enabling gRPC paths in deployment.

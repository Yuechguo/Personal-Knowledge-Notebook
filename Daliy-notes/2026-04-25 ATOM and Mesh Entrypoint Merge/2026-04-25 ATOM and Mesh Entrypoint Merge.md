# ATOM and Mesh
base PR：[[ATOM_MESH] PD disaggregation router with multi-node support by Jasen2201 · Pull Request #502 · ROCm/ATOM](https://github.com/ROCm/ATOM/pull/502)

## Router impl for ATOM and Atomesh
### ATOM FastAPI 
``` python
def main():
...
    global engine
	engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine(tokenizer=tokenizer)
...
    uvicorn.run(app, host=args.host, port=args.server_port)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    logger.info("Server started successfully and ready to accept requests")
    yield
    logger.info("Server shutting down, releasing resources...")
    if engine is not None:
        engine.close()

app = FastAPI(title="ATOM OpenAI API Server", lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
	global engine, tokenizer, model_name
...
    seq = engine.io_processor.preprocess(...)
    
class ChatCompletionRequest(BaseModel):
    """Request model for chat completions (OpenAI-compatible)."""

    model_config = {"extra": "ignore"}

    model: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    prompt: Optional[List[ChatMessage]] = None  # Accept 'prompt' as alias
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_k: Optional[int] = DEFAULT_TOP_K
    top_p: Optional[float] = DEFAULT_TOP_P
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    seed: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = (
        None  # "auto", "none", "required", or {function: {name}}
    )
    # Accepted for compatibility, not actively used:
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1

    def get_messages(self) -> List[ChatMessage]:
        """Get messages from either 'messages' or 'prompt' field."""
        if self.messages is not None:
            return self.messages
        elif self.prompt is not None:
            return self.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' field is required"
```

### Atomesh Axum
```rust
pub fn build_app(
    app_state: Arc<AppState>,
    auth_config: AuthConfig,
    control_plane_auth_state: Option<crate::auth::ControlPlaneAuthState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Router {
    let protected_routes = Router::new()
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
...

// state = app_state, global var
async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    state
        .router
        .route_chat(Some(&headers), &body, Some(&body.model))
        .await
}
...
let router_manager = RouterManager::from_config(&config, &app_context).await?;
let router: Arc<dyn RouterTrait> = router_manager.clone();
let app_state = Arc::new(AppState {
	router, // router in app_state 
	context: app_context.clone(),
	concurrency_queue_tx: limiter.queue_tx.clone(),
	router_manager: Some(router_manager),
	mesh_handler,
	mesh_sync_manager,
});

...
impl RouterManager {
...
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // In IGW mode, resolve model_id and fail fast if not resolvable
        // In non-IGW mode, pass through to router (router handles validation)
        let effective_model_id = if self.enable_igw {
            // Use provided model_id or fall back to body.model
            let model = model_id.or(Some(&body.model));
            match self.resolve_model_id(model) {
                Ok(id) => Some(id),
                Err(err_response) => return *err_response,
            }
        } else {
            None
        };

        // openai-router http_router pd_router
        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id)); // 

        if let Some(router) = router {
            router
                .route_chat(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

impl PDRouter {
...
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            body.messages.first().and_then(|msg| match msg {
                ChatMessage::User { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::Developer { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::System { content, .. } => Some(content.to_simple_string()),
                _ => None,
            })
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

```

## Atomesh and ATOM Code Base Merge
### Comprehensive decoupling(current PR state)
**Retain the respective service frameworks of ATOM and MESH, with a unified startup entry point. The role is fixed at startup. **
**ATOM and MESH have their own implementations of entrypoints, which need to be maintained separately.**
![](./Pasted%20image%2020260430100903.png)

#### **Atom standalone mode**
```python 
# start atom
python -m atom.entrypoints.server ... （default for ATOM standalone）
```
**There is no difference from the current usage of ATOM.**
![](./Pasted%20image%2020260430101743.png)
#### **ATOM/ATOMESH PD Mode**
```python 
# start MESH mode
python -m atom.entrypoints.server
    ...
	--mesh_only 
	--prefill "${p_ip}:${p_port} ${BOOTSTRAP_PORT}"
	--decode "${d_ip}:${d_port} ${BOOTSTRAP_PORT}"
	
#start ATOM prefill node
python -m atom.entrypoints.server
	...
	--disaggregation-mode prefill
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    --disaggregation-ib-device "${IB_DEVICE}"
#start ATOM decode node
python -m atom.entrypoints.server
	...
	--disaggregation-mode decode
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    --disaggregation-ib-device "${IB_DEVICE}"
```

#### **gRPC impl in ATOM entrypoints**
The current ATOM only has HTTP RESTful APIs. We need to implement gRPC network access in the future.
The main purpose of the gRPC mode is to address situations where access requests involve tokenized data with a large volume, and the traditional web-service JSON/XML byte sequences take up too many bytes.
``` python
#sglang grpc data proto defination
syntax = "proto3";
message GenerateRequest {
  string request_id = 1;

  // Input must be tokenized (no raw text)
  TokenizedInput tokenized = 2;

  // Multimodal inputs
  MultimodalInputs mm_inputs = 3;

  // Generation parameters
  SamplingParams sampling_params = 4;

  // Return options
  bool return_logprob = 5;
  int32 logprob_start_len = 6;
  int32 top_logprobs_num = 7;
  repeated uint32 token_ids_logprob = 8;
  bool return_hidden_states = 9;

  // For disaggregated serving
  DisaggregatedParams disaggregated_params = 10;

  // Custom logit processor (serialized)
  string custom_logit_processor = 11;

  // Request metadata
  google.protobuf.Timestamp timestamp = 12;
  bool log_metrics = 13;

  // Input embeddings (alternative to text/tokens)
  repeated float input_embeds = 14;

  // LoRA adapter ID (if pre-loaded)
  string lora_id = 15;

  // Data parallel routing
  int32 data_parallel_rank = 16;

  // Whether client wants streaming response
  bool stream = 17;
}
```

ATOM impl grpc pseudo code:
``` python
import grpc
# Create gRPC server
server = grpc.aio.server(
	futures.ThreadPoolExecutor(max_workers=10),
	options=[
		("grpc.max_send_message_length", 1024 * 1024 * 256),
		("grpc.max_receive_message_length", 1024 * 1024 * 256),
	],
)

# Add service
servicer = ATOMSchedulerServicer(
	request_manager=request_manager,
	server_args=server_args,
	model_info=model_info,
	scheduler_info=scheduler_info,
	server=server,
)

# Start server
listen_addr = f"{server_args.host}:{server_args.port}"
server.add_insecure_port(listen_addr)

def ATOMSchedulerServicer():
...
	async def Generate(
		self,
		request:GenerateRequest,
	):
		# Convert gRPC request to internal format
        tokenized_req = self._convert_generate_request(request)
...
		global engine
		seq = engine.io_processor.preprocess(tokenized_req)
```
**The gRPC service and the FastAPI HTTP service are parallel.**

### ATOM and Atomesh use the same Axum for web-server
![](./Pasted%20image%2020260429121929.png)

**we need to impl a atom_standalone_router in mesh Axum**
``` rust
impl AtomStandaloneRouter {
	pub async fn new (ctx: &Arc<AppContext>) -> Result<Self, String> {
		global engine, tokenizer
		engine_args = EngineArgs.from_cli_args(AppContext.args)
	    engine = engine_args.create_engine(tokenizer=AppContext.tokenizer)
	}
...	
	async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
...
        tokenizer = tokenizer.trans(context)
		seq = engine.io_processor.preprocess(...)
...
		return ...	
    }

use pyo3::prelude::*;
use pyo3::types::PyList;
fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let sys = py.import_bound("sys")?;
        let path: Bound<'_, PyList> = sys.getattr("path")?.downcast_into()?;
        path.insert(0, "/path/to/engine/package")?;
        let engine_mod = py.import_bound("engine")?;
        let Engine = engine_mod.getattr("Engine")?;
        // init engine instance
        let inst = Engine.call1(("demo",))?;
        // call engine run method
        let y: i32 = inst.call_method1("run", (21,))?.extract()?;
        println!("run -> {}", y);
        // call other method
        let s: String = inst.call_method0("greet")?.extract()?;
        println!("{}", s);
        Ok(())
    })
}
```
**Upon creation of AtomStandaloneRouter, an Atom Engine instance is instantiated, and all router calls are ultimately forwarded to the engine's I/O implementation.** 
**Any user modifying or adding entrypoints must modify both `entrypoints.rs` in the mesh and the `AtomStandaloneRouter` implementation — though all modifications are done in Rust.**

#### **Atomesh/atom standalone mode**
```python
#only start one process
python -m atom.entrypoints.server ... (default for standalone mode)
```
The red line represents the actual request data flow.
![](./Pasted%20image%2020260429213526.png)

#### **Atomesh/atom pd mode**
``` python
#start mesh node
python -m atom.entrypoints.server
    ... 
	--mesh_only 
	--prefill "${p_ip}:${p_port} ${BOOTSTRAP_PORT}"
	--decode "${d_ip}:${d_port} ${BOOTSTRAP_PORT}"
	
#start prefill node
python -m atom.entrypoints.server （default for ATOM engine path）
	...
	--disaggregation-mode prefill
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    --disaggregation-ib-device "${IB_DEVICE}"
#start decode node
python -m atom.entrypoints.server （default for ATOM engine path）
	...
	--disaggregation-mode decode
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    --disaggregation-ib-device "${IB_DEVICE}"
```
**When starting with MESH_ONLY, atomesh/atom only includes the routing part, and the Atom engine will not be started. The routing part serves as a mesh distribution/dispatch layer.**
**When a P-node or D-node starts, atomesh/atom essentially just bypasses the request to the Atom engine, and the actual model runs through the Atom engine.**
![](./Pasted%20image%2020260429215041.png)

#### **With Sglang(Sglang-ATOM) or vLLM(Sglang-ATOM)**
``` python 
#start mesh node
python -m atom.entrypoints.server
    ... 
	--mesh_only 
	--prefill "${p_ip}:${p_port} ${BOOTSTRAP_PORT}"
	--decode "${d_ip}:${d_port} ${BOOTSTRAP_PORT}"
	
#start sglang prefill node
python3 -m sglang.launch_server
	...
	--disaggregation-mode prefill
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    --disaggregation-ib-device "${IB_DEVICE}"
#start sglang decode node
python3 -m sglang.launch_server
	...
	--disaggregation-mode decode
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    --disaggregation-ib-device "${IB_DEVICE}"
```

![](./Pasted%20image%2020260429220720.png)

#### **Project file hierarchy**
Ideal file hierarchy structure
```shell
project_root
|-----atom
|    |-----entrypoints
|    |     |--server.py             # python warper, rust axum app starter
|    |     |--atomesh_server.rs     # main http or grpc api for endpoints      
|    |     |--atomesh_appcontext.rs # app context for axum web server
|    |     |--atom_standalone_router.rs # atom engine create, engine usage
|    |
|    |-----mesh
|    |     |--src
|    |     |...
|    |
```
#### **gRPC impl in mesh**
**A minimal gRPC server needs to be implemented using Rust's native Tonic.**
``` rust
mod server;

use tonic::transport::Server;
use server::{MyGreeter, GreeterServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let greeter = MyGreeter::default();

    println!("gRPC server listening on {}", addr);

    Server::builder()
        .add_service(GreeterServer::new(greeter))
        .serve(addr)
        .await?;

    Ok(())
}
```

#### **Abnormal termination**
``` shell
[yuechguo@hjbog-srdc-18 ~]$ kill -l
 1) SIGHUP       2) SIGINT       3) SIGQUIT      4) SIGILL       5) SIGTRAP
 2) SIGABRT      7) SIGBUS       8) SIGFPE       9) SIGKILL     10) SIGUSR1
3) SIGSEGV     12) SIGUSR2     13) SIGPIPE     14) SIGALRM     15) SIGTERM
4) SIGSTKFLT   17) SIGCHLD     18) SIGCONT     19) SIGSTOP     20) SIGTSTP
5) SIGTTIN     22) SIGTTOU     23) SIGURG      24) SIGXCPU     25) SIGXFSZ
6) SIGVTALRM   27) SIGPROF     28) SIGWINCH    29) SIGIO       30) SIGPWR
7) SIGSYS      34) SIGRTMIN    35) SIGRTMIN+1  36) SIGRTMIN+2  37) SIGRTMIN+3
8) SIGRTMIN+4  39) SIGRTMIN+5  40) SIGRTMIN+6  41) SIGRTMIN+7  42) SIGRTMIN+8
9) SIGRTMIN+9  44) SIGRTMIN+10 45) SIGRTMIN+11 46) SIGRTMIN+12 47) SIGRTMIN+13
10) SIGRTMIN+14 49) SIGRTMIN+15 50) SIGRTMAX-14 51) SIGRTMAX-13 52) SIGRTMAX-12
11) SIGRTMAX-11 54) SIGRTMAX-10 55) SIGRTMAX-9  56) SIGRTMAX-8  57) SIGRTMAX-7
12) SIGRTMAX-6  59) SIGRTMAX-5  60) SIGRTMAX-4  61) SIGRTMAX-3  62) SIGRTMAX-2
13) SIGRTMAX-1  64) SIGRTMAX
```
Common forced termination signals
``` shell
kill -9：SIGKILL => Signal cannot be caught
kill -15：SIGTERM => Signal can be caught
Ctrl+C：SIGINT (2) = kill -2  => Signal can be caught
```
SIGKILL prevents both the Rust process and the Python process from exiting gracefully.

``` rust
1.rust abnormal exit // kill -9, process coredump 
2.python abnormal exit // kill -9, process coredump
=>
the whole project processes can be exiting gracefully.

Rust atom-mesh
  └── embedded Python LLMEngine
        └── multiprocessing EngineCore
rust are parent process which python process is child process.

easy impl : rust abnormal exit => python gracefully shutdown
```

``` rust
standalone mode：
Python interface and supervisor
  ├── Rust atom-mesh
  └── EngineCore

mesh mode：
Python interface and supervisor
  ├── Rust atom-mesh
```
1.python interface and supervisor abnormal exit => send SIGTERM to rust and python, which rust and python can both shotdown gracefully.
2.A resident thread monitors both Rust and Python; if either one encounters an issue, the entire system performs a graceful exit.
The downside is that an additional IPC channel is required between Rust and Python.

### ATOM and Atomesh keep own web-server framework
#### 1. ATOM request route to  Atomesh server
The processes of ATOM and Atomesh starting their respective web servers, but the unified entrypoint is atom.entrypoints.
![](./Pasted%20image%2020260429135631.png)

```python
# only start atom instance for atom standalone
python -m atom.entrypoints.server ... (default for atom standalone)

# start atom and atomesh instance, but atom instance only route all request to atomesh entrypoints
python -m atom.entrypoints.server ... --mesh_only

# start atom and atomesh instance, local request directly call atom_engine, other route to atomesh, But if atom engine is in PD mode, we need new api for pd metadata sync
python -m atom.entrypoints.server ... --hybird
```

```python
python -m atom.entrypoints.server ... --hybird

def main():
...
	local_egine_ip, local_egine_port = ...
	atomesh_ip, atomesh_port = ..
...

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
	global engine, tokenizer, model_name
...
    if local:
        seq = engine.io_processor.preprocess(...)
    else:
	    atomesh_context = ...
	    atomesh.router.to(atomesh_context, atomesh_ip, atomesh_port)
...
```
#### 2. Atom and Atomesh share the same entrypoints/endpoints
**When the server starts, the role is already determined, whether it is ATOM's standalone mode or Atomesh mode.**
![](./Pasted%20image%2020260429141819.png)

**Both roles share the same endpoints entry.**
``` python
# only start atom instance
python -m atom.entrypoints.server ... (default for atom standalone)

#only atomesh instance
python -m atom.entrypoints.server ... --mesh_only
```

**All the main interfaces of Axum on Rust need to be made into Python-compatible interfaces.**
**For each HTTP service interface, the Python approach cannot be directly reused. For each Axum HTTP endpoint, it is estimated that separate development is required.**
``` python
if mesh_only:
  import py-axum
else:
  import fastapi
  app = FastAPI(title="ATOM OpenAI API Server", lifespan=lifespan)

def main():
...
	if mesh_only:
		axum_app = py-axum.Router.New.route(
		"/v1/chat/completions", post(axum_chat_completions)
		);
	else :
	    global engine
		engine_args = EngineArgs.from_cli_args(args)
	    engine = engine_args.create_engine(tokenizer=tokenizer)
...
	if mesh_only:
		axum_app.app.start()
	else:
	    uvicorn.run(app, host=args.host, port=args.server_port)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
	global engine, tokenizer, model_name
...
    seq = engine.io_processor.preprocess(...)

async def axum_chat_completions(
	appstate: state, 
	HeaderMap: headers: http::HeaderMap, 
	request: ChatCompletionRequest):
	return state.router.route_chat().await
```
**Exposing and integrating Python interfaces for debugging is a very large undertaking.**

# Web Framework Benchmark  with Techempower
https://www.techempower.com/benchmarks
TechEmpower Web Benchmarks采用多维度的评测指标，全面衡量Web框架在不同应用场景下的性能表现。Round 23评测包含以下六项核心测试：
1. **[JSON序列化](https://zhida.zhihu.com/search?content_id=255191350&content_type=Article&match_order=1&q=JSON%E5%BA%8F%E5%88%97%E5%8C%96&zhida_source=entity)(JSON Serialization)**：测试框架序列化简单JSON对象的能力
2. **[单数据查询](https://zhida.zhihu.com/search?content_id=255191350&content_type=Article&match_order=1&q=%E5%8D%95%E6%95%B0%E6%8D%AE%E6%9F%A5%E8%AF%A2&zhida_source=entity)(Single Query)**：测试框架执行单次数据库查询的性能
3. **[多数据查询](https://zhida.zhihu.com/search?content_id=255191350&content_type=Article&match_order=1&q=%E5%A4%9A%E6%95%B0%E6%8D%AE%E6%9F%A5%E8%AF%A2&zhida_source=entity)(Multiple Queries)**：测试框架执行多次数据库查询的能力
4. **[数据更新](https://zhida.zhihu.com/search?content_id=255191350&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E6%9B%B4%E6%96%B0&zhida_source=entity)(Data Updates)**：测试框架执行数据库更新操作的性能
5. **[纯文本](https://zhida.zhihu.com/search?content_id=255191350&content_type=Article&match_order=1&q=%E7%BA%AF%E6%96%87%E6%9C%AC&zhida_source=entity)(Plaintext)**：测试框架处理简单文本响应的性能
6. **[模板渲染](https://zhida.zhihu.com/search?content_id=255191350&content_type=Article&match_order=1&q=%E6%A8%A1%E6%9D%BF%E6%B8%B2%E6%9F%93&zhida_source=entity)(Fortunes)**：测试框架结合数据库查询和HTML模板渲染的性能

### Rust Axum and Python FastAPI
#### JSON Serialization and Plaintext 参考价值较高
JSON Serialization 描述的是服务框架本身提供服务时候接口通过json传输http数据的时候的序列化/反序列化的能力，是服务框架的基础接口能力。
Plaintext 描述的是服务框架本身处理字符的能力，测试方案中有一些基础的字符处理的方法，包括各种各样的正则匹配方法等，对应是Rust和Python的原生正则方法的实现。
![](./Pasted%20image%2020260507131911.png)
![](./Pasted%20image%2020260507132009.png)

结论：Axum的性能优势还是非常明显的。

### Single Query,  Multiple Queries and Data Updates 有一定参考价值
Single Query,  Multiple Queries and Data Updates测试主要是面向特定的服务框架+语言生态下的常用的数据sql的访问场景。
主要是反应了web本身性能+框架语言生态下的常用的web服务能力（数据库的增删改查）
![](./Pasted%20image%2020260507132523.png)
![](./Pasted%20image%2020260507132543.png)
![](./Pasted%20image%2020260507132606.png)

结论：Axum +Rust生态下的数据库能提供的服务能力也是明显高于FastAPI+python的生态的。

### Fortunes 测试参考较小
Fortunes测试是结合了数据库查询和HTML的渲染能力，和我们的atomesh的应用关系不大，参考价值不太。
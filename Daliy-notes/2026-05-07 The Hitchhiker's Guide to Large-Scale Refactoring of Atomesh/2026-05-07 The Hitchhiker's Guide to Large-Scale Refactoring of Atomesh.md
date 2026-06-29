当前的目录结构
``` shell
atom/mesh
+---scripts
+---src
|   +---config
|   +---core
|   |   \---steps
|   |       \---worker
|   |           +---local
|   |           \---shared
|   +---observability
|   +---policies
|   \---routers
|       +---conversations
|       +---grpc
|       |   +---common
|       |   |   +---responses
|       |   |   \---stages
|       |   \---regular
|       |       +---responses
|       |       \---stages
|       |           +---chat
|       |           \---generate
|       +---http
|       +---parse
|       \---tokenize
\---tests
    +---api
    +---common
    +---reliability
    +---routing
    +---security
    \---spec
```
5月份target： entrypoints and routers重构
entrypoints重构按照之前讨论的方案。
#### routers文件夹功能
##### 顶层文件
- `mod.rs`：定义统一的 `RouterTrait`，所有 HTTP/gRPC/PD router 都实现它。
- `factory.rs`：根据 `RoutingMode` 和 `ConnectionMode` 创建具体 router：HTTP regular、HTTP PD、gRPC regular、gRPC PD。
- `router_manager.rs`：router 管理器，目前是 single-router 模式，负责保存默认 router 并转发请求。
- `error.rs`：统一构造 router 层错误响应，比如 `bad_request`、`bad_gateway`、`service_unavailable`。
- `header_utils.rs`：请求/响应 header 转发工具，过滤 hop-by-hop header。
- `persistence_utils.rs`：Responses API 和 conversation item 的持久化工具。
- `conversations/handlers.rs`：`/v1/conversations` 的 CRUD handler。
- `parse/handlers.rs`：解析模型输出中的 tool call 和 reasoning 内容。
- `tokenize/handlers.rs`：`/v1/tokenize`、`/v1/detokenize`、tokenizer 管理接口。
##### HTTP 路由
- `http/router.rs`：普通 HTTP router。选一个 regular worker，把 chat/completions/generate/responses 请求转发过去，支持 retry、streaming、header 转发和负载 guard。
- `http/pd_router.rs`：HTTP Prefill-Decode router。一次请求会选择 prefill worker 和 decode worker，并处理 bootstrap room、vLLM Mooncake metadata、PD streaming/non-streaming。
- `http/pd_types.rs`：PD router 的辅助类型，比如 bootstrap request wrapper、room id、PD 选择策略、URL 拼接。
##### gRPC 顶层
- `grpc/router.rs`：普通 gRPC router。实现 chat/generate/completions/responses，核心通过 `RequestPipeline::new_regular` 跑 pipeline。
- `grpc/pd_router.rs`：gRPC PD router。使用 `RequestPipeline::new_pd`，支持 prefill/decode 双 dispatch。
- `grpc/client.rs`：统一 gRPC client wrapper，屏蔽 SGLang/vLLM backend 差异。
- `grpc/proto_wrapper.rs`：统一封装 SGLang/vLLM 的 proto request、response、stream、embed 类型。
- `grpc/context.rs`：gRPC pipeline 中流转的 `RequestContext`，保存请求输入、阶段产物、worker/client、proto request、执行结果。
- `grpc/pipeline.rs`：pipeline 编排器，把 preparation、worker selection、client acquisition、request building、execution、response processing 串起来。
- `grpc/completion_adapter.rs`：把 OpenAI `/v1/completions` 转成 generate 请求，并把 generate response 转回 completion response。
- `grpc/utils.rs`：gRPC 大杂烩工具：tokenizer 解析、chat template、tool constraints、stream 收集、logprobs 转换、parser 获取、metrics endpoint label 等。
##### gRPC Common
- `grpc/common/stages/worker_selection.rs`：按策略选择 single worker 或 PD 的 prefill/decode worker。
- `grpc/common/stages/client_acquisition.rs`：从选中的 worker 取 gRPC client。
- `grpc/common/stages/dispatch_metadata.rs`：生成 request id、model、created、weight version 等 dispatch metadata。
- `grpc/common/stages/request_execution.rs`：真正发起 gRPC 请求，支持 single 和 dual dispatch。
- `grpc/common/stages/helpers.rs`：公共 stage helper，比如注入 PD bootstrap metadata。
- `grpc/common/response_collection.rs`：收集 gRPC stream 响应，PD 时可合并 prefill input logprobs 到 decode response。
- `grpc/common/response_formatting.rs`：构造 usage、chat completion response 等公共格式化逻辑。
- `grpc/common/responses/context.rs`：Responses API handler 共用上下文。
- `grpc/common/responses/handlers.rs`：`GET /v1/responses/{id}` 和 cancel 的公共实现。
- `grpc/common/responses/streaming.rs`：Responses API streaming SSE 事件生成器。
- `grpc/common/responses/utils.rs`：Responses API 工具函数，比如提取 tools、按 `store=true` 持久化响应。
#####  gRPC Regular
- `grpc/regular/processor.rs`：普通和 PD gRPC router 共用的响应处理器。
- `grpc/regular/streaming.rs`：chat/generate 的 streaming SSE 处理，支持 tool parser、reasoning parser、stop decoder、usage/logprobs。
- `grpc/regular/stages/preparation.rs`：根据请求类型分发到 chat/generate preparation。
- `grpc/regular/stages/request_building.rs`：根据请求类型分发到 chat/generate request building。
- `grpc/regular/stages/response_processing.rs`：根据请求类型分发到 chat/generate response processing。
- `grpc/regular/stages/chat/preparation.rs`：chat 请求预处理：tool filtering、chat template、tokenize、constraints。
- `grpc/regular/stages/chat/request_building.rs`：把 chat 请求构造成后端 proto `GenerateRequest`。
- `grpc/regular/stages/chat/response_processing.rs`：chat response 处理，区分 streaming 和 non-streaming。
- `grpc/regular/stages/generate/preparation.rs`：generate 请求预处理：解析 text/input_ids、tokenize、stop decoder。
- `grpc/regular/stages/generate/request_building.rs`：把 generate 请求构造成后端 proto request。
- `grpc/regular/stages/generate/response_processing.rs`：generate response 处理，区分 streaming 和 non-streaming。
#####  Responses API
- `grpc/regular/responses/handlers.rs`：`POST /v1/responses` 入口，决定走同步还是 streaming。
- `grpc/regular/responses/non_streaming.rs`：非流式 Responses API：加载上下文、转 chat、执行 pipeline、转回 responses、持久化。
- `grpc/regular/responses/streaming.rs`：把 chat SSE stream 转成 Responses API SSE stream，并累计最终状态用于持久化。
- `grpc/regular/responses/conversions.rs`：`ResponsesRequest <-> ChatCompletionRequest/Response` 转换。
- `grpc/regular/responses/common.rs`：Responses API 共享 helper，比如加载 conversation history / previous response chain。

#### 重构点1 ：smart routers统一
```shell

atom/mesh/src/core -> atom/mesh/src/workers
atom/mesh/src/polices -> atom/mesh/src/polices
atom/mesh/src/routers -> atom/mesh/src/routers

atom/mesh/src/routers/http
atom/mesh/src/routers/grpc
-> 
atom/mesh/src/routers/smart_router
    /grpc # grpc stuff
	mod.rs
	smart_http_router.rs
	smart_http_pd_router.rs
	smart_grpc_router.rs
	smart_grpc_pd_router.rs
```
按功能提取出smart routers的功能, 统一现在routers的实现。
5月底目标：功能抽取，代码精简，文件合并统一。
长远的目标：配合重构后的worker pool，重构后的router_manger的结构，router变成一个协议实现层，不再绑定资源，只是一层执行的impl。

#### 重构点2：worker pool的统一
``` shell
当前启动模式，启动项和api访问注册：
./atomesh_bin --prefill xxx --decode xxx # pd模式
./atomesh_bin --regular xx # 普通路由转发模式

// Build worker routes
let worker_routes = Router::new()
	.route("/workers", post(create_worker).get(list_workers_rest))
	.route(
		"/workers/{worker_id}",
		get(get_worker).put(update_worker).delete(delete_worker),
	);
```
当前启动的时候，根据输入的args确定pd节点的worker，或者regular模式的节点。
新增功能worker信息的ETCD注册方式（可以后期实现）：
``` rust
// 听过ETCD监听得到
use etcd_client::{Client, WatchOptions};
let client = Client::connect([${etcd_addr}], None).await?;
let mut watcher = client.watch_client().await?;

let (mut stream, mut canceller) = watcher.watch(key, Some(options)).await?;
 
while let Some(event_result) = stream.message().await? {
        for event in event_result.events() {
            match event.event_type() {
                etcd_client::EventType::Put => {
                    worker_pool.add_worker(kv);
                }
                etcd_client::EventType::Delete => {
                    let kv = event.kv().unwrap();
                    worker_pool.remove_worker(kv);
                }
            }
        }
    }
```

![](./Pasted%20image%2020260513110700.png)
worker_pool 分为两个 regular_worker_pool, pd_worker_pool
其中每个pool各自有一个sub worker pool去维护http链接或者grpc链接下的对应的worker 节点。
5月份target，实现worker pool的分类管理


#### 重构点3：router_manger的统一
当前的mesh在启动的时候，会自动分配router角色，根据config的配置。
当前的request path：
``` rust
request ->  routers(启动的时候根据config确定，http regular，grpc regular，pd， grpc_pd) -> policies -> worker(从资源池中) -> dispatch_to_workers
```

我们去掉这些固定模式的启动的模式。
根据注册的worker的实际模式，动态的选择router。且策略的实现是横跨http worker和 grpc的worker的资源池的。
下游的worker 可能是pd，也可能是regular的，链接方式可能是http也可能是grpc的，统一支持。实现mesh dynamic router
``` rust
request -> policies -> worker_pool-> workers -> routers -> dispatch_to_workers

//代码结构
backend = {sglang， vllm，atom}
let workers = get_workers_from_policies_and_request(policies, request， backend);
let router = get_router_from(workers, policies);
routers.route(workers, request, backend); // RAII design patten

```
5月份的target，router manger的模式暂时不做变动。
长远目标：重构成新的模式，同时在一个atomesh进程中支持多backend，多链接方式（http/grpc）, 多节点部署方式（regular，pd模式）

#### 重构点4：测试方法 
#####  重构单元测试 -> p0：
当前的mesh中的测试例的服用和修改。面向每个重构源文件的单元测试。

##### 开发mock测试：
1.大量的重构后，我们需要一个本地测试环境，来测试atomesh的路由接口的功能是否正常
2.沙盒环境测试Atomesh的基础服务能力
![](./Pasted%20image%2020260513112517.png)
通过功能设置虚拟的worker，包含regular或者pd worker。虚拟的work可以读取MOCK test中准备的测试的prompt和对应的response，虚拟的worker中模拟一个真实的推理engine的行为。
``` python
# virtual regular worker
app = FastAPI.build(xxx);

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
	texts = parser.read(mock_test.txt)
	if match(request.prompt, texts):
		t = match.results
		# return wapper.simulation_latency(t)
		return t
```
5月份的目标：完成所有mock功能的开发

#### 重构点4：重构metric
``` shell
mesh/src/observablility 
->
mesh/src/metrics

pub fn liveness
    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/engine_metrics", get(engine_metrics))
        .route("/v1/models", get(v1_models))
        .route("/get_model_info", get(get_model_info))
        .route("/get_server_info", get(get_server_info));
    
metrics_route = metrics_factory.get()
```
5月份目标：完成metrics的重构，主要是路由分离，命名规范等方面。

#### 重构点6：路由组合部分各自分离
当前申请中，build路由的时候，分为protected_routes， public_routes，admin_routes，这些routes各自分离到不同功能的文件夹下，通过工厂模式统一在启动文件出设置，各自解耦后期方便维护和扩展。
（主要面向ATOM的entrypoints统一）

### 人员分工
yajie ： router测的重构，单元测试整合处理。
yuechguo ：entrypoints， work_pool重构。
wanzhen ：本地mock测试方案开发，metrics部分重构。
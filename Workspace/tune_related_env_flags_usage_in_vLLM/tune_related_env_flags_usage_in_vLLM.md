### callable chain of TuneGemm method
when set/unset ENV VLLM_TUNE_GEMM,  VLLM_UNTUNE_FILE or VLLM_TUNE_FILE, python class TuneGemm was inited  in the process of vLLM setup:
``` Python
path : vllm/model_executor/layers/tuned_gemm.py

class TunedGemm:

    def __init__(self):
        #rocb_create_extension()
        #hipb_create_extension()
        self.extensions_created = False
        self.save_gemm = int(os.environ.get('VLLM_TUNE_GEMM', 0))
        self.untune_path = os.environ.get('VLLM_UNTUNE_FILE',
                                          "/tmp/vllm_untuned.csv")
        self.tune_path = os.environ.get('VLLM_TUNE_FILE', "tuned.csv")
        self.bestsols = {}
        self.load_best_sols()
        self.create_ds()
        self.cu_count = torch.cuda.get_device_properties(
            device='cuda').multi_processor_count

        if (self.save_gemm == 1):
            self.tuned_df = pd.DataFrame(columns=['M', 'N', 'K'])
        else:
            self.tuned_df = None

    def load_best_sols(self):
        if self.tune_path is not None and Path(self.tune_path).is_file():
            self.bestsols = pd.read_csv(self.tune_path)
......
tgemm = TuneGemm()
```

tgemm and it's static method was called by vLLM linear.py to construct model layer:
``` Python
path : vllm/model_executor/layers/linear.py

class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization.

    Args:
        separate_bias_add: If true, add bias separately after matrix
                           multiplication.
    """

    def __init__(self, separate_bias_add: bool = False):
        self.separate_bias_add = separate_bias_add

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        if self.separate_bias_add and bias is not None:
            return tgemm.mm(x, weight) + bias
        return tgemm.mm(x, weight, bias)
```

tgemm was also called by logits computation:
``` Python
path : vllm/model_executor/layers/logits_processor.py

    def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor,
                    embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Get the logits for the next tokens.
        logits = tgemm.mm(hidden_states, embedding)
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_gather(logits)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[:, :self.org_vocab_size]
        return logits
```

### TuneGemm running specification
when set VLLM_TUNE_FILE，TunedGemm load csv file, make local solids dict:
``` Python
class TunedGemm
	def __init__(self):
		self.self.tune_path = os.environ.get('VLLM_TUNE_FILE', "tuned.csv")
	......
	def load_best_sols(self):
        if self.tune_path is not None and Path(self.tune_path).is_file():
            self.bestsols = pd.read_csv(self.tune_path)

    def create_ds(self):
        df: pd.DataFrame = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = df.iloc[i]
            key = (ds['M'], ds['N'], ds['K'])
            if ds['libtype'] == 'hipblaslt':
                soltype = 1
            elif ds['libtype'] == 'rocblas':
                soltype = 2
            solds[key] = (soltype, int(ds['solidx']))
        self.solids = solds
        #print('>>>',solds)
    def query_sol(self, m, n, k):
        return self.solids.get((m, n, k), (0, 0))
```

when call TunedGemm mm method, if `out` is None and `soltype` is 1 or 2, direct call to rpcblas  or hipblaslt mm method:
``` Python
def mm(self, inp, weights, bias=None):
        # F.Linear can take a 3 dimensional input. vllm
        # uses this for linear units. However, sampler
        # will use torch.matmul with 2 dimensions only
        if inp.dim() == 3:
            inp_view = inp.view(-1, inp.size(-1))
            batched = True
        else:
            inp_view = inp
            batched = False
        if self.extensions_created is False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
        m = weights.shape[0]
        n = inp_view.shape[0]
        k = inp_view.shape[1]
        soltype, solidx = self.query_sol(m=m, n=n, k=k)
        out = self.apply_skinny(m, n, k, inp_view, weights)
        if out is not None:
            pass
        elif soltype == 1:
            out = hipb_mm(inp_view, weights.t(), solidx)
        elif soltype == 2:
            out = rocb_mm(inp_view, weights.t(), solidx)
        else:
            if (self.save_gemm == 1):
                self.tuned_df = pd.concat([
                    self.tuned_df,
                    pd.DataFrame({
                        'M': [m],
                        'N': [n],
                        'K': [k]
                    })
                ]).drop_duplicates()
                self.tuned_df.to_csv(self.untune_path, index=False)
            return F.linear(inp, weights, bias)
        if batched:
            out = out.view(inp.shape[0], inp.shape[1], weights.shape[0])
        if bias is not None:
            return out + bias
        return out

hipb_mm :
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hipb_create_extension", &hipb_create_extension, "create_extension");
  m.def("hipb_destroy_extension", &hipb_destroy_extension, "destroy_extension");
  m.def("hipb_mm", &HipbSolIdxBlas, "mm", py::arg("mat1"), py::arg("mat2"),
        py::arg("solution_index"), py::arg("outType") = at::nullopt,
        py::arg("scale1") = at::nullopt, py::arg("scale2") = at::nullopt,
        py::arg("scaleOut") = at::nullopt);
  m.def("hipb_findallsols", &HipbFindAllSolIdxBlas, "hipblas_find_all_sols",
        py::arg("mat1"), py::arg("mat2"), py::arg("outType") = at::nullopt);

rocb_mm:
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rocb_create_extension", &rocb_create_extension, "create_extension");
  m.def("rocb_destroy_extension", &rocb_destroy_extension, "destroy_extension");
  m.def("rocb_mm", &RocSolIdxBlas, "mm");
  m.def("rocb_findallsols", &RocFindAllSolIdxBlas, "rocblas_find_all_sols");
}
```

when $M.N,K$ meet some specifical value, `out` is None.`out` is None means mm call to vLLM custom implementation kernels.
``` Python
    def apply_skinny(self, m, n, k, inp_view, weights):
        if inp_view.dtype != torch.float16 or k % 8 != 0:
            return None
        if m > 8 and n <= 4:
            out = torch.empty(inp_view.shape[0],
                              weights.shape[0],
                              dtype=inp_view.dtype,
                              device='cuda')
            _custom_C.wvSpltK(weights, inp_view, out, n, self.cu_count)
            return out
        elif m % 4 == 0 and n == 1 and k <= 8192:
            out = torch.empty(inp_view.shape[0],
                              weights.shape[0],
                              dtype=inp_view.dtype,
                              device='cuda')
            _custom_C.LLMM1(weights, inp_view, out, 4)
            return out
        else:
            return None
```

if VLLM_TUNE_GEMM set to 1, `soltype` is 0(which means find none solids for this specific  matrix shape), TunedGemm auto save every matrix shape $M.N,K$ to VLLM_UNTUNE_FILE path and tgemm mm method just call to pytorch linear mm :
``` Python
import torch.nn.functional as F
......
	self.save_gemm = int(os.environ.get('VLLM_TUNE_GEMM', 0))
	self.untune_path = os.environ.get('VLLM_UNTUNE_FILE',
									  "/tmp/vllm_untuned.csv")
......
    def mm(self, inp, weights, bias=None):
        m = weights.shape[0]
        n = inp_view.shape[0]
        k = inp_view.shape[1]
        soltype, solidx = self.query_sol(m=m, n=n, k=k)
        out = self.apply_skinny(m, n, k, inp_view, weights)
        if out is not None:
            pass
        elif soltype == 1:
            out = hipb_mm(inp_view, weights.t(), solidx)
        elif soltype == 2:
            out = rocb_mm(inp_view, weights.t(), solidx)
        else:
            if (self.save_gemm == 1):
                self.tuned_df = pd.concat([
                    self.tuned_df,
                    pd.DataFrame({
                        'M': [m],
                        'N': [n],
                        'K': [k]
                    })
                ]).drop_duplicates()
                self.tuned_df.to_csv(self.untune_path, index=False)
            return F.linear(inp, weights, bias)
                                          
```
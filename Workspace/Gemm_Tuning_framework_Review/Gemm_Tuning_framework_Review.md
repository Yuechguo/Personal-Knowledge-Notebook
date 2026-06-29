[01 What is gemm tuning](#what-is-gemm-tuning)
[02 A code deep dive of tensile tunning](#a-code-deep-dive-of-tensile-tunning)
[03 Manually tunning with gradlib method](#manually-tunning-with-gradlib-method)
[04 Pytorch afo tuning with tunableop method](#pytorch-afo-tuning-with-tunuableop-method)

# What is gemm tuning
01 ref:[https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
02 ref:[https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
## 1.  Background Matrix-Matrix Multiplication
GEMMs (General Matrix Multiplications) are a fundamental building block for many operations in neural networks. GEMM is defined as the operation:
$$ C=\alpha AB+\beta C $$
with _A_ and _B_ as matrix inputs, _α_ and _β_ as scalar inputs, and _C_ as a pre-existing matrix which is overwritten by the output. 
A plain matrix product _AB_ is a GEMM with _α_ equal to one and _β_ equal to zero：
$$ C=AB $$
we will say that matrix _A_ is an $M\times K$  matrix, meaning that it has M rows and K columns. Similarly, _B_ and _C_ will be assumed to be $K\times N$ and $M\times N$ matrices, respectively.
## 2. GPU Implementation
### a) Tiling Technique
Tile size usually refers to the dimensions of these tiles (_Mtile_ x _Ntile_ in Figure 1). Each thread block computes its output tile by stepping through the K dimension in tiles, loading the required values from the _A_ and _B_ matrices, and multiplying and accumulating them into the output.

_Figure 1. Tiled outer product approach to GEMMs_
![](./Pasted%20image%2020240710105751.png)
### b) Computing data flow
The figure below illustrates the flow of data within this structure. This is the hierarchical GEMM computation embodied by CUTLASS. Each stage depicts a nested level of tiling which corresponds to a layer of concurrency within the CUDA execution model and to a level within the memory hierarchy, becoming increasingly finer moving left to right.

![](./Pasted%20image%2020240710110558.png)

## 3. Tuning scheme
GEMM tuning is **a powerful technique for enhancing the performance of matrix-multiplication operations**. This process includes selecting the most appropriate algorithm based on factors such as memory, cache, and compute capabilities.
For example, following parameters may impact the performance of gemm:
$$ M_{tile},K_{tile},N_{tile},thread_{x},thread_{y},transpose\_flag $$

# A code deep dive of tensile tunning 
01 ref:[http://iwapt.org/2018/iwapt2018_proceedings/Tensile_Paper_Talk.pdf](http://iwapt.org/2018/iwapt2018_proceedings/Tensile_Paper_Talk.pdf)
02 ref:[https://ieeexplore.ieee.org/document/8425532](https://ieeexplore.ieee.org/document/8425532)
03 ref:[https://github.com/ROCm/Tensile] (https://github.com/ROCm/Tensile)
04 ref :[https://github.com/ROCm/Tensile/wiki](https://github.com/ROCm/Tensile/wiki)
## a) Tensile paper introduction
_Figure. 1. Interdependencies of kernel parameter and GPU performance limitations; changing a single kernel parameter can improve and worsen competing performance limitations leading to a non-linear change in overall performance._
![](./Pasted%20image%2020240710161925.png)

Tensile employs several parameters to describe gemm computing kernels; each parameter both helps and hurts multiple GPU performance aspects as illustrated in Figure 1, thus necessitating automated tuning to search for the best combinations of kernel parameters. Tensile’s main kernel parameters affecting performance are:
- **LoopDoWhile:** True=DoWhile loop, False=While or For loop
- **LoopTail:** Additional loop with LoopUnroll=1.
- **EdgeType:** Branch, ShiftPtr or None
- **WorkGroup:** \[dim0, dim1, LocalSplitU\]
- **ThreadTile:** \[dim0, dim1\]
 - **LocalSplitU**: Splits up the summation within a workgroup; increasing this splitting will increase whole GPU occupancy, but at the cost of reducing global data sharing.
- **MatrixInstruction:** Type of matrix instruction used for the calculation, and wave tiling parameters \[InstructionM, InstructionN, InstructionK, InstructionB, BlocksInMDir, WaveTileM, WaveTileN, WaveGroupM, WaveGroupN\]
- **GlobalSplitU:** Split up summation among work-groups to create more concurrency. This option launches a kernel to handle the beta scaling, then a second kernel where the writes to global memory are atomic.
- **PrefetchGlobalRead:** True means outer loop should prefetch global data one iteration ahead.
- **PrefetchLocalRead:** True means inner loop should prefetch lds data one iteration ahead.
- **WorkGroupMapping:** In what order will work-groups compute C; affects cacheing.
- **LoopUnroll:** How many iterations to unroll inner loop; helps loading coalesced memory.
- **MacroTile:** Derrived from WorkGroup * ThreadTile.
- **DepthU:** Derrived from LoopUnroll * SplitU.
- **NumLoadsCoalescedA,B:** Number of loads from A in coalesced dimension.
- **GlobalReadCoalesceGroupA,B:** True means adjacent threads map to adjacent global read elements (but, if transposing data then write to lds is scattered).
- **GlobalReadCoalesceVectorA,B:** True means vector components map to adjacent global read elements (but, if transposing data then write to lds is scattered).
- **VectorWidth:** Thread tile elements are contiguous for faster memory accesses. For example VW=4 means a thread will read a float4 from memory rather than 4 non-contiguous floats.
- **KernelLanguage:** Whether kernels should be written in source code (HIP, OpenCL) or assembly (gfx803, gfx900, ...).

_Figure. 2. Tensile’s 7 step programmable benchmarking protocol first generates fast kernel candidates in Steps (1)-(6), then benchmarks those kernel candidates against a set of problem sizes in Step (7)._
![](./Pasted%20image%2020240710143618.png)

> _Step 1_: Initial Solution Parameters: Before Tensile is able to benchmark a kernel parameter in Step 2 of Figure 2, such as PrefetchGlobalRead={False, True}, all other kernel parameters not being benchmarked must be specified. Therefore, the first step is to initialize a list of default kernel parameters, then subsequent steps of benchmarking will override a parameter from this default list, with the parameter determined from benchmarking. Tensile is pre-loaded with default parameters for any unspecified during tuning.

> _Step 2_: Benchmark Common Parameters: Benchmarking common parameters determines parameters which are universally preferable to their alternatives regardless of other parameters. To benchmark common parameters: 
> 	(a) User specifies parameters and values to benchmark.
> 	(b) Tensile benchmarks all parameter combinations for a user-specified problem size.
> 	(c) Tensile selects the fastest parameter combination which is now labeled determined and will subsequently be used.
> In practice, this parameters isn’t used, since globally prefered parameters are set as defaults in Tensile and don’t need to be re-benchmarked.

> _Step 3_: Fork Parameters: Rather than continuing to determine globally fastest parameters, which eventually leads to a single fastest kernel, forking creates many different kernels, all of which will be considered for use. All forked parameters are considered determined, i.e., they aren't benchmarked to determine which is fastest. Figure 2 shows 7 kernels being forked in Step 3.

> _Step 4_: Benchmark Fork Parameters: Next, tuning continues its refinement by determining fastest parameters for each forked permutation, same as in Step 2.

>_Step 5_: Join Parameters: After tuning the forked kernels, joining reduces the list of kernels so that fewer kernels will be considered for final use. Each kernel in the resulting list must have different values for the listed JoinParameters, for example, employing JoinParameters = MacroTile will result in only a few final kernels, each with a different MacroTile. If there are multiple kernels with the same MacroTile, only the fastest is kept. In Figure 2 the 7 forked kernel have been reduced to 3 joined kernels.

>_Step 6_: Benchmark Join Parameters: Users can further tune parameters of the joined kernels. This steps is same as Steps 4 except that it tunes after joing so that there are fewer kernels to be tuned. In practice, this step isn’t used; using Step 4 is preferred so that all parameters are benchmarked before joinning.

> _Step 7_: Benchmark Final Parameters: At the conclusion of Step 6, all parameters of all kernels have been determined and the final set of kernels for consideration has been established. Now all final kernels will be benchmarked against all problem sizes specified by the user. Problem sizes can be specified as Range sizes and Exact sizes. Range sizes cause benchmarking of a broad range of sizes, and Tensile will be able to interpolate which kernel is best even between the specifically benchmarked sizes. Exact sizes cause a single problem size to be benchmarked, and the final library is guaranteed to choose the fastest kernel for that size. This final benchmarking generates the data that is subsequently analyzed for creating the mapping of problem size to optimal kernel.

## b) Tensile experiment
_Table 1: Table of fastest kernel parameters used in Figure 4,  columns denote kernel ID, MacroTile, DepthU, GlobalSplitU, ThreadTile, VectorWidth, WorkGroup AND LocalSplitU_.
![](./Pasted%20image%2020240711202158.png)

_Figure 3: For each $M\times N, K=3104$ problem size, the fastest kernel from Table 1 is charted. Some regions of problem sizes have the same fastest kernel (upper left and lower right) while other regions have different fastest kernels_.
![](./Pasted%20image%2020240711202640.png)


## c) Tensile code review
`TensileCreateLibrary` 
TensileCreateLibrary has a very important purpose in Tensile to generate and organize libraries of kernels to run on AMDGPU architectures. Given a set of previously defined solutions to a problem, each solution metadata is used to generate concrete GPU code (kernel) either in assembly or c++ source. The kernels themselves are basic operation building blocks that form the foundation of solving more complex problems and are common to many different applications. For example, kernels that implement the Generalized Matrix-Matrix Multiplication (GEMM) could ultimately end up being used in more complex machine learning or image processing techniques.
Of course we can define any solution meta-data that we want, but we have an aim to generate the most efficient and highest performing kernels possible. The benchmarking process that we use helps us find optimal kernels by testing out many varieties of parameterized solutions. Components from TensileCreateLibraries are used to generate the initial set of test solutions from the parameterized meta-data. After testing, the most optimal solutions are then selected and stored as configuration files. These configuration files are passed to TensileCreateLibraries to finally become the Master Solution Library.

`MasterSolutionLibrary`
we have an aggregate of solutions and their kernels, how are solutions selected and kernels executed at run time? To see how this works, we need to know a little bit more about the structure of the MasterSolutionLibrary. It is inefficient to check EVERY solution for suitability, so they are organized in a hierarchical structure to drastically reduce search time.
Libraries are used to implement hierarchical searches. At each level of the hierarchy, the predicates must be asserted before moving to the next level.
![](./Pasted%20image%2020240711173209.png)
problem sizes are matched based on minimum `euclidean distance` or `granularity loss`. Benchmarking is not done for every size imaginable so we must match the closest possible size. Solutions must also pass a final predicate that compares finer details of the problem description (example: CDStridesEqual: true AND KernelLanguageCompatible=ASM, ...). If the predicate fails then it is not included in the final selection.
At this point we may have a small pool of kernels that can correctly solve the problem and have performance data for solving a problem of similar size. Based on what we know of the benchmarking, the kernel with the highest speed is selected to ultimately solve the problem.
![](./Pasted%20image%2020240711143257.png)

`Tensile code deep dive`
When use tensile to tuning a Specifical problem size，first setup a tensile class object and create a `TenslieLibrary` from pre-generated `.dat` file
``` c
class TensileHost；
	std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>> m_library;

......

std::string tensileLibPath;
#if ROCBLASLT_TENSILE_LAZY_LOAD
#ifdef TENSILE_YAML
    tensileLibPath = path + "/TensileLibrary_lazy_" + processor + ".yaml";
#else
    tensileLibPath = path + "/TensileLibrary_lazy_" + processor + ".dat";
#endif
#else
#ifdef TENSILE_YAML
    tensileLibPath = path + "/TensileLibrary.yaml";
#else
    tensileLibPath = path + "/TensileLibrary.dat";
#endif
#endif

......

	auto lib = Tensile::LoadLibraryFile<Tensile::ContractionProblemGemm>(tensileLibPath);
              
	if(!lib)
	    std::cerr << "\nrocblaslt error: Could not load " << tensileLibPath
                              << std::endl;
    else
    {
	    using MSL = Tensile::MasterSolutionLibrary<Tensile::ContractionProblemGemm>;
	    m_library = std::dynamic_pointer_cast<MSL>(lib);
    }
```

The `.dat` file is obtained by serializing the `solidx <-> problem` map table, which is generated through brute-force searching of different parameter combinations. 
rocblas library dat and kernel file:
![](./Pasted%20image%2020240711220909.png)
hipblaslt library dat and kernel file:
![](./Pasted%20image%2020240711220824.png)

matching algo:
![](./Pasted%20image%2020240711183029.png)

Granularity matching algo:
``` c
// Compares the tile sizes of each kernel, the dimensions of the problem,
// and the number of compute units on the target GPU to select a kernel
// that fits the best on the GPU with the lowest amount of waste
// ("granularity loss").
struct GranularitySelectionLibrary : public SolutionLibrary<MyProblem, MySolution>
{
std::map<int, std::shared_ptr<MySolution>> solutions;
std::map<std::vector<size_t>, int>         exactMap;
};

virtual std::shared_ptr<MySolution> findBestSolution(
	MyProblem const& problem, Hardware const&  hardware,
    double* fitness = nullptr) const override
{
	const bool debug = Debug::Instance().printPropertyEvaluation();

	std::vector<size_t> key;
	size_t M = problem.freeSizeA(0);
	key.push_back(M);
	size_t N = problem.freeSizeB(0);
	key.push_back(N);
	size_t NumBatches = problem.batchSize(0);
	key.push_back(NumBatches);
	size_t K = problem.boundSize(0);
	key.push_back(K);

	auto exactMatch = exactMap.find(key);
	if(exactMatch != this->exactMap.end())
	{
		int index = exactMatch->second;
		auto rv = solutions.at(index);
		if(debug)
		{
			std::cout << "Exact match: " << rv->description();
			rv->problemPredicate->debugEval(problem, std::cout);
			std::cout << std::endl;
			rv->hardwarePredicate->debugEval(hardware, std::cout);
			std::cout << std::endl;
		}

		if((*rv->problemPredicate)(problem) && (*rv->hardwarePredicate)(hardware))
		{
			return rv;
		}
		else if(debug)
		{
			std::cout << "Predicate failure" << std::endl;
		}
	}

	double bestPerformance = 0.0;
	std::shared_ptr<MySolution> bestSolution;

	for(auto const& row : solutions)
	{
		auto myPerformance
			= row.second->projectedPerformance(problem, hardware).speedGFlops;

		if(debug)
		{
			std::cout << row.second->description() << ": " << myPerformance;
		}

		if(myPerformance > bestPerformance)
		{
			if((*row.second->problemPredicate)(problem)
			   && (*row.second->hardwarePredicate)(hardware))
			{
				bestPerformance = myPerformance;
				bestSolution    = row.second;

				if(debug)
					std::cout << " <-- Best so far";
			}
			else if(debug)
			{
				std::cout << " <-- Best, but predicate failure";
			}

			if(debug)
			{
				row.second->problemPredicate->debugEval(problem, std::cout);
				std::cout << std::endl;
				row.second->hardwarePredicate->debugEval(hardware, std::cout);
				std::cout << std::endl;
			}
		}
	}

	return bestSolution;
}

......

virtual SolutionSet<MySolution> findAllSolutions(
	MyProblem const& problem, Hardware const& hardware,
    SolutionLibrarySearchType searchType = SolutionLibrarySearchType::DEFAULT) const override
{
	bool debug = Debug::Instance().printPropertyEvaluation();

	SolutionSet<MySolution> rv;

	for(auto const& row : solutions)
	{
		if(debug)
		{
			std::cout << row.second->description() << ": ";
		}

		if(softwarePredicate(searchType, *(row.second), problem)
		   && (*row.second->hardwarePredicate)(hardware))
		{
			rv.insert(row.second);

			if(debug)
				std::cout << " Works";
		}
		else if(debug)
		{
			if(debug)
				std::cout << " Predicate failed";
		}

		if(debug)
		{
			if(searchType == SolutionLibrarySearchType::DEFAULT)
			{
				row.second->problemPredicate->debugEval(problem, std::cout);
				std::cout << std::endl;
			}
			row.second->hardwarePredicate->debugEval(hardware, std::cout);
			std::cout << std::endl;
		}
	}

	return rv;
}
```

euclidean distance matching algo:
``` c
// Uses a distance function to select solutions based on benchmarks.
// Benchmarks are performed to determine the optimal solution at a number of
// specific sizes. At runtime, we find the benchmarked size that is closest
// to the size asked for.
struct ProblemMatchingLibrary : public SolutionLibrary<MyProblem, MySolution>
{
using Element = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
using Table   = Matching::MatchingTable<MyProblem, Element, std::shared_ptr<MySolution>>;
std::shared_ptr<Table> table;
};

......

virtual std::shared_ptr<MySolution> findBestSolution(
	MyProblem const& problem, Hardware const&  hardware,
	double* fitness= nullptr) const override
{
	bool useDebugSelection = Debug::Instance().enableDebugSelection();

	typename Table::Transform transform
		= [&](Element library) -> std::shared_ptr<MySolution> {
		return library->findBestSolution(problem, hardware);
	};

	if(useDebugSelection)
	{
		std::shared_ptr<MySolution> evaluationSolution
			= table->findBestEvaluationSolution(problem, hardware, transform);
		return evaluationSolution;
	}
	else
	{
		double localFitness = std::numeric_limits<double>::max();
		fitness             = (fitness) ? fitness : &localFitness;
		std::shared_ptr<MySolution> solution;
		std::tie(solution, *fitness) = table->findBestMatch(problem, transform);
		return solution;
	}
}

......

virtual SolutionSet<MySolution>findAllSolutions(
	MyProblem const& problem, Hardware const& hardware,
    SolutionLibrarySearchType searchType = SolutionLibrarySearchType::DEFAULT) const override
{
	bool debug = Debug::Instance().printPropertyEvaluation();

	SolutionSet<MySolution> rv;

	auto matches = searchType != SolutionLibrarySearchType::DEFAULT
					   ? table->GetAll()
					   : table->matchesInOrder(problem);

	for(auto const& row : matches)
	{
		if(debug)
			std::cout << row->description() << std::endl;

		auto rowSolutions = row->findAllSolutions(problem, hardware, searchType);
		rv.insert(rowSolutions.begin(), rowSolutions.end());

		if(debug)
			std::cout << std::endl;
	}

	return rv;
}
```

### d) Tensile benchmarks library generate
build operation:
``` shell
cmake -DTensile_LIBRARY_FORMAT=msgpack -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON
```

``` shell
rocblas library create:
hipBLASLt/library/src/amd_detail/rocblaslt/src/CMakeLists.txt
  # Add a build target for Tensile kernel library
  # Runtime language is HIP by default
  # warning our Tensile_ variables may shadow variable in TensileCreateLibraryFiles
  # thus bypassing the function argument parameter system (mainly the options list) and CPU_THREADS
  if(Tensile_CPU_THREADS MATCHES "^[0-9]+$")
    # only including threads argument if number
    TensileCreateLibraryFiles(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/rocblaslt/src/Tensile/Logic/${Tensile_LOGIC}"
      "${PROJECT_BINARY_DIR}/Tensile"
      ARCHITECTURE        ${Tensile_ARCHITECTURE}
      CODE_OBJECT_VERSION ${Tensile_CODE_OBJECT_VERSION}
      COMPILER            ${Tensile_COMPILER}
      LIBRARY_FORMAT      ${Tensile_LIBRARY_FORMAT}
      CPU_THREADS         ${Tensile_CPU_THREADS}
      ${Tensile_Options}
    )
  else()
    TensileCreateLibraryFiles(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/rocblaslt/src/Tensile/Logic/${Tensile_LOGIC}"
      "${PROJECT_BINARY_DIR}/Tensile"
      ARCHITECTURE        ${Tensile_ARCHITECTURE}
      CODE_OBJECT_VERSION ${Tensile_CODE_OBJECT_VERSION}
      COMPILER            ${Tensile_COMPILER}
      LIBRARY_FORMAT      ${Tensile_LIBRARY_FORMAT}
      ${Tensile_Options}
    )
  endif()

hipblas library create:
hipBLASLt/tensilelite/Tensile/Source/TensileCreateLibrary.cmake

function(TensileCreateLibraryCmake
    Tensile_LOGIC_PATH
    Tensile_RUNTIME_LANGUAGE
    Tensile_COMPILER
    Tensile_CODE_OBJECT_VERSION
    Tensile_ARCHITECTURE
    Tensile_LIBRARY_FORMAT
    Tensile_MERGE_FILES
    Tensile_SHORT_FILE_NAMES
    Tensile_LIBRARY_PRINT_DEBUG
    Tensile_CPU_THREADS
    Tensile_SEPARATE_ARCHITECTURES
    Tensile_LAZY_LIBRARY_LOADING,
    Tensile_BUILD_ID)

  # execute python command
  if($ENV{TENSILE_SKIP_LIBRARY})
    message(STATUS "Skipping build of ${Tensile_OUTPUT_PATH}")
  else()
    if (WIN32)
      set(CommandLine ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} ${Tensile_CREATE_COMMAND})
    else()
      set(CommandLine ${Tensile_CREATE_COMMAND})
    endif()
    execute_process(
      COMMAND ${CommandLine}
      RESULT_VARIABLE Tensile_CREATE_RESULT
    )
    if(Tensile_CREATE_RESULT)
      message(FATAL_ERROR "Error generating kernels")
    endif()
  endif()
```
# Manually tuning with gradlib method
untuned.csv file format:
``` csv
M,N,K
12288,4096,4096
4096,4096,4096
22016,4096,4096
4096,4096,11008
32000,256,4096
12288,256,4096
4096,256,4096
22016,256,4096
```
M,N,K means in matrix multiplication，matrix A : NxK , matrix B : MxK
![](./Pasted%20image%2020240708130703.png)

Any piece of M，N，K is concat to one pd.dataframe:
``` Python
    def add_gemm(self, m, n, k):
        if (self.gdf is None
                or (self.gdf[(self.gdf['M'] == m) & (self.gdf['N'] == n) &
                             (self.gdf['K'] == k)].empty)):
            entry = {'M': [m], 'N': [n], 'K': [k]}
            df = pd.DataFrame(entry)
            self.gemm_problems = pd.concat([self.gemm_problems, df],
                                           ignore_index=True)
        else:
            print(
                f">>>Info: Found Duplicate shape(M:{m}, N:{n}, K:{k}), skipping"
            )
```

For any M.N,K matrix multiplication, build one GemmTuner class and get best sols:
``` Python
    def find_best_sols(self):
        df = self.gemm_problems
        soldf = pd.DataFrame()
        for i in range(len(df)):
            ds = df.iloc[i]
            gemmobj = Gemm(ds['M'],
                           ds['N'],
                           ds['K'],
                           indtype=self.indtype,
                           outdtype=self.outdtype,
                           rocblas_decode=self.rocblas_decode)
            gemmobj.find_fastest_solution()
            soldf.loc[i, 'libtype'] = gemmobj.best_libtype
            soldf.loc[i, 'solidx'] = gemmobj.best_solidx
            soldf.loc[i, 'soltimems'] = gemmobj.best_soltime
        soldf['indtype'] = self.indtype
        soldf['outdtype'] = self.outdtype
        finaldf = pd.concat([self.gemm_problems, soldf], axis=1)
        finaldf = pd.concat([finaldf, self.gdf])
        finaldf.to_csv(self.tuned_file, index=False)
        print(finaldf)
```

tuned.csv format:
``` csv
M,N,K,indtype,outdtype,libtype,solidx,soltimems
12288,4096,4096,torch.float16,torch.float16,rocblas,621282888.0,0.7278419017791748
4096,4096,4096,torch.float16,torch.float16,rocblas,621282495.0,0.2544680118560791
22016,4096,4096,torch.float16,torch.float16,rocblas,621282757.0,1.407721996307373
4096,4096,11008,torch.float16,torch.float16,rocblas,621282961.0,0.6782539844512939
32000,256,4096,torch.float16,torch.float16,rocblas,621282996.0,0.1622859001159668
12288,256,4096,torch.float16,torch.float16,rocblas,621282682.0,0.0722965002059936
4096,256,4096,torch.float16,torch.float16,rocblas,621282634.0,0.033076998591423
22016,256,4096,torch.float16,torch.float16,rocblas,621282728.0,0.1286059975624084
```

## How to find fastest solution
``` Python
def find_fastest_solution(self):
        if self.use_rocblas:
            self.find_rocblas_sols()
        if not (self.rocblas_decode and self.n == 1):
            self.find_hipblas_sols()
    .......
```

### Find rocblas or hipblaslt all sols for one M,N,K matrix multiplication:
``` Python
rocblas :
    def find_rocblas_sols(self):
        sols = rocsolidxgemm.rocb_findallsols(self.inp, self.weights.t())
        print('M N K',
              self.m,
              self.n,
              self.k,
              '>>> Total rocb solutions',
              len(sols),
              flush=True)
        #print(sols)
        self.rocb_sols = sols

hipblaslt :
	def find_hipblas_sols(self):
	sols = hipbsolidxgemm.hipb_findallsols(self.inp, self.weights.t(),
										   self.outdtype)
	print('M N K',
		  self.m,
		  self.n,
		  self.k,
		  '>>> Total hipb solutions',
		  len(sols),
		  flush=True)
	#print(sols)
	self.hipb_sols = sols

above internal var:
self.inp = torch.randn((self.n, self.k), device='cuda').to(self.indtype)
self.weights = torch.randn((self.m, self.k), device='cuda').to(self.indtype)
```

rocsolgemm code rocb_findallsols api:
``` c
std::vector<rocblas_int> RocFindAllSolIdxBlas(
    const torch::Tensor& mat1,
    const torch::Tensor& mat2
    )
{
......

#define GEMM_EX_ARGS                                                                              \
      r_handle, transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none, transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none, \
      m, n, k, &one, ptrA, abcRtype, mat1_ld, ptrB, abcRtype, mat2_ld, &zero, ptrC, \
      abcRtype, result_ld, ptrC, abcRtype, result_ld, rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index

      rocblas_int sizeSolve;
      //CHECK_ROCBLAS_ERROR(
      rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, NULL, &sizeSolve);
                  
      // Fill array with list of solutions that match type
      // Note: some of these may be invalid
      std::vector<rocblas_int> solutionsSolve(sizeSolve);
      //CHECK_ROCBLAS_ERROR(
      rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, solutionsSolve.data(), &sizeSolve);

      std::vector<rocblas_int> validSolutions;
      for(auto sol : solutionsSolve) {
        auto status = rocblas_gemm_ex(r_handle, 
                        transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
                        transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,
                        m, n, k, 
                        &one, ptrA, abcRtype, mat1_ld, ptrB, abcRtype, mat2_ld, 
                        &zero, ptrC, abcRtype, result_ld, 
                        ptrC, abcRtype, result_ld,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, sol, rocblas_gemm_flags_none);
        if (status == rocblas_status_success) {
          validSolutions.push_back(sol);
        }
      }

    return validSolutions;
}
```

``` c
rocblas_gemm_ex_get_solutions call chains:

{
	RocblasContractionProblem<Ti, To, Tc> problem{handle,
												  trans_a,
												  trans_b,
												  m,
												  n,
												  k,
												  (const Tc*)alpha,
												  (const Ti*)a,
												  nullptr,
												  lda,
												  stride_a,
												  offsetAin,
												  (const Ti*)b,
												  nullptr,
												  ldb,
												  stride_b,
												  offsetBin,
												  (const Tc*)beta,
												  (const To*)c,
												  nullptr,
												  ldc,
												  stride_c,
												  offsetCin,
												  (To*)d,
												  nullptr,
												  ldd,
												  stride_d,
												  offsetDin,
												  batch_count,
												  true,
												  flags};
	return getAllSolutions(problem, option, list_array, list_size);
}

......

template <typename TiA, typename To, typename Tc, typename TiB, typename TcA, typename TcB>
rocblas_status getAllSolutions(const RocblasContractionProblem<TiA, To, Tc, TiB, TcA, TcB>& prob,
                               rocblas_tensile_get_solution_option                          option,
                               rocblas_int* list_array,
                               rocblas_int* list_size)
{
    rocblas_status                                          status = rocblas_status_internal_error;
    std::set<std::shared_ptr<Tensile::ContractionSolution>> solutions;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
        std::shared_ptr<hipDeviceProp_t>                                             deviceProp;
        std::shared_ptr<Tensile::Hardware>                                           hardware;

        auto& adapter = get_library_and_adapter(&library, &deviceProp, prob.handle->getDevice());
        hardware      = Tensile::hip::GetDevice(*deviceProp);
        auto tensile_prob = ConstructTensileProblem(prob);

        if(option == CAN_SOLVE)
        {
            solutions = library->findAllSolutions(tensile_prob, *hardware);
        }
        else if(option == MATCHES_TYPE)
        {
            solutions = library->findAllSolutionsMatchingType(tensile_prob, *hardware);
        }
        else
        {
            return rocblas_status_invalid_value;
        }

        if(list_size == nullptr)
        {
            status = rocblas_status_invalid_pointer;
        }
        else if(list_array == nullptr)
        {
            *list_size = solutions.size();
            status     = rocblas_status_success;
        }
        else
        {
            rocblas_int i  = 0;
            auto        it = solutions.begin();
            while(i < *list_size && it != solutions.end())
            {
                list_array[i] = it->get()->index + 1;
                ++it;
                ++i;
            }
            status = rocblas_status_success;
        }
    }
    catch(const std::exception& e)
    {
        rocblas_internal_ostream msg;
        print_once(msg << "\nrocBLAS error: exception thrown for " << prob << e.what());
    }
    catch(...)
    {
        rocblas_internal_ostream msg;
        print_once(msg << "\nrocBLAS error: unknown exception thrown for " << prob);
    }
    return status;
}
```


hipbsolidxgemm code hipb_findallsols api:
``` c
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hipb_create_extension", &hipb_create_extension, "create_extension");
  m.def("hipb_destroy_extension", &hipb_destroy_extension, "destroy_extension");
  m.def("hipb_mm", &HipbSolIdxBlas, "mm", py::arg("mat1"), py::arg("mat2"),
        py::arg("solution_index"), py::arg("outType") = at::nullopt,
        py::arg("scale1") = at::nullopt, py::arg("scale2") = at::nullopt,
        py::arg("scaleOut") = at::nullopt);
  m.def("hipb_findallsols", &HipbFindAllSolIdxBlas, "hipblas_find_all_sols",
        py::arg("mat1"), py::arg("mat2"), py::arg("outType") = at::nullopt);
}

// find all hipblas solutions and return them to python land
std::vector<int> HipbFindAllSolIdxBlas(
    const torch::Tensor& mat1, const torch::Tensor& mat2,
    at::optional<py::object> Type = at::nullopt) {
......

return hipblasLtMatmul_findallsols_wrapper(
      hipblaslt_handle, transpose_mat1 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
      transpose_mat2 ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k, &one, ptrA,
      mat1_ld, ptrB, mat2_ld, &zero, ptrC, result_ld, hipblasInType,
      hipblasOutType, current_stream);
}

// find all hipblaslt solutions for given gemm problem
std::vector<int> hipblasLtMatmul_findallsols_wrapper(
    hipblasLtHandle_t handle, hipblasOperation_t op_A, hipblasOperation_t op_B,
    int m, int n, int k, const void* alpha, const void* a, int lda,
    const void* b, int ldb, const void* beta, void* c, int ldc,
    hipDataType intype, hipDataType outtype, hipStream_t& stream) {
......

  // std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(10);
  // CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(
  //     handle, matmul, matA, matB, matC, matC,
  //     preference, 10, heuristicResult.data(), &returnedAlgoCount));
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
  CHECK_HIPBLAS_ERROR(hipblaslt_ext::getAllAlgos(
      handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, op_A, op_B, intype,
      intype, outtype, outtype, HIPBLAS_COMPUTE_32F, heuristicResult));

  std::vector<int> algoIndex;
  int returned_algo_count = heuristicResult.size();
  // for (int i = 0; i < returnedAlgoCount; i++) {
  for (int i = 0; i < returned_algo_count; i++) {
    auto algo = heuristicResult[i].algo;
    size_t ret_workspace_size = 0;
    auto status = hipblaslt_ext::matmulIsAlgoSupported(
        handle, matmul, alpha, matA, matB, beta, matC, matC, algo,
        ret_workspace_size);
    if (status == HIPBLAS_STATUS_SUCCESS) {
      if (ret_workspace_size < workspace_size) {
        algoIndex.push_back(hipblaslt_ext::getIndexFromAlgo(algo));
      }
    }
  }

  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(matmul));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matA));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matB));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matC));
  return algoIndex;
}
```

### run every sol and get exec time
``` Python
    def find_fastest_solution(self):
        .......
        self.warmup()
        self.rocb_time_all_sols(fast_mode=1)
        self.warmup()
        self.hipb_time_all_sols(fast_mode=1)
        self.functional_check_topn_fastest()
        self.warmup()
        self.rocb_time_all_sols(fast_mode=0, top_sols=1)
        self.warmup()
        self.hipb_time_all_sols(fast_mode=0, top_sols=1)
        .......
```

rocblas sol time:
``` Python
    def rocb_time_sol(self, solidx, cold_iters=2, warm_iters=10):
        for i in range(cold_iters):
            rocsolidxgemm.rocb_mm(self.inp, self.weights.t(), solidx)
        self.start.record()
        for i in range(warm_iters):
            rocsolidxgemm.rocb_mm(
                self.inp, self.weights2[random.randint(0, self.nb - 1)].t(),
                solidx)
        self.end.record()
        torch.cuda.synchronize()
        gtime = self.start.elapsed_time(self.end) / warm_iters
        #print('>>> RocSolidx GTime',solidx,gtime,'ms')
        return gtime
```

rocsolidxgemm code rocb_mm api:
``` c
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rocb_create_extension", &rocb_create_extension, "create_extension");
  m.def("rocb_destroy_extension", &rocb_destroy_extension, "destroy_extension");
  m.def("rocb_mm", &RocSolIdxBlas, "mm");
  m.def("rocb_findallsols", &RocFindAllSolIdxBlas, "rocblas_find_all_sols");
}

torch::Tensor RocSolIdxBlas(
    const torch::Tensor& mat1,
    const torch::Tensor& mat2,
    const int32_t solution_index=0
    ) {
......

    rocblas_gemm_ex(r_handle, 
                    transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
                    transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,
                    m, n, k, 
                    &one, ptrA, abcRtype, mat1_ld, ptrB, abcRtype, mat2_ld, 
                    &zero, ptrC, abcRtype, result_ld, 
                    ptrC, abcRtype, result_ld,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, solution_index, flags);

}
```

hipblaslt sol time:
``` Python
    def hipb_time_sol(self, solidx, cold_iters=2, warm_iters=10):
        #print('>>>hipbtime',solidx)
        for i in range(cold_iters):
            hipbsolidxgemm.hipb_mm(self.inp, self.weights.t(), solidx,
                                   self.outdtype)
        self.start.record()
        for i in range(warm_iters):
            hipbsolidxgemm.hipb_mm(
                self.inp, self.weights2[random.randint(0, self.nb - 1)].t(),
                solidx, self.outdtype)
        self.end.record()
        torch.cuda.synchronize()
        gtime = self.start.elapsed_time(self.end) / warm_iters
        #print('>>> Solidx GTime',solidx,gtime,'ms')
        return gtime
```

hipbsolidxgemm code hipb_mm api:
``` c
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hipb_create_extension", &hipb_create_extension, "create_extension");
  m.def("hipb_destroy_extension", &hipb_destroy_extension, "destroy_extension");
  m.def("hipb_mm", &HipbSolIdxBlas, "mm", py::arg("mat1"), py::arg("mat2"),
        py::arg("solution_index"), py::arg("outType") = at::nullopt,
        py::arg("scale1") = at::nullopt, py::arg("scale2") = at::nullopt,
        py::arg("scaleOut") = at::nullopt);
  m.def("hipb_findallsols", &HipbFindAllSolIdxBlas, "hipblas_find_all_sols",
        py::arg("mat1"), py::arg("mat2"), py::arg("outType") = at::nullopt);
}

torch::Tensor HipbSolIdxBlas(
    const torch::Tensor& mat1, const torch::Tensor& mat2,
    const int solution_index, at::optional<py::object> Type = at::nullopt,
    at::optional<torch::Tensor> scale1 = at::nullopt,
    at::optional<torch::Tensor> scale2 = at::nullopt,
    at::optional<torch::Tensor> scaleOut = at::nullopt) {
......

CHECK_HIPBLAS_ERROR(hipblasLtMatmul_sol_wrapper(
      hipblaslt_handle, transpose_mat1 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
      transpose_mat2 ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k, &one, ptrA,
      mat1_ld, d_scale1, ptrB, mat2_ld, d_scale2, &zero, ptrC, result_ld,
      d_scaleOut, hipblasInType, hipblasOutType, current_stream,
      solution_index));

  return result;
}

hipblasStatus_t hipblasLtMatmul_sol_wrapper(
    hipblasLtHandle_t handle, hipblasOperation_t op_A, hipblasOperation_t op_B,
    int m, int n, int k, const void* alpha, const void* a, int lda,
    const void* scaleA, const void* b, int ldb, const void* scaleB,
    const void* beta, void* c, int ldc, const void* scaleC, hipDataType intype,
    hipDataType outtype, hipStream_t& stream, int solution_index = -1) {
......

  hipblasStatus_t status = hipblasLtMatmul(
      handle, matmul, alpha, a, matA, b, matB, beta, c, matC, c, matC,
      &heuristicResult[0].algo, d_workspace, workspace_size, stream);

  // nvtxRangePushA("hipBLASLt variables deletion");
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(matmul));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matA));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matB));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matC));
  // nvtxRangePop();

  return status;
}
```

##  get best sol which sol time is min
``` Python
 def find_fastest_solution(self):
        ......
        if len(self.rocb_gtimedf) > 0 and len(self.hipb_gtimedf) > 0:
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            if best_rocb_time < best_hipb_time * self.hipb_prefer_ratio:
                self.best_libtype = 'rocblas'
                self.best_solidx = self.rocb_gtimedf.index[0]
                self.best_soltime = best_rocb_time
            else:
                self.best_libtype = 'hipblaslt'
                self.best_solidx = self.hipb_gtimedf.index[0]
                self.best_soltime = best_hipb_time
            #self.check_gemm_ref(self.best_libtype,self.best_solidx)
        elif len(self.hipb_gtimedf) > 0:
            print('>>> Only hipblas solutions found!', flush=True)
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            self.best_libtype = 'hipblaslt'
            self.best_solidx = self.hipb_gtimedf.index[0]
            self.best_soltime = best_hipb_time
        elif len(self.rocb_gtimedf) > 0:
            print('>>> Only rocblas solutions found!', flush=True)
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            self.best_libtype = 'rocblas'
            self.best_solidx = self.rocb_gtimedf.index[0]
            self.best_soltime = best_rocb_time
        else:
            print('>>> No rocblas or hipblas solutions found!', flush=True)
            self.best_libtype = 'rocblas'
            self.best_solidx = 0
            self.best_soltime = 0
```
#  Pytorch afo tuning with tunableop method
afo tunning step:
``` shell
$ ROCBLAS_LAYER=4 bash train_7b.sh 2>&1 | grep "\- { rocblas_function:" | uniq | tee rocblas.yaml
$ python pytorch_afo_testkit/afo/tools/tuning/tune_from_rocblasbench.py rocblas.yaml --cuda_device 0 1 2 3 4 5 6 7
```

read input gemm problem from rocblas.yaml
``` Python
def main():
......
	for gemm in input_yaml:
	# loop through all sizes to benchmark

	default_dtype = "f32_r" if "sgemm" in gemm["rocblas_function"] else "f16_r"

	# FIXME: This is a workaround hack to skip known bad configs with hipblaslt:
	if (
		gemm["transA"] == "N"
		and gemm["transB"] == "N"
		and (
			(gemm.get("a_type", default_dtype) == "f16_r")
			or (gemm.get("a_type", default_dtype) == "f32_r")
		)
	):
		continue

	new = {
		"queue": queue,
		"m": gemm["M"],
		"n": gemm["N"],
		"k": gemm["K"],
		"b": gemm.get("batch_count", 1),
		"transpose": gemm["transA"] + gemm["transB"],
		"dtype": AFO_DTYPES.get(gemm.get("a_type", default_dtype), "fp16"),
		"num_iterations": 1,
		"num_warmup_iterations": 0,
		"profiling": False,
	}

	if new not in inputs:
		inputs.append(new)
......
    # Set untunable ops envs
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "afo_tune_device_%d_from_yaml.csv"
    with Pool(processes) as pool:
        for entry in tqdm.tqdm(
            pool.imap_unordered(wrapper, inputs), total=len(inputs), ascii=" 🛸="
        ):
            pass
        pool.close()
        pool.join()
    # Get csv files
    csvs = [
        filename for filename in os.listdir(".") if filename.endswith("from_yaml.csv")
    ]
```

multi-process benchmarks:
``` Python
# Need a wrapper to unpack args since tqdm doesn't work with starmap
def wrapper(kwargs):
    return distributed_benchmark_mm(**kwargs)


def distributed_benchmark_mm(queue=None, **kwargs):
    if queue is None:
        # Single GPU Case
        elapsed, cpu_elapsed, overhead, overhead_p, throughput, bw, ai = benchmark_mm(
            **kwargs
        )
    else:
        gpu_id = queue.get()
        try:
            elapsed, cpu_elapsed, overhead, overhead_p, throughput, bw, ai = (
                benchmark_mm(**kwargs, device=gpu_id)
            )
        finally:
            queue.put(gpu_id)
    entry = {
        "M": [kwargs["m"]],
        "N": [kwargs["n"]],
        "K": [kwargs["k"]],
        "B": [kwargs["b"]],
        "dtype": [kwargs["dtype"]],
        "transA": [kwargs["transpose"][0]],
        "transB": [kwargs["transpose"][1]],
        "elapsed_time_min(us)": [round(elapsed[0] * 10**6, 3)],
        "elapsed_time_max(us)": [round(elapsed[1] * 10**6, 3)],
        "elapsed_time_avg(us)": [round(elapsed[2] * 10**6, 3)],
        "elapsed_cpu_time_min(us)": [round(cpu_elapsed[0] * 10**6, 3)],
        "elapsed_cpu_time_max(us)": [round(cpu_elapsed[1] * 10**6, 3)],
        "elapsed_cpu_time_avg(us)": [round(cpu_elapsed[2] * 10**6, 3)],
        "overhead_min(us)": [round(overhead[0] * 10**6, 3)],
        "overhead_max(us)": [round(overhead[1] * 10**6, 3)],
        "overhead_avg(us)": [round(overhead[2] * 10**6, 3)],
        "overhead_min(%)": [round(overhead_p[0] * 100)],
        "overhead_max(%)": [round(overhead_p[1] * 100)],
        "overhead_avg(%)": [round(overhead_p[2] * 100)],
        "throughput(TF/s)": [round(throughput, 2)],
        "bandwidth(GB/s)": [round(bw, 2)],
        "arithmetic_intensity": [round(ai, 2)],
    }

    return entry

def benchmark_mm(
    m,n,k,num_iterations,num_warmup_iterations,num_rotating_buff=225,b=1,
    dtype="fp16",transpose="TN",profiling=False,device="0",
    do_flush_icache=True,fp16_alt=False,
):
......
    # XXX: Note that if m or k = 1 and b = 0 pytorch will not do the transpose?
    if transpose[1] == "T":
        A = generate_tensor(
            b, k, n, device=device, dtype=dtype, num_rotating_buff=0, fp16_alt=fp16_alt
        ).transpose(-1, -2)
    else:
        A = generate_tensor(
            b, n, k, device=device, dtype=dtype, num_rotating_buff=0, fp16_alt=fp16_alt
        )

    if transpose[0] == "T":
        B = generate_tensor(
            b,
            m,
            k,
            device=device,
            dtype=dtype,
            num_rotating_buff=num_rotating_buff,
            fp16_alt=fp16_alt,
        ).transpose(-1, -2)
    else:
        B = generate_tensor(
            b,
            k,
            m,
            device=device,
            dtype=dtype,
            num_rotating_buff=num_rotating_buff,
            fp16_alt=fp16_alt,
        )

......

    mm = get_torch_mm_func(A.dim(), fp16_alt)
    with get_profiling_context("torch" if profiling else "", trace_file_name):
        for i in range(num_warmup_iterations + num_iterations):
            start_time = time.perf_counter()
            # with torch.no_grad():
            start.record()
            mm(A, B[random.randint(0, num_rotating_buff - 1)])
            end.record()
            torch.cuda.synchronize()
            cpu_times[i] = time.perf_counter() - start_time
            times[i] = start.elapsed_time(end)
            if do_flush_icache:
                icache_ops.flush_icache(int(device))

def get_torch_mm_func(dims, fp16_alt):
    if fp16_alt:
        return AFO_mm.alt_mm
    elif dims == 2:
        return torch.mm
    elif dims == 3:
        return torch.bmm
    else:
        raise ValueError(f"Unsupported dim: {dims}")
```


pytorch tunableop scheme:
ref:[https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable)
when turn on tunable env flag:
``` c
template <typename DType>
inline void bgemm_tunable(CUDABLAS_BGEMM_ARGTYPES(DType)) {
  tunable::GemmStridedBatchedParams<DType> params;
  params.transa = transa;
  params.transb = transb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = alpha;
  params.a = a;
  params.lda = lda;
  params.stride_a = stridea;
  params.b = b;
  params.ldb = ldb;
  params.stride_b = strideb;
  params.beta = beta;
  params.c = c;
  params.ldc = ldc;
  params.stride_c = stridec;
  params.batch = num_batches;

  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  if (transa_ && transb_) {
    static tunable::GemmStridedBatchedTunableOp<DType, tunable::BlasOp::T, tunable::BlasOp::T> bgemm{};
    bgemm(&params);
  }
  else if (transa_ && !transb_) {
    static tunable::GemmStridedBatchedTunableOp<DType, tunable::BlasOp::T, tunable::BlasOp::N> bgemm{};
    bgemm(&params);
  }
  else if (!transa_ && transb_) {
    static tunable::GemmStridedBatchedTunableOp<DType, tunable::BlasOp::N, tunable::BlasOp::T> bgemm{};
    bgemm(&params);
  }
  else if (!transa_ && !transb_) {
    static tunable::GemmStridedBatchedTunableOp<DType, tunable::BlasOp::N, tunable::BlasOp::N> bgemm{};
    bgemm(&params);
  }
  else {
    TORCH_CHECK(false, "unreachable");
  }
}
```

register rcoblas or hipblaslt tunning op in local call map:
``` c
template <typename T, BlasOp ALayout, BlasOp BLayout>
class GemmStridedBatchedTunableOp : public TunableOp<GemmStridedBatchedParams<T>, StreamTimer> {
 public:
  GemmStridedBatchedTunableOp() {
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmStridedBatchedOp<T>>());

#ifdef USE_ROCM
    bool rocm_validators = false;

    static const char *env_rocblas = std::getenv("PYTORCH_TUNABLEOP_ROCBLAS_ENABLED");
    if (env_rocblas == nullptr || strcmp(env_rocblas, "1") == 0) {
      rocm_validators = true;
      for (auto&& [name, op] : GetRocBlasGemmStridedBatchedTypeStringAndOps<T>()) {
        this->RegisterOp(std::move(name), std::move(op));
      }
      AddRocblasValidator();
    }

    static const char *env_hipblaslt = std::getenv("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED");
    if (env_hipblaslt == nullptr || strcmp(env_hipblaslt, "1") == 0) {
      rocm_validators = true;
      // disallow tuning of hipblaslt with c10::complex
      if constexpr (
          !std::is_same_v<T, c10::complex<float>> &&
          !std::is_same_v<T, c10::complex<double>>) {
        for (auto&& [name, op] : GetHipBlasLtGemmStridedBatchedTypeStringAndOps<T, ALayout, BLayout>()) {
          this->RegisterOp(std::move(name), std::move(op));
        }
      }
      AddHipblasltValidator();
    }

    if (rocm_validators) {
      AddRocmValidator();
    }
#endif
  }

  std::string Signature() override {
    return c10::str("GemmStridedBatchedTunableOp_", TypeName<T>(T{}), "_", BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};
```

rocblas op:
``` c
template <typename T>
auto GetRocBlasGemmTypeStringAndOps() {
  rocblas_handle handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();
  int solution_size;
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();
  // Get the number of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            nullptr,
                                                            &solution_size));
  std::vector<int> solutions(solution_size);
  // Get the list of available solutions
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            solutions.data(),
                                                            &solution_size));
  // Sort the solutions in ascending order to make the solution vector deterministic across runs
  std::sort(solutions.begin(), solutions.end());

  std::vector<std::pair<std::string, std::unique_ptr<Callable<GemmParams<T>>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto callable = std::make_unique<RocblasGemmOp<T>>(solutions[i]);
    ret.emplace_back(std::make_pair(c10::str("Gemm_Rocblas_", solutions[i]), std::move(callable)));
  }
  return ret;
}
```

hipblaslt op:
``` c
template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
auto GetHipBlasLtTypeStringAndOps() {
  hipblasOperation_t transa_outer = MapLayoutToHipBlasLt(ALayout);
  hipblasOperation_t transb_outer = MapLayoutToHipBlasLt(BLayout);
  auto a_datatype = HipBlasDataTypeFor<AT>();
  auto b_datatype = HipBlasDataTypeFor<BT>();
  auto in_out_datatype = HipBlasDataTypeFor<CT>();
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;

  hipblasLtHandle_t handle;
  TORCH_HIPBLASLT_CHECK(hipblasLtCreate(&handle));
  TORCH_HIPBLASLT_CHECK(hipblaslt_ext::getAllAlgos(handle,
        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        transa_outer,
        transb_outer,
        a_datatype,
        b_datatype,
        in_out_datatype,
        in_out_datatype,
        HIPBLAS_COMPUTE_32F,
        heuristic_result));
  TORCH_HIPBLASLT_CHECK(hipblasLtDestroy(handle));

  // Sort heuristic_result by algo index to make sure the order of returned algos is deterministic.
  std::sort(heuristic_result.begin(),
      heuristic_result.end(),
      [](hipblasLtMatmulHeuristicResult_t& a, hipblasLtMatmulHeuristicResult_t& b) {
      return hipblaslt_ext::getIndexFromAlgo(a.algo) < hipblaslt_ext::getIndexFromAlgo(b.algo);
      });

  int returned_algo_count = heuristic_result.size();
  std::vector<std::pair<std::string, std::unique_ptr<Callable<ParamsT>>>> ret;
  for (int i = 0; i < returned_algo_count; i++) {
    auto algo = heuristic_result[i].algo;
    int algo_index = hipblaslt_ext::getIndexFromAlgo(algo);
    auto callable = std::make_unique<HipblasltGemmOp<AT, BT, CT, ALayout, BLayout, ParamsT>>(algo);
    std::string type_string = c10::str(
        "Gemm_Hipblaslt_", _charFromhipblasOp(transa_outer), _charFromhipblasOp(transb_outer), "_", algo_index);
    ret.emplace_back(type_string, std::move(callable));
  }

  return ret;
}
```
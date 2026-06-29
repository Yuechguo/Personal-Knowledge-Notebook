

# Tuning Gemm
### Prepare Model and Dataset
Install huggingface-cli for model download：
``` shell
pip3 install -U "huggingface_hub[cli]"
```
huggingface-cli tool spec ref：[https://huggingface.co/docs/huggingface_hub/guides/cli](https://huggingface.co/docs/huggingface_hub/guides/cli)

Download model Llama-2-7b-chat-hf（HF_HOME usage ：local model cache path）
``` shell
HF_HOME=$(pwd) huggingface-cli download NousResearch/Llama-2-7b-chat-hf
```

Qwen1.5-72B：
``` shell
HF_HOME=$(pwd) huggingface-cli download Qwen/Qwen1.5-72B
```

Download one dataset for tuning run：
``` shell
 wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Env Setup
docker image for tuning run：
``` shell
rocm/ali-private:vllm0.4.3_ROCm6.1.2_alibaba_240621_84fdb4
```

Download vllm：
``` shell
git clone clone https://github.com/ROCm/vllm.git
```

docker run:
``` shell
sudo docker run -it -v {path/do/dir}/vllm:/vllm rocm/ali-private:vllm0.4.3_ROCm6.1.2_alibaba_240621_84fdb4
```

Install gradlib：
``` shell
cd /vllm/gradlib/
pip3 uninstall gradlib
python3 setup.py install
```

docker commit:
``` shell
docker commit {containerId} rocm/ali-private:vllm0.4.3_ROCm6.1.2_alibaba_240621_84fdb4
```
### Tuning Steps
ref:[https://confluence.amd.com/display/~yushilin/gtuner](https://confluence.amd.com/display/~yushilin/gtuner)
docker run：
``` shell
sudo docker run -it --cap-add=SYS_PTRACE --privileged --shm-size=64GB --ipc=host --security-opt seccomp=unconfined --network=host --device=/dev/kfd --device=/dev/dri --group-add video rocm/ali-private:vllm0.4.3_ROCm6.1.2_alibaba_240621_84fdb4
```

Setup env flag：
```shell
export VLLM_UNTUNE_FILE="{/path/to/dir}/vllm_untuned.csv"
export VLLM_TUNE_FILE="{/path/to/dir}/tuned.csv"
export HIP_FORCE_DEV_KERNARG=1
export DEBUG_CLR_GRAPH_PACKET_CAPTURE=1
export VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1
```

Gen untuned.csv with benchmark_throughput( pytorch version >= 4.0.0):
``` Python
VLLM_TUNE_GEMM=1 python3 benchmark_throughput.py --model {path/to/dir}/Llama-2-7b-chat-hf --num-prompts=1000 --input-len 1000 --output-len 500 --tensor-parallel-size=1 --dtype float16 --worker-use-ray

or

VLLM_TUNE_GEMM=1 torchrun --standalone --nproc_per_node=1 --nnodes=1  benchmarks/benchmark_throughput.py --model {path/to/dir}/Llama-2-7b-chat-hf --trust-remote-code --num-prompts=1000 --input-len 1000 --output-len 500 --tensor-parallel-size 1
```

Gen untuned.csv with specify dataset:
``` Python
VLLM_TUNE_GEMM=1 python3 benchmark_throughput.py --model {path/to/dir}/Llama-2-7b-chat-hf --num-prompts=1000 --output-len 500 --tensor-parallel-size=1 --dtype float16 --worker-use-ray --dataset {/path/to/dir}/ShareGPT_V3_unfiltered_cleaned_split.json
or
VLLM_TUNE_GEMM=1 torchrun --standalone --nproc_per_node=1 --nnodes=1  benchmarks/benchmark_throughput.py --model {path/to/dir}/Llama-2-7b-chat-hf --trust-remote-code --num-prompts=1000 --output-len 500 --tensor-parallel-size 1 --dtype float16 --dataset {/path/to/dir}/ShareGPT_V3_unfiltered_cleaned_split.json
```

untuned performance：
``` shell
Llama-7b-chat-hf:
	Random input : 6623 tokens/s
	ShareGPT_V3 input ：5051.67 tokens/s
```

Gened untuned csv files：
![](./Pasted%20image%2020240701183748.png)


Run gemm tuner：
``` python
python3 vllm/gradlib/gradlib/gemm_tuner.py --outdtype f16 --input_file {/path/to/dir}/vllm_none_llama2untuned.csv --tuned_file {path/to/dir}/tuned.csv
```

Generated tuned csv file：
![](./Pasted%20image%2020240701183934.png)

run with tuned.csv：
``` python
export VLLM_TUNE_FILE="{path/to/dir}/tuned.csv"
unset VLLM_TUNE_GEMM

python3 benchmark_throughput.py --model /data/Llama-2-7b-chat-hf --num-prompts=1000 --input-len 1000 --output-len 500 --tensor-parallel-size=1 --worker-use-ray
or
torchrun --standalone --nproc_per_node=1 --nnodes=1  benchmarks/benchmark_throughput.py --model {path/to/dir}/Llama-2-7b-chat-hf --trust-remote-code --num-prompts=1000 --input-len 1000 --output-len 500 --tensor-parallel-size 1
```

run with tuned.csv with specify dataset:
``` Python
export VLLM_TUNE_FILE="{path/to/dir}/tuned.csv"
unset VLLM_TUNE_GEMM

python3 benchmark_throughput.py --model {path/to/dir}/Llama-2-7b-chat-hf --num-prompts=1000 --output-len 500 --tensor-parallel-size=1 --dtype float16 --worker-use-ray --dataset {/path/to/dir}/ShareGPT_V3_unfiltered_cleaned_split.json
or
torchrun --standalone --nproc_per_node=1 --nnodes=1  benchmarks/benchmark_throughput.py --model {path/to/dir}/Llama-2-7b-chat-hf --trust-remote-code --num-prompts=1000 --output-len 500 --tensor-parallel-size 1 --dtype float16 --dataset {/path/to/dir}/ShareGPT_V3_unfiltered_cleaned_split.json
```

tuned performance:
``` shell
Llama-7b-chat-hf:
	Random input :  6875 tokens/s (6623 -> 6864, 4% preformace up)
	ShareGPT_V3 input ： 5685 tokens/s (5051 -> 5685, 12% performace up)
```

### Tuning with multi-GPUs
If there are multiple sets of different combinations of GEMMs (corresponding to multiple CSV files), multiple GPUs can be utilized simultaneously to perform tuning calculations. 
The code reference for multi-GPU tuning is provided below as `multl_gpu_tuner.py`. The basic logic employs Python's `tempfile` mechanism to simulate input and output data files in a multi-process environment, and subsequently merges the results to generate a `tuned.csv` file.
``` Python
import os
import argparse
import math
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
from glob import glob

COLSUMNS=['M', 'N', 'K', 'indtype', 'outdtype', 'libtype', 'solidx', 'soltimems']

def start_one_process(device_id, input_file, output_file):
	# gemm_tuner is gradlib tuning api interface
    command = f"ROCR_VISIBLE_DEVICES={device_id} python3 {path/to/dir}/vllm/gradlib/gradlib/gemm_tuner.py --input_file {input_file} --tuned_file {output_file} --outdtype f16"
    # command = f"python gemm_tuner.py --input_file {input_file} --tuned_file {output_file} --outdtype f16"
    print(f"Starting: {command}")
    return subprocess.Popen(command, shell=True)

def main(args: argparse.Namespace):
    print(args)
    search_str = "{}/*{}*untuned*.csv".format(args.src, args.key_word)
    match_files = glob(search_str)
    
    all_lines = pd.DataFrame()
    for f in match_files:
        # column_names = ['M', 'N', 'K'] 
        lines = pd.read_csv(f)
        info = f"Pre-Read :{f} valid-lines:{len(lines)}"
        print(info)
        all_lines = pd.concat([all_lines, lines], axis=0, ignore_index=True)
    
    print(all_lines)

    input_tmpfiles = list()
    output_tmpfiles = list()
    gap_internal = math.ceil(len(all_lines) / args.num_gpus)
    for idx in range(0, args.num_gpus):
        # virtual input file for gemm_tuner input
        tmp_f = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        end_index = min((idx + 1) * gap_internal, len(all_lines))
        selected_rows = all_lines.iloc[idx * gap_internal : end_index]
        selected_rows.to_csv(tmp_f, index=False)
        input_tmpfiles.append(tmp_f)
        # virtual output file for gemm_tuner output
        o_tmp_f = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        pd.DataFrame(columns=COLSUMNS).to_csv(o_tmp_f, index=False)
        output_tmpfiles.append(o_tmp_f)
        #print(gap_internal)

    procs = list()
    for idx in range(0, args.num_gpus):
        f = input_tmpfiles[idx]
        o_tmp_f = output_tmpfiles[idx]
        completed = start_one_process(idx, input_file=f.name, output_file=o_tmp_f.name)
        procs.append(completed)
        
    for c in procs:
        c.wait()
    
    try :
        out_all_lines = pd.DataFrame()
        for of in output_tmpfiles:
            of_lines = pd.read_csv(of.name)
            info = f"Post-Process :{of.name} valid-lines:{len(of_lines)}"
            print(info)
            out_all_lines = pd.concat([out_all_lines, of_lines], axis=0, ignore_index=True)
        
        o_csv_str = os.path.join(args.src, "tuned.csv")
        if Path(o_csv_str).is_file():
            old_lines = pd.read_csv(o_csv_str)
            write_lines = pd.concat([out_all_lines, old_lines], axis=0, ignore_index=True)
            write_lines.to_csv(o_csv_str, index=False)
        else :
            out_all_lines.to_csv(o_csv_str, index=False)
    except Exception as e:
        print(e)
    
    for f in input_tmpfiles:
        f.close()
        os.remove(f.name)
    
    for f in output_tmpfiles:
        f.close()
        os.remove(f.name)

    return 0   

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Tuning.")
    parser.add_argument("--num_gpus",
                        "-n",
                        type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        default=1)
    parser.add_argument("--key_word",
                        "-k",
                        type=str,
                        default="")
    parser.add_argument("--src",
                        "-s",
                        type=str,
                        default="")

    args = parser.parse_args()
    if args.num_gpus > 8 :
        raise ValueError("num_gpus must less than 8")
    main(args)
```

run to generate tuned.csv：
``` shell
python3 multi_gpu_tuner.py -n 4 -k llama -s {path/to/dir}/
```

# How to find untuned gemm layout

docker image：
``` shell
rocm/ali-private:vllm0.4.3_ROCm6.1.2_alibaba_240621_84fdb4
```

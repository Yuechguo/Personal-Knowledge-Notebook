## Prepare
docker image with base env：
``` shell
docker pull rocm/ali-private:tf115
```

git clone tf115-rocm:
ref:[https://github.com/ROCm/tensorflow-upstream/commits/r1.15-rocm61-albm/](https://github.com/ROCm/tensorflow-upstream/commits/r1.15-rocm61-albm/))
``` shell
https://github.com/ROCm/tensorflow-upstream.git
cd tensorflow-upstream && git checkout r1.15-rocm61-albm
```

Alibaba-TF115-benchmark
ref:[https://amdcloud-my.sharepoint.com/my?id=%2Fpersonal%2Fyuechguo%5Famd%5Fcom%2FDocuments%2FAlibaba%2DTF115&ga=1](https://amdcloud-my.sharepoint.com/my?id=%2Fpersonal%2Fyuechguo%5Famd%5Fcom%2FDocuments%2FAlibaba%2DTF115&ga=1)
``` shell
download amd-blaze-benchmark.tar.gz
```

## compiling
run container:
``` shell
# /bin/bash

container_name=tuning_yuechao
docker_image=rocm/ali-private:tf115

sudo docker run -it --cap-add=SYS_PTRACE --privileged --shm-size=64GB --ipc=host --security-opt seccomp=unconfined --network=host --device=/dev/kfd --device=/dev/dri --group-add video --name $container_name -v {path/to/dir}/tf-ali:/data/ $docker_image

```
### compile of tensorflow-upstream
``` shell
	cd /data/tensorflow-upsteam && ./build_whl.sh
```
编译完成后，会自动安装tensorflow-upstream 1.15的版本到镜像里

### compile of Alibaba-TF115-Benchmark
first modify blaze-benchmark CMakeList.txt, make sure usage of google protobuf.
![](./Pasted%20image%2020240722201752.png)

build blaze-benchmark
``` shell
2. Install dependencies (glog-devel and boost) if not installed.

  * `apt-get install libgoogle-glog-dev`

  * `apt-get install libboost-all-dev`

3. Edit `Build.sh`, set cmake variables, and build the benchmark tool:

4. Prepare inputs and model.

* [Optional] Prepare runmeta, set biz\_options (after "session\_config") and send requests to biz, and copy generated runmeta files to local.

   "run\_options": {

     "traceTensorInfos": true,

     "traceLevel": "SOFTWARE\_TRACE"

   },

* Copy pbtxt and pb models to local.

5. Prepare benchmark configs (see example) and run.
```

## Running benchmark
modify blaze-benchmark/benchmark/core/model.cc
![](./Pasted%20image%2020240722205857.png)

running:
``` shell
cd {path/to/dir}/blaze-benchmark/example/DNN/all_star_newstag_nmd_jrc/
bash run.sh
```
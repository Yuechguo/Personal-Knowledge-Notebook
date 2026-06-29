获取所有commit的方法
```
git log --pretty=format:"%h | %ad | %an | %s" --date=short <tag1>..<tag2>

git log --pretty=format:"%h | %ad | %an | %s" --date=short 16267d4fa721b3e8c11e9b3d41f5d53fa8bedaf1..eb118d88c47d95f0006506ea7abeb0bd51a780b5 > pr-patch.log
```
保留，commit，arthur,  mail，和 patch info

初筛方法
```
cat pr-patch.log | grep -i 'aiter\|amd\|AMD\|rocm\|ROCm\|MI300X\|MI308X\|MI325x'
```

匹配筛选+去掉不重要
```
653b873b | 2025-07-08 | kk 43161300+kkHuang-amd@users.noreply.github.com| Fix cache modules of triton import error (#7832)
076313bd | 2025-07-07 | Haohui Mai ricetons@gmail.com| [AMD] Fail gracefully when AITER is unavailable gfx90a GPUs (#7187)
b116b21a | 2025-07-02 | Hubert Lu 55214931+hubertlu-tw@users.noreply.github.com| [AMD] Temporarily disable test_no_overlap_scheduler and test_vision_chunked_prefill (#7717)      
802815e4 | 2025-06-25 | valarLip 103567126+valarLip@users.noreply.github.com| take aiter get_rope back (#7521)
4c6675c4 | 2025-06-25 | valarLip 103567126+valarLip@users.noreply.github.com| enable aiter fp8 blockscale quant (#7520)
e984d507 | 2025-06-24 | valarLip 103567126+valarLip@users.noreply.github.com| enable aiter_biased_grouped_topk kernel (#7423)
755f3147 | 2025-06-24 | Alex Sun alex.s@amd.com| [AMD] add aiter fused moe in DeepEP path (#7268)
bd4f5818 | 2025-06-23 | kk 43161300+kkHuang-amd@users.noreply.github.com| Fix torch compile run (#7391)
405780bc | 2025-06-17 | kk 43161300+kkHuang-amd@users.noreply.github.com| [amd] Opt dsv3 moe (#7160)
1a9c2c92 | 2025-06-16 | Lianmin Zheng lianminzheng@gmail.com| Fix AMD speculative decoding (#7252)
8e2363dc | 2025-06-17 | Alex Sun alex.s@amd.com| fix amd EP MoE FP8 issue (#7125)
019851d0 | 2025-06-10 | Lianmin Zheng lianminzheng@gmail.com| Fix eagle on AMD (#7051)
47402883 | 2025-06-08 | Hubert Lu 55214931+hubertlu-tw@users.noreply.github.com| [AMD] Add more tests to per-commit-amd (#6926)
b819381f | 2025-06-05 | HAI hixiao@gmail.com| AITER backend extension and workload optimizations (#6838)
f4a8987f | 2025-05-28 | Sai Enduri saimanas.enduri@amd.com| Update amd docker and nightly models. (#6687)
eb8f02dd | 2025-05-26 | Sai Enduri saimanas.enduri@amd.com| Update nightly thresholds and dependencies. (#6635)
7a5e6ce1 | 2025-05-25 | kk 43161300+kkHuang-amd@users.noreply.github.com| Fix GPU OOM (#6564)
24c035f2 | 2025-05-24 | Sai Enduri saimanas.enduri@amd.com| Temporarily disable MI325x 8 gpu testing. (#6576)
5c0b38f3 | 2025-05-20 | HAI hixiao@gmail.com| aiter attention-backend (default enabled on AMD/ROCm) (#6381)
03886917 | 2025-05-20 | Lianmin Zheng lianminzheng@gmail.com| Disable all two stream overlap on amd (#6475)
6317c5c6 | 2025-05-19 | HAI hixiao@gmail.com| Address performance regression: disable multiple streams on ROCm (#6412)
198b9056 | 2025-05-14 | Hubert Lu 55214931+hubertlu-tw@users.noreply.github.com| [AMD] Fix Llama 4 Scout and Maverick accuracy issues on MI300X (#6274)
```


手动筛选
```
4b9971e4 | 2025-06-13 | sogalin 39478626+sogalin@users.noreply.github.com| Add gfx950 support for sgl-kernel. (#7092)
485a023b | 2025-05-29 | ChangyiYang 112288487+ChangyiYang@users.noreply.github.com| refactor apply_w8a8_block_fp8_linear in fp (#6545)
```


v0.4.6.post5
7e257cd666c0d639626487987ea8e590da1e9395

v0.4.7.post1
f9dc9dd28b53888f96f1953d6bb56085063dd913

v0.4.9.post2
eb118d88c47d95f0006506ea7abeb0bd51a780b5

git log --pretty=format:"%h | %ad | %an <%ae> | %s" --date=short 7e257cd666c0d639626487987ea8e590da1e9395..eb118d88c47d95f0006506ea7abeb0bd51a780b5 > v0.4.6.post5_v0.4.9.post2.log

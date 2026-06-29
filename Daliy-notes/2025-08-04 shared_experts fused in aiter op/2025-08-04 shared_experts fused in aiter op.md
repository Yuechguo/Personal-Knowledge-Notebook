# Fused shared experts with in EPmoe OP

###  TP mode without  shared experts
![](./Pasted%20image%2020250804101140.png)

###  TP mode with shared experts in vllm or sglang
shared experts was split to diff GPUs
![](./Pasted%20image%2020250804102327.png)

### EPmoe mode with shared experts in vllm or sglang
![](./Pasted%20image%2020250804103926.png)

### EPmoe mode with fused_shared_experts with in epmoe op itsef(we propose method)
![](./Pasted%20image%2020250804105348.png)

[Untitled Diagram - draw.io](https://app.diagrams.net/?src=about)

### impl
```
1.Modified the MOE sorted method in sgl-kernel to unify the valid range of expert_id.
2.Implemented fusion of shared expert computations into a consolidated MOE operator in EPMoE mode.
3.Aiter enabe for EPMoE
```
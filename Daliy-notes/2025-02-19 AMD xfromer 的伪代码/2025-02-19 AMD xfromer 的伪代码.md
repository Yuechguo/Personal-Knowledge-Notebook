
``` python
def print_before_execution(func):
    def wrapper(*args, **kwargs):
        query = args[0]
        key = args[1]
        value = args[2]
        attention_mask = kwargs.get("attn_mask", None)
        if attention_mask is None:
            batch_size, head_num, seq_len, head_dim = query.shape
            # if head_dim == 512:
            #     print (f"batch_size {batch_size}, head_num {head_num}, seq_len {seq_len}, head_dim {head_dim}")
            # query = query.reshape(batch_size * head_num, -1, head_dim)
            # key = key.reshape(batch_size * head_num, -1, head_dim)
            # value = value.reshape(batch_size * head_num, -1, head_dim)
            scale = kwargs.get("scale", None)
            p = kwargs.get("dropout_p", None)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=None, op=[xformers.ops.fmha.ck.FwOp], scale=scale, p=p
            )
            # [n, q, h * d] -> [n, q, h, d] -> [n, h, q, d]
            hidden_states = hidden_states.reshape(batch_size, -1, head_num, head_dim).transpose(1, 2)
            hidden_states = hidden_states.to(query.dtype)
            # breakpoint()
            return hidden_states
        return func(*args, **kwargs)
    return wrapper
 
torch.nn.functional.scaled_dot_product_attention = print_before_execution(
    torch.nn.functional.scaled_dot_product_attention
)
```
import numpy as np

def paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_table: list,
    context_len: int
) -> np.ndarray:
    """
    Perform scaled dot-product attention with paged KV cache.

    Args:
        query:       (num_heads, head_dim) query for a single token
        key_cache:   (num_physical_blocks, block_size, num_heads, head_dim)
        value_cache: (num_physical_blocks, block_size, num_heads, head_dim)
        block_table: list of physical block indices (logical -> physical)
        context_len: number of valid KV tokens in the sequence

    Returns:
        (num_heads, head_dim) attention output, rounded to 4 decimal places
    """
    num_heads = query.shape[0]
    head_dim = query.shape[1]
    block_size = key_cache.shape[1]
    all_keys = []
    all_values = []
    for logical_idx, physical_idx in enumerate(block_table):
        start = logical_idx * block_size
        valid_tokens = min(block_size, context_len-start)
        if valid_tokens <= 0:
            break
        all_keys.append(key_cache[physical_idx, :valid_tokens])
        all_values.append(value_cache[physical_idx, :valid_tokens])

    keys = np.concatenate(all_keys, axis=0)
    values = np.concatenate(all_values, axis=0)
    scores = np.sum(keys * query[None,:,:],axis = 2)
    scores = scores.T/np.sqrt(head_dim)
    scores = scores - np.max(scores,axis = 1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights,axis = 1, keepdims=True)
    values_by_head = values.transpose(1,0,2)
    output = np.matmul(weights[:,None,:], values_by_head).squeeze(axis=1)
    return np.round(output,4)



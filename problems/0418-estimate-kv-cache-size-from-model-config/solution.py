def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    # Your code here
    if 'num_kv_heads' in model_config:
        kv_cache_elements = model_config['num_layers'] * model_config['hidden_size'] * seq_len *batch_size *2 * model_config['num_kv_heads'] // model_config['num_attention_heads']
    else:
        kv_cache_elements = model_config['num_layers'] * model_config['hidden_size'] * seq_len *batch_size *2

    kv_cache_size_bytes = round(kv_cache_elements * model_config['dtype_bytes'],1)

    kv_cache_size_mb = round(kv_cache_size_bytes / (1024**2), 4)

    per_layer_size_mb = round(kv_cache_size_mb / model_config['num_layers'],4)

    per_token_size_kb = round(
    kv_cache_size_bytes / (seq_len * batch_size * 1024), 4
)

    ans = {'kv_cache_elements': kv_cache_elements, 'kv_cache_size_bytes': kv_cache_size_bytes, 'kv_cache_size_mb': kv_cache_size_mb, 'per_layer_size_mb': per_layer_size_mb, 'per_token_size_kb': per_token_size_kb}
    return ans

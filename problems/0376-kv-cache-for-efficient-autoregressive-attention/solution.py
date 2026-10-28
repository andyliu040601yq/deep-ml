import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    d_k = W_Q.shape[1]

    # 只计算新 token 的投影
    q = x_new @ W_Q
    k = x_new @ W_K
    v = x_new @ W_V

    # 把新 token 的 K、V 加入缓存
    if cache is None:
        K_cache = k[None, :]
        V_cache = v[None, :]
    else:
        K_cache, V_cache = cache
        K_cache = np.concatenate([K_cache, k[None, :]], axis=0)
        V_cache = np.concatenate([V_cache, v[None, :]], axis=0)

    # 新 token 对缓存中的所有 token 做 attention
    scores = K_cache @ q / np.sqrt(d_k)

    # 稳定 softmax
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)

    # 加权汇总所有缓存的 value
    output = weights @ V_cache

    return output, (K_cache, V_cache)
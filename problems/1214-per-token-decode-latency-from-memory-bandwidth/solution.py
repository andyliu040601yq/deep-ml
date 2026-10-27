def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    # Return [latency_ms, tokens_per_sec]
    latency_ms = num_params * bytes_per_param*1000/bandwidth_bytes_per_s
    token_per_sec = 1*1000/latency_ms
    ans = [latency_ms,token_per_sec]
    return ans
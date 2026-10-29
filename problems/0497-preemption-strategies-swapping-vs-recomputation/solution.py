def evaluate_preemption(
    num_tokens_list: list,
    kv_cache_bytes_per_token: int,
    swap_bandwidth_gbps: float,
    base_flops_per_token: int,
    attn_flops_per_token_pair: int,
    compute_tflops: float
) -> dict:
    ans = {
        'strategies':[],
        'swap_costs_ms': [],
        'recompute_costs_ms': [],
        'total_cost_ms': 0
    }
    for i in range(len(num_tokens_list)):
        tokens = num_tokens_list[i]
        swap_costs = round(2 *tokens*kv_cache_bytes_per_token/(1000000*swap_bandwidth_gbps),4)
        recompute_costs = round((tokens*base_flops_per_token + tokens**2 * attn_flops_per_token_pair)/(1000000000*compute_tflops),4)
        if swap_costs<recompute_costs:
            ans['strategies'].append('swap')
            ans['total_cost_ms'] += swap_costs
        else:
            ans['strategies'].append('recompute')
            ans['total_cost_ms'] += recompute_costs
        ans['swap_costs_ms'].append(swap_costs)
        ans['recompute_costs_ms'].append(recompute_costs)
    return ans


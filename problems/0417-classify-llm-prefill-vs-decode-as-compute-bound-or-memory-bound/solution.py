def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters
        sequence_length: Number of input tokens processed during prefill
        batch_size: Number of sequences processed in parallel during decode
        bytes_per_param: Memory footprint per parameter (e.g., 2 for FP16)
        peak_flops: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    prefill_flops = num_params * sequence_length * 2
    decode_flops = num_params * batch_size * 2
    prefill_bytes = num_params * bytes_per_param
    decode_bytes = num_params * bytes_per_param
    AI_prefill = round(prefill_flops/prefill_bytes,1)
    AI_decode = round(decode_flops/decode_bytes,1)
    ridge_point = round(peak_flops/peak_bandwidth,1)
    bottleneck_prefill = 'memory-bound'
    bottleneck_decode = 'memory-bound'
    if AI_prefill >= ridge_point:
        bottleneck_prefill = 'compute-bound'
    if AI_decode >= ridge_point:
        bottleneck_decode = 'compute-bound'
    prefill_achieved_flops = min(AI_prefill * peak_bandwidth,peak_flops)
    decode_achieved_flops = min(AI_decode * peak_bandwidth, peak_flops)
    decode_utilization_percent = 100*decode_achieved_flops/peak_flops
    prefill_utilization_percent = 100*prefill_achieved_flops/peak_flops

    ans = {
        'ridge_point':ridge_point,
        'prefill':{
            'total_flops':prefill_flops,
            'memory_bytes':prefill_bytes,
            'arithmetic_intensity':AI_prefill,
            'bottleneck':bottleneck_prefill,
            'achieved_flops':prefill_achieved_flops,
            'utilization_percent':prefill_utilization_percent
        },
        'decode':{
            'total_flops':decode_flops,
            'memory_bytes':decode_bytes,
            'arithmetic_intensity':AI_decode,
            'bottleneck':bottleneck_decode,
            'achieved_flops':decode_achieved_flops,
            'utilization_percent':decode_utilization_percent
        }
    }

    return ans



def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    AI = round(flops/bytes_accessed,1)
    ridge_point = round(peak_performance/peak_bandwidth,1)
    achieved_performance = round(min(AI * peak_bandwidth, peak_performance),1)
    utilization_percent = round(achieved_performance*100/peak_performance, 1)
    bottleneck = 'memory-bound'
    if AI>=ridge_point:
        bottleneck = 'compute-bound'
    dict = {
        'arithmetic_intensity': AI,
        'ridge_point': ridge_point,
        'bottleneck': bottleneck,
        'achieved_performance': achieved_performance,
        'utilization_percent': utilization_percent
    }
    return dict

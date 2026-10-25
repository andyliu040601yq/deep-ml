def compute_inference_metrics(timestamps: list[float]) -> dict:
    num_tokens = len(timestamps) - 1

    ttft = timestamps[1] - timestamps[0]

    tps = num_tokens / (timestamps[-1] - timestamps[0])

    if num_tokens == 1:
        itl = 0.0
    else:
        itl = (timestamps[-1] - timestamps[1]) / (num_tokens - 1)

    return {
        'ttft': round(ttft, 4),
        'tps': round(tps, 4),
        'itl': round(itl, 4)
    }
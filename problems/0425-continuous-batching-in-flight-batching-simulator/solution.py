from collections import deque

def continuous_batching_sim(requests: list[dict], max_batch_size: int) -> dict:
    """
    Simulate continuous batching for LLM inference.

    Args:
        requests: List of dicts with 'arrival_time' and 'tokens_needed'
        max_batch_size: Maximum concurrent sequences

    Returns:
        Dict with 'total_time', 'avg_latency', 'avg_ttft', 'throughput'
    """
    waiting = deque()
    active = []  # 每项是 [请求索引, 剩余 token 数]
    start_times = [None] * len(requests)
    completion_times = [None] * len(requests)

    next_request = 0
    completed = 0
    time = 0

    while completed < len(requests):
        # 1. 将当前时间及之前到达的请求加入等待队列
        while (
            next_request < len(requests)
            and requests[next_request]["arrival_time"] <= time
        ):
            waiting.append(next_request)
            next_request += 1

        # 2. 从等待队列头部填充空槽
        while waiting and len(active) < max_batch_size:
            idx = waiting.popleft()
            start_times[idx] = time
            active.append([idx, requests[idx]["tokens_needed"]])

        if not active:
            # 没有请求在处理或等待，时间跳到下一请求的到达时间
            time = requests[next_request]["arrival_time"]
            continue

        # 3. 每个活动请求生成一个 token
        for item in active:
            item[1] -= 1

        # 4. 移除完成的请求，并记录完成时间
        still_active = []
        for idx, remaining in active:
            if remaining == 0:
                completion_times[idx] = time
                completed += 1
            else:
                still_active.append([idx, remaining])
        active = still_active

        # 5. 推进到下一个时间步
        time += 1

    total_tokens = sum(req["tokens_needed"] for req in requests)

    avg_latency = sum(
        completion_times[i] - requests[i]["arrival_time"]
        for i in range(len(requests))
    ) / len(requests)

    avg_ttft = sum(
        start_times[i] - requests[i]["arrival_time"]
        for i in range(len(requests))
    ) / len(requests)

    throughput = total_tokens / time if time > 0 else 0.0

    return {
        "total_time": time,
        "avg_latency": round(avg_latency, 4),
        "avg_ttft": round(avg_ttft, 4),
        "throughput": round(throughput, 4),
    }
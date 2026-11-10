def chunked_prefill_schedule(
    prefill_requests,
    decode_requests,
    max_batch_tokens,
    max_chunk_size
):
    # 复制请求状态，避免修改传入的列表和字典
    pending_prefill = [
        {
            "id": req["id"],
            "remaining_prompt": req["prompt_tokens"],
            "decode_tokens": req["decode_tokens"],
        }
        for req in prefill_requests
    ]

    active_decode = [
        {
            "id": req["id"],
            "remaining_decode": req["remaining_decode"],
        }
        for req in decode_requests
    ]

    schedule = []
    step = 0

    while pending_prefill or active_decode:
        budget = max_batch_tokens
        decode_ids = []
        prefill_chunks = []

        # 1. 优先处理活动 Decode 请求，每个请求生成一个 token
        for req in active_decode:
            if budget <= 0:
                break

            decode_ids.append(req["id"])
            req["remaining_decode"] -= 1
            budget -= 1

        # 移除已完成 Decode 的请求
        active_decode = [
            req for req in active_decode
            if req["remaining_decode"] > 0
        ]

        # 2. 使用剩余预算按顺序处理 Prefill
        prefill_completed = []
        for req in pending_prefill:
            if budget <= 0:
                break

            chunk_size = min(
                req["remaining_prompt"],
                max_chunk_size,
                budget
            )

            if chunk_size <= 0:
                continue

            prefill_chunks.append((req["id"], chunk_size))
            req["remaining_prompt"] -= chunk_size
            budget -= chunk_size

            if req["remaining_prompt"] == 0:
                prefill_completed.append(req)

        # 移除已完成 Prefill 的请求
        pending_prefill = [
            req for req in pending_prefill
            if req["remaining_prompt"] > 0
        ]

        schedule.append({
            "step": step,
            "prefill_chunks": prefill_chunks,
            "decode_ids": decode_ids,
            "total_tokens": max_batch_tokens - budget,
        })

        # Prefill 完成的请求从下一步起加入 Decode 池
        for req in prefill_completed:
            if req["decode_tokens"] > 0:
                active_decode.append({
                    "id": req["id"],
                    "remaining_decode": req["decode_tokens"],
                })

        step += 1

    return schedule
def goodput_metrics(requests, sla_ttft_ms, sla_itl_ms, wall_time_s):
    """
    Throughput vs goodput for a decode trace under a TTFT/ITL SLA.

    Returns a dict with throughput_tps, goodput_tps, n_ok, n_total.
    """
    tokens = 0
    good_tokens = 0
    n_ok = 0
    n_total = len(requests)
    for i in range(len(requests)):
        ttft = requests[i]['ttft_ms']
        flag = False
        if ttft > sla_ttft_ms:
            flag = True
        for j in range(len(requests[i]['itl_ms'])):
            if requests[i]['itl_ms'][j] > sla_itl_ms:
                flag = True
                break
        if flag == False:
            n_ok += 1
            good_tokens += requests[i]['output_tokens']
        tokens += requests[i]['output_tokens']
    throughtput_tps = tokens/wall_time_s
    goodput_tps = good_tokens/wall_time_s
    ans = {
        'throughput_tps': throughtput_tps, 'goodput_tps': goodput_tps, 'n_ok': n_ok, 'n_total': n_total
    }
    return ans
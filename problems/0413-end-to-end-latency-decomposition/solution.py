import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    # 1. 计算每个请求的端到端延迟
    e2e = np.sum(list(stage_latencies.values()), axis=0)

    # 2. 计算端到端统计
    e2e_mean = round(float(np.mean(e2e)), 2)
    e2e_percentiles = {
        p: round(float(np.percentile(e2e, p)), 2)
        for p in percentiles
    }

    # 3. 计算每个阶段的统计
    stage_stats = {}
    stage_pct = {}

    for name, latencies in stage_latencies.items():
        mean = float(np.mean(latencies))

        stage_stats[name] = {
            'mean': round(mean, 2),
            'std': round(float(np.std(latencies)), 2),
            'percentiles': {
                p: round(float(np.percentile(latencies, p)), 2)
                for p in percentiles
            }
        }

        # 阶段平均延迟占比
        stage_pct[name] = round(mean / np.mean(e2e) * 100, 2)

    # 4. 找出平均延迟最大的阶段
    bottleneck = max(
        stage_latencies,
        key=lambda name: np.mean(stage_latencies[name])
    )

    return {
        'e2e_mean': e2e_mean,
        'e2e_percentiles': e2e_percentiles,
        'stage_stats': stage_stats,
        'bottleneck': bottleneck,
        'stage_pct': stage_pct
    }
import numpy as np
from typing import List, Tuple

def fit_bradley_terry(
    comparisons: List[Tuple[int, int]],
    n_items: int,
    learning_rate: float = 0.5,
    n_iterations: int = 100
) -> np.ndarray:
    beta = np.zeros(n_items, dtype=float)

    for _ in range(n_iterations):
        gradients = np.zeros(n_items, dtype=float)

        for winner, loser in comparisons:
            diff = beta[winner] - beta[loser]

            # 数值稳定的 sigmoid
            if diff >= 0:
                p = 1.0 / (1.0 + np.exp(-diff))
            else:
                exp_diff = np.exp(diff)
                p = exp_diff / (1.0 + exp_diff)

            # 对数似然的梯度：赢家增加，输家减少
            gradients[winner] += 1.0 - p
            gradients[loser] -= 1.0 - p

        beta += learning_rate * gradients

        # 中心化，使参数总和为 0
        beta -= np.mean(beta)

    return beta
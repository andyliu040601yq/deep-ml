import math

def elo_rating_update(ratings: dict, matches: list, k_factor: float) -> dict:
    updated = ratings.copy()

    for model_a, model_b, result in matches:
        rating_a = updated[model_a]
        rating_b = updated[model_b]

        # 根据当前分差计算双方的预期得分
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        # 根据比赛结果确定实际得分
        if result == "a":
            actual_a, actual_b = 1.0, 0.0
        elif result == "b":
            actual_a, actual_b = 0.0, 1.0
        else:  # draw
            actual_a, actual_b = 0.5, 0.5

        # 更新评分；下一场比赛使用更新后的评分
        updated[model_a] = rating_a + k_factor * (actual_a - expected_a)
        updated[model_b] = rating_b + k_factor * (actual_b - expected_b)

    return updated
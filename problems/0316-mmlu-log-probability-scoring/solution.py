import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    correctness = 0
    predictions = []
    avg_correct_prob = 0
    for i in range(len(log_probs)):
        nums = log_probs[i]
        temp = np.array(nums)
        maxnum = -np.inf
        idx = 0
        for j in range(len(nums)):
            if nums[j]>maxnum:
                maxnum = nums[j]
                idx = j
        shifted = temp - np.max(temp)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        avg_correct_prob += probs[correct_answers[i]]
        if idx==correct_answers[i]:
            correctness+=1
        predictions.append(idx)
    avg_correct_prob /= len(correct_answers)
    correctness /= len(correct_answers)
    ans = {
        'accuracy': correctness, 'predictions': predictions, 'avg_correct_prob': avg_correct_prob}
    return ans
     
            
import torch

def predict_logistic(X: torch.Tensor, weights: torch.Tensor, bias: float) -> torch.Tensor:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Your code here
    # pass
    y = X @ weights + bias
    p = torch.sigmoid(y)
    return (p>=0.5).int()
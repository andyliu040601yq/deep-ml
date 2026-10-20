import torch

def softsign(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the Softsign activation function.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: The Softsign of the input
    """
    result = x/(1+torch.abs(x))
    return result
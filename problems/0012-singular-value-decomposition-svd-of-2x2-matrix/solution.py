import torch

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 torch tensor
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Your code here
    B = A.T @ A

    eigenvalues, V = torch.linalg.eigh(B)

    indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[indices]
    V = V[:, indices]

    S = torch.sqrt(torch.clamp(eigenvalues, min=0))

    U = (A @ V) / S  # 仅适用于所有奇异值非零的情况

    return U, S, V.T
    



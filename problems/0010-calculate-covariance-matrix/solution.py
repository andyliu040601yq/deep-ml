import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    mean = v_t.mean(dim=1, keepdim=True)
    v_t = v_t - mean
    n = v_t.shape[1]
    cov_matrix = torch.matmul(v_t, v_t.T)/(n-1)
    return cov_matrix

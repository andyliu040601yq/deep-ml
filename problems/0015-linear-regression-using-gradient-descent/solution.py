import torch

def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Perform linear regression using gradient descent with PyTorch autograd.

    Args:
        X: Feature matrix (m, n) - can be tensor or array-like
        y: Target vector (m,) - can be tensor or array-like  
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D tensor of shape (n,)
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).flatten()
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True)
    
    model = torch.nn.Linear(X_t.shape[1], 1, bias=False)
    torch.nn.init.zeros_(model.weight)

    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

    for _ in range(iterations):
        optimizer.zero_grad()
        loss = ((model(X_t).flatten() - y_t) ** 2).mean() / 2
        loss.backward()
        optimizer.step()

    return model.weight.detach().flatten()
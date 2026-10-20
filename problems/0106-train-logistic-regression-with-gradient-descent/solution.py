
import torch

def train_logreg(X: torch.Tensor, y: torch.Tensor, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:

    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1, 1)

    input_dim = X.shape[1]
    output_dim = 1


    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid()
    ).double()


    torch.nn.init.zeros_(model[0].weight)
    torch.nn.init.zeros_(model[0].bias)


    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate
    )
    loss_fn = torch.nn.BCELoss(reduction="sum")

    losses = []


    for _ in range(iterations):
        predictions = model(X)
        loss = loss_fn(predictions, y)

        losses.append(round(loss.item(), 4))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    bias = model[0].bias.detach().item()
    weights = model[0].weight.detach().flatten().tolist()

    coefficients = [bias] + weights
    coefficients = [round(v, 4) for v in coefficients]

    return coefficients, losses

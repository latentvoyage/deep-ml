import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    f_mean = X.mean(dim = -1)
    f_std = torch.sqrt(X.var(dim = -1, correction = 0)+epsilon)
    #correction = 0 ensures that it is divided by D and not D-1,
    # apply to every row of X - is there some kind of apply function? or unsqueeze
    f_mean = f_mean.unsqueeze(-1)
    f_std = f_std.unsqueeze(-1)
    # print(X.shape, f_mean.shape, f_std.shape)
    X_ss = (X-f_mean)/(f_std)
    return gamma * X_ss + beta
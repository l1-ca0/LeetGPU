import torch

# data is a tensor on the GPU
def solve(data: torch.Tensor, N: int):
    # Sort inplace
    # torch.sort returns (values, indices) tuple. We take values and copy back.
    sorted_data, _ = torch.sort(data)
    data.copy_(sorted_data)
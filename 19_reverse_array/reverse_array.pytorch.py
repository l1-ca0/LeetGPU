import torch

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    # Reverse input in-place
    input[:] = torch.flip(input, [0]) 
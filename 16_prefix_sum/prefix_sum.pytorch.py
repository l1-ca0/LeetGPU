import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # torch.cumsum calculates the cumulative sum of elements
    # dim=0 because input is 1D
    # out=output ensures the result is written directly to the output buffer
    torch.cumsum(input, dim=0, out=output) 
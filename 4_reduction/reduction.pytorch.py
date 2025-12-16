import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # Compute sum of all elements and store in output
    # torch.sum returns a 0-dim tensor (scalar), we fill the output tensor with it.
    output.fill_(torch.sum(input))

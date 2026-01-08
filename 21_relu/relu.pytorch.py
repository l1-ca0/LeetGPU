import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # ReLU: max(0, x)
    output.copy_(torch.relu(input)) 
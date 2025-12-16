import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # PyTorch softmax along dimension 0, in-place update
    output.copy_(torch.softmax(input, dim=0))
 

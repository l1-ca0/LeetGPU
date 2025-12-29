import torch

# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    # Compute dot product
    # torch.dot treats tensors as 1D vectors
    res = torch.dot(A, B)
    
    # Store result in the output scalar tensor
    result.copy_(res) 
import torch

# A, x, y are tensors on the GPU
def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    A_mat = A.view(M, N)
    res = torch.matmul(A_mat, x)
    y.copy_(res)
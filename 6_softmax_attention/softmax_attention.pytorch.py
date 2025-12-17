import torch

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          M: int, N: int, d: int):
    scale = d ** 0.5
    # Score = Q @ K.T / sqrt(d)
    attn = torch.matmul(Q, K.t()) / scale
    # Softmax on last dimension (N)
    attn = torch.softmax(attn, dim=1)
    # Output = Attn @ V
    torch.matmul(attn, V, out=output) 
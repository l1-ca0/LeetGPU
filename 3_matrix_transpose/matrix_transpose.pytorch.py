import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    # input is (rows, cols), output is (cols, rows)
    # output[:] = input.t() copies the transposed data into the output tensor
    output.copy_(input.t()) 
import torch
import torch.nn.functional as F

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
          input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
    # Reshape input and kernel to (Batch=1, Channel=1, H, W)
    input_reshaped = input.view(1, 1, input_rows, input_cols)
    kernel_reshaped = kernel.view(1, 1, kernel_rows, kernel_cols)
    
    # Perform 2D convolution
    # Default padding is 0 (Valid), stride is 1
    res = F.conv2d(input_reshaped, kernel_reshaped)
    
    # Flatten result and copy
    output.copy_(res.flatten())
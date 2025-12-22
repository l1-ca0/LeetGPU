import torch

import torch.nn.functional as F

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    # Reshape input and kernel to (Batch=1, Channels=1, Length) for conv1d
    input_reshaped = input.view(1, 1, -1)
    kernel_reshaped = kernel.view(1, 1, -1)
    
    # Perform 1D convolution (valid padding by default)
    res = F.conv1d(input_reshaped, kernel_reshaped)
    
    # Flatten result and copy to output tensor
    output.copy_(res.flatten()) 
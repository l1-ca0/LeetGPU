import torch

# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    # 1. Initialize Output
    histogram.zero_()
    
    # 2. Filter / Validation
    mask = (input >= 0) & (input < num_bins)
    valid_input = input[mask]
    
    # 3. Compute Histogram (Bincount)
    # torch.bincount counts frequency of each non-negative integer value.
    # minlength=num_bins ensures the result has size at least `num_bins`,
    # even if the largest value in valid_input < num_bins - 1.
    counts = torch.bincount(valid_input, minlength=num_bins)
    
    # 4. Copy to Output
    histogram.copy_(counts)
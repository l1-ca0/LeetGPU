import torch

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    # Reshape flattened tensor to (height, width, 4) to access RGBA channels
    # memory layout is preserved, allowing in-place modification
    img_view = image.view(height, width, 4)
    
    # Invert RGB channels (indices 0, 1, 2). Alpha (index 3) is unchanged.
    img_view[..., :3] = 255 - img_view[..., :3]

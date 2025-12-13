import cutlass
import cutlass.cute as cute

# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    idx = cute.threadIdx.x + cute.blockIdx.x * cute.blockDim.x
    if idx < N:
        C[idx] = A[idx] + B[idx]

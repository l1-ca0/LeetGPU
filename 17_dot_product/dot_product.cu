#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// Warp Reduction using shuffle intrinsics
__device__ float warp_reduce(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Block Reduction
__device__ float block_reduce(float val) {
  static __shared__ float shared[32]; // Shared memory for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warp_reduce(val); // Reduce within warp

  if (lane == 0) {
    shared[wid] = val; // Store warp sum
  }
  __syncthreads();

  // Read back and reduce the warp sums (assuming block size 256 -> 8 warps)
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid == 0) {
    val = warp_reduce(val);
  }
  return val;
}

__global__ void dot_product_kernel(const float *A, const float *B,
                                   float *result, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = gridDim.x * blockDim.x;
  float local_sum = 0.0f;

  // Grid-Stride Loop
  for (int i = idx; i < N; i += grid_stride) {
    local_sum += A[i] * B[i];
  }

  // Block-Level Reduction
  float block_sum = block_reduce(local_sum);

  // Atomic Add to global memory
  if (threadIdx.x == 0) {
    atomicAdd(result, block_sum);
  }
}

// A, B, result are device pointers
extern "C" void solve(const float *A, const float *B, float *result, int N) {
  // 1. Initialize result to 0
  cudaMemset(result, 0, sizeof(float));

  // 2. Launch Configuration
  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  // Limit grid size to maximize occupancy but rely on grid-stride loop for
  // larger N For N=10000, blocks=40. For very large N, we might cap blocks.
  if (blocksPerGrid > 1024)
    blocksPerGrid = 1024;

  dot_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);

  cudaDeviceSynchronize();
}
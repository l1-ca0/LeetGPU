#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// Warp Reduction
__device__ float warp_reduce(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Block Reduction
__device__ float block_reduce(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warp_reduce(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid == 0) {
    val = warp_reduce(val);
  }
  return val;
}

// Block-per-Row Dense Matrix-Vector Multiplication
// A is flattened (M * N), x is (N), y is (M)
__global__ void matvec_kernel(const float *A, const float *x, float *y, int M,
                              int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;

  float sum = 0.0f;
  // Iterate over columns with grid-stride (actually block-stride since 1
  // block/row)
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    sum += A[row * N + col] * x[col];
  }

  // Reduce partial sums within the block
  sum = block_reduce(sum);

  if (threadIdx.x == 0) {
    y[row] = sum;
  }
}

// A, x, y are device pointers
extern "C" void solve(const float *A, const float *x, float *y, int M, int N,
                      int nnz) {
  // Each block handles one row
  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = M;

  matvec_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, x, y, M, N);

  cudaDeviceSynchronize();
}
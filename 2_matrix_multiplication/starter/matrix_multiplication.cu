#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  // Calculate global row and column indices for the current thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the thread is within the bounds of the output matrix (M x K)
  if (row < M && col < K) {
    float sum = 0.0f;
    // Iterate over the shared dimension N to compute the dot product
    for (int k = 0; k < N; ++k) {
      // Accumulate product of elements from A (row) and B (col)
      sum += A[row * N + k] * B[k * K + col];
    }
    // Store the final result in the output matrix C
    C[row * K + col] = sum;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}

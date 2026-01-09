#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// GEMM kernel: C = alpha * A * B + beta * C
// A: (M, K), B: (K, N), C: (M, N)
__global__ void gemm_kernel(const half *A, const half *B, half *C, int M, int N,
                            int K, float alpha, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    float c_val = __half2float(C[row * N + col]);
    C[row * N + col] = __float2half(alpha * sum + beta * c_val);
  }
}

// A, B, and C are device pointers
extern "C" void solve(const half *A, const half *B, half *C, int M, int N,
                      int K, float alpha, float beta) {
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha,
                                                  beta);
  cudaDeviceSynchronize();
}

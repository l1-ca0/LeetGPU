#include <cuda_runtime.h>

#include <math.h>

// Kernel 1: Calculate Scores S = Q * K^T / sqrt(d)
// Grid: (N / 16, M / 16), Block: (16, 16)
__global__ void matmul_qk_kernel(const float *Q, const float *K, float *S,
                                 int M, int N, int d) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float scale = 1.0f / sqrtf((float)d);

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < d; ++k) {
      // Q is (M, d), K is (N, d) -> K^T is (d, N)
      // S[row, col] = dot(Q[row, :], K[col, :])
      sum += Q[row * d + k] * K[col * d + k];
    }
    S[row * N + col] = sum * scale;
  }
}

// Kernel 2: Row-wise Softmax on S -> A
// Grid: (M, 1), Block: (256, 1) or similar. Each block handles one row.
// Kernel 2: Row-wise Softmax on S -> A
// Grid: (M, 1), Block: (256, 1) or similar. Each block handles one row.
__global__ void softmax_kernel(float *S, int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;

  // Use shared memory for reduction.
  // Each thread processes part of the row, then we reduce in shared memory.
  extern __shared__ float sdata[];
  float *s_val = sdata;

  // 1. Find Max
  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    local_max = fmaxf(local_max, S[row * N + i]);
  }
  s_val[threadIdx.x] = local_max;
  __syncthreads();

  // Block reduction for Max
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_val[threadIdx.x] =
          fmaxf(s_val[threadIdx.x], s_val[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float row_max = s_val[0];

  // 2. Compute Sum Exp
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    local_sum += expf(S[row * N + i] - row_max);
  }
  s_val[threadIdx.x] = local_sum;
  __syncthreads();

  // Block reduction for Sum
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_val[threadIdx.x] += s_val[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float row_sum = s_val[0];

  // 3. Normalize
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    S[row * N + i] = expf(S[row * N + i] - row_max) / row_sum;
  }
}

// Kernel 3: Calculate Output O = A * V
// A is (M, N), V is (N, d) -> O is (M, d)
// Grid: (d / 16, M / 16), Block: (16, 16)
__global__ void matmul_av_kernel(const float *A, const float *V, float *O,
                                 int M, int N, int d) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < d) {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] * V[k * d + col];
    }
    O[row * d + col] = sum;
  }
}

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int N, int d) {
  // 1. Allocate temporary memory for Scores (M x N)
  float *d_S;
  cudaMalloc(&d_S, M * N * sizeof(float));

  // 2. Launch QK^T
  dim3 block(16, 16);
  dim3 grid_qk((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  matmul_qk_kernel<<<grid_qk, block>>>(Q, K, d_S, M, N, d);

  // 3. Launch Softmax
  int threads_softmax = 256;
  int blocks_softmax = M;
  // Shared mem: threads_softmax * sizeof(float)
  softmax_kernel<<<blocks_softmax, threads_softmax,
                   threads_softmax * sizeof(float)>>>(d_S, M, N);

  // 4. Launch AV
  dim3 grid_av((d + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  matmul_av_kernel<<<grid_av, block>>>(d_S, V, output, M, N, d);

  cudaDeviceSynchronize();
  cudaFree(d_S);
}

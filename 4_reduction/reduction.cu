#include <cuda_runtime.h>

__global__ void partial_reduction_kernel(const float *input,
                                         double *partial_outputs, int N) {
  // Shared memory for block-level reduction
  extern __shared__ double sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockDim.x * gridDim.x;

  // Grid-Stride Loop
  double threadSum = 0.0;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gridSize) {
    threadSum += (double)input[i];
  }

  sdata[tid] = threadSum;
  __syncthreads();

  // Block reduction
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write block partial sum to global memory
  if (tid == 0) {
    partial_outputs[blockIdx.x] = sdata[0];
  }
}

__global__ void final_reduction_kernel(const double *partials, float *output,
                                       int num_partials) {
  // Single thread accumulates all partial sums in high precision
  if (threadIdx.x == 0) {
    double sum = 0.0;
    for (int i = 0; i < num_partials; ++i) {
      sum += partials[i];
    }
    *output = (float)sum;
  }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = 1024;
  // Handle small N case where we don't need many blocks
  if (N < blocksPerGrid * threadsPerBlock) {
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0)
      blocksPerGrid = 1;
  }

  // Allocate temporary buffer for partial sums
  double *d_partials;
  cudaMalloc(&d_partials, blocksPerGrid * sizeof(double));

  // Pass 1: Reduce to partial sums
  partial_reduction_kernel<<<blocksPerGrid, threadsPerBlock,
                             threadsPerBlock * sizeof(double)>>>(input,
                                                                 d_partials, N);

  // Pass 2: Final reduction
  final_reduction_kernel<<<1, 1>>>(d_partials, output, blocksPerGrid);

  cudaDeviceSynchronize();
  cudaFree(d_partials);
}

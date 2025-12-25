#include <cuda_runtime.h>

// "Grid-Stride Loop" pattern.
// stride equal to the total number of threads in the grid.
__global__ void histogram_kernel(const int *input, int *histogram, int N,
                                 int num_bins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Grid-Stride Loop
  for (int i = idx; i < N; i += stride) {
    int val = input[i];
    // Check if value is within valid bin range
    if (val >= 0 && val < num_bins) {
      // Atomic Add
      atomicAdd(&histogram[val], 1);
    }
  }
}

// input, histogram are device pointers
extern "C" void solve(const int *input, int *histogram, int N, int num_bins) {
  // Zero out histogram memory first
  cudaMemset(histogram, 0, num_bins * sizeof(int));

  int threadsPerBlock = 256;

  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N,
                                                       num_bins);
  cudaDeviceSynchronize();
}

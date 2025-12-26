#include <math_constants.h>

// Kernel to fill padding with a specific value (infinity)
__global__ void fill_kernel(float *data, int start_idx, int end_idx,
                            float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx + start_idx < end_idx) {
    data[idx + start_idx] = value;
  }
}

__global__ void bitonic_sort_step(float *data, int j, int k, int M) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= M)
    return;

  unsigned int ixj = i ^ j;

  // The threads with the lowest coordinate of a pair (i, ixj) exchange values
  if (ixj > i) {
    if (ixj < M) {
      float a = data[i];
      float b = data[ixj];

      // Ascending if in even block of size 2*k, Descending otherwise?
      // Standard Bitonic Sort Logic:
      // For a stage 'k' (width of monotonic sequences being built):
      // Direction depends on position relative to 'k'.
      // Actually, for a pure sort (ascending), the direction logic is:
      // (i & k) == 0 -> Ascending
      // (i & k) != 0 -> Descending
      bool ascending = (i & k) == 0;

      if (ascending) {
        if (a > b) {
          data[i] = b;
          data[ixj] = a;
        }
      } else {
        if (a < b) {
          data[i] = b;
          data[ixj] = a;
        }
      }
    }
  }
}

// data is device pointer
extern "C" void solve(float *data, int N) {
  int M = 1;
  while (M < N)
    M <<= 1;

  float *d_arr = data;
  bool using_padding = (M != N);

  if (using_padding) {
    cudaMalloc(&d_arr, M * sizeof(float));
    cudaMemcpy(d_arr, data, N * sizeof(float), cudaMemcpyDeviceToDevice);

    // Fill padding with INFINITY
    int threads = 256;
    int blocks = (M - N + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(d_arr, N, M, INFINITY);
  }

  // Launch Bitonic Sort Kernels
  // k is the size of the sequence to be sorted (2, 4, ... M)
  int threadsPerBlock = 256;
  int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

  for (int k = 2; k <= M; k <<= 1) {
    // j is the stride (k/2 ... 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonic_sort_step<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, M);
    }
  }

  if (using_padding) {
    // Copy back sorted N elements
    cudaMemcpy(data, d_arr, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_arr);
  }

  cudaDeviceSynchronize();
}
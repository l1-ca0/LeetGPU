#include <cuda_runtime.h>

// Kernel to reverse array in-place.
// Each thread handles one element in the first half of the array.
__global__ void reverse_array(float *input, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Only processed by threads mapping to the first half
  if (idx < N / 2) {
    float temp = input[idx];
    input[idx] = input[N - 1 - idx];
    input[N - 1 - idx] = temp;
  }
}

// input is device pointer
extern "C" void solve(float *input, int N) {
  // No operation needed for 0 or 1 element
  if (N <= 1)
    return;

  int threadsPerBlock = 256;
  // swap N/2 pairs
  int num_swaps = N / 2;
  int blocksPerGrid = (num_swaps + threadsPerBlock - 1) / threadsPerBlock;

  reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
  cudaDeviceSynchronize();
}
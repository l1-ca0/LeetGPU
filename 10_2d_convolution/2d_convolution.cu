#include <cuda_runtime.h>

// Each thread computes one output pixel (out_row, out_col)
__global__ void convolution_2d_kernel(const float *input, const float *kernel,
                                      float *output, int input_rows,
                                      int input_cols, int kernel_rows,
                                      int kernel_cols, int output_rows,
                                      int output_cols) {
  // 1. Map thread indices to output pixel coordinates
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;

  // 2. Check bounds: ensure thread is within valid output dimensions
  if (out_row < output_rows && out_col < output_cols) {
    float sum = 0.0f;

    // 3. Compute dot product between kernel and corresponding input patch
    // "Valid" padding means the kernel window sits 'on top' of the input
    // starting at (out_row, out_col)
    for (int i = 0; i < kernel_rows; ++i) {
      for (int j = 0; j < kernel_cols; ++j) {
        // Calculate input coordinates
        int in_row = out_row + i;
        int in_col = out_col + j;

        // Flatten 2D indices to 1D arrays
        int kernel_idx = i * kernel_cols + j;
        int input_idx = in_row * input_cols + in_col;

        // Accumulate
        sum += input[input_idx] * kernel[kernel_idx];
      }
    }

    // 4. Write result
    int output_idx = out_row * output_cols + out_col;
    output[output_idx] = sum;
  }
}

// input, kernel, output are device pointers
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  int output_rows = input_rows - kernel_rows + 1;
  int output_cols = input_cols - kernel_cols + 1;

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  convolution_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
      output_rows, output_cols);
  cudaDeviceSynchronize();
}
#include <cuda_runtime.h>

#include <math.h>

// Structure to hold max and sum of exponentials
struct __align__(8) MaxSum {
  float max_val;
  float sum_exp;
};

// Device function to merge two MaxSum states
__device__ MaxSum merge_max_sum(MaxSum a, MaxSum b) {
  if (a.max_val == -INFINITY)
    return b;
  if (b.max_val == -INFINITY)
    return a;
  if (a.max_val > b.max_val) {
    return {a.max_val,
            a.sum_exp + b.sum_exp * exp((double)(b.max_val - a.max_val))};
  } else {
    return {b.max_val,
            b.sum_exp + a.sum_exp * exp((double)(a.max_val - b.max_val))};
  }
}

// Pass 1: Local blocks reduction
__global__ void partial_softmax_kernel(const float *input,
                                       MaxSum *partial_results, int N) {
  extern __shared__ MaxSum sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockDim.x * gridDim.x;

  // Grid-Stride Loop initialization
  MaxSum local_ms = {-INFINITY, 0.0f};

  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gridSize) {
    float val = input[i];
    // Merge current value (viewed as a MaxSum with sum_exp=1)
    // Optimization: update local state directly
    if (val > local_ms.max_val) {
      local_ms.sum_exp = local_ms.sum_exp * expf(local_ms.max_val - val) + 1.0f;
      local_ms.max_val = val;
    } else {
      local_ms.sum_exp += expf(val - local_ms.max_val);
    }
  }

  sdata[tid] = local_ms;
  __syncthreads();

  // Block reduction
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = merge_max_sum(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_results[blockIdx.x] = sdata[0];
  }
}

// Pass 2: Final reduction of partial results
__global__ void final_reduction_kernel(MaxSum *partials, MaxSum *global_result,
                                       int num_partials) {
  if (threadIdx.x == 0) {
    MaxSum total_ms = {-INFINITY, 0.0f};
    for (int i = 0; i < num_partials; ++i) {
      total_ms = merge_max_sum(total_ms, partials[i]);
    }
    *global_result = total_ms;
  }
}

// Pass 3: Normalization
__global__ void normalization_kernel(const float *input, float *output,
                                     MaxSum *global_result, int N) {
  float max_val = global_result->max_val;
  float sum_exp = global_result->sum_exp;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    output[idx] = expf(input[idx] - max_val) / sum_exp;
  }
}

extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = 1024;
  if (N < blocksPerGrid * threadsPerBlock) {
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0)
      blocksPerGrid = 1;
  }

  // Allocate device memory for intermediate and final results
  MaxSum *d_partials;
  MaxSum *d_global;
  cudaMalloc(&d_partials, blocksPerGrid * sizeof(MaxSum));
  cudaMalloc(&d_global, sizeof(MaxSum));

  // 1. Partial reduction
  partial_softmax_kernel<<<blocksPerGrid, threadsPerBlock,
                           threadsPerBlock * sizeof(MaxSum)>>>(input,
                                                               d_partials, N);

  // 2. Final reduction
  final_reduction_kernel<<<1, 1>>>(d_partials, d_global, blocksPerGrid);

  // 3. Normalization (using full grid to cover N)
  int norm_blocks = 1024; // Use consistent grid size
  normalization_kernel<<<norm_blocks, threadsPerBlock>>>(input, output,
                                                         d_global, N);

  cudaDeviceSynchronize();
  cudaFree(d_partials);
  cudaFree(d_global);
}
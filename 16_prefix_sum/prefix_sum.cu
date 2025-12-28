#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Helper: Inclusive scan of a block using shared memory (Hillis-Steele)
__device__ void inclusive_scan_block(float *s_data, int idx, int n) {
  for (int stride = 1; stride < n; stride *= 2) {
    float val = 0.0f;
    if (idx >= stride) {
      val = s_data[idx - stride];
    }
    __syncthreads();
    if (idx >= stride) {
      s_data[idx] += val;
    }
    __syncthreads();
  }
}

// Phase 1: Per-block inclusive scan.
// Stores the sum of each block into a separate 'block_sums' array.
__global__ void prescan_kernel(const float *input, float *output,
                               float *block_sums, int N) {
  __shared__ float s_data[BLOCK_SIZE];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  // 1. Load data into shared memory
  if (gid < N) {
    s_data[tid] = input[gid];
  } else {
    s_data[tid] = 0.0f;
  }
  __syncthreads();

  // 2. Perform inclusive scan on the block
  inclusive_scan_block(s_data, tid, BLOCK_SIZE);

  // 3. Write result to output
  if (gid < N) {
    output[gid] = s_data[tid];
  }

  // 4. Write the block's total sum to auxiliary array
  if (tid == BLOCK_SIZE - 1) {
    block_sums[blockIdx.x] = s_data[tid];
  }
}

// Phase 2: Scan the block sums.
// This kernel assumes the number of blocks is small enough to fit in one block
// (<= 1024).
__global__ void scan_block_sums_kernel(float *data, int num_blocks) {
  // Large shared memory buffer to support up to 1024 blocks in a single sweep
  __shared__ float s_data[1024];
  int tid = threadIdx.x;

  // Load
  if (tid < num_blocks) {
    s_data[tid] = data[tid];
  } else {
    s_data[tid] = 0.0f;
  }
  __syncthreads();

  // Scan
  inclusive_scan_block(s_data, tid, 1024);

  // Write back
  if (tid < num_blocks) {
    data[tid] = s_data[tid];
  }
}

// Phase 3: Add the scanned block sums offsets to the output.
__global__ void add_offsets_kernel(float *output, const float *block_sums,
                                   int N) {
  int tid = threadIdx.x;
  int block_id = blockIdx.x;
  int gid = block_id * blockDim.x + tid;

  // Skip the first block (it has no preceding block to add)
  if (block_id > 0 && gid < N) {
    // block_sums is now inclusive prefix sum of blocks.
    // The offset for block `i` is `block_sums[i-1]`.
    output[gid] += block_sums[block_id - 1];
  }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Allocate device memory for block sums
  float *d_block_sums;
  cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));

  // Phase 1: Local Scan (Block-wise)
  // Computes per-block prefix sums and writes total block sums to d_block_sums.
  prescan_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output,
                                                     d_block_sums, N);

  // Phase 2: Scan Block Sums
  // Computes the prefix sums of the block totals.
  // We assume blocksPerGrid <= 1024 (valid for N <= 250k with block size 256).
  // If N were larger, this step would need to be recursive.
  if (blocksPerGrid > 1) {
    int aux_threads = 1024;
    scan_block_sums_kernel<<<1, aux_threads>>>(d_block_sums, blocksPerGrid);
  }

  // Phase 3: Add Offsets
  // Adds the base offset (from previous blocks) to each element.
  if (blocksPerGrid > 1) {
    add_offsets_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, d_block_sums,
                                                           N);
  }

  cudaFree(d_block_sums);
  cudaDeviceSynchronize();
}
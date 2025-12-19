#include <cuda_runtime.h>

// Kernel: Invert RGB channels of RGBA image
// Each thread processes one pixel (4 bytes) efficiently using uchar4
__global__ void invert_kernel(unsigned char *image_raw, int width, int height) {
  int num_pixels = width * height;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Cast to uchar4* to view RGBA pixels
  uchar4 *image = (uchar4 *)image_raw;

  if (idx < num_pixels) {
    uchar4 pixel = image[idx];
    // Invert R, G, B channels
    pixel.x = 255 - pixel.x;
    pixel.y = 255 - pixel.y;
    pixel.z = 255 - pixel.z;
    // Alpha (w) channel is preserved
    image[idx] = pixel;
  }
}

// image is device pointer to flattened RGBA data
extern "C" void solve(unsigned char *image, int width, int height) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

  invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
  cudaDeviceSynchronize();
}
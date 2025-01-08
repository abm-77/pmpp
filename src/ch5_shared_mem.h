#pragma once

/*
 * matmul_sqr_opt is a more optimized matrix multiplication kernel for square
 *matrices. It makes use of faster shared memory and computes the output via
 *tiling.
 **/
constexpr int TILE_WIDTH = 16;
__global__ void matmul_sqr_opt(float *A, float *B, float *C, int width) {
  // allocate shared memory for matrix tiles
  __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

  // setup register variables
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;
  int row = block_y * TILE_WIDTH + thread_y;
  int col = block_x * TILE_WIDTH + thread_x;

  float val = 0;
  for (int phase = 0; phase < ceil(width / (float)TILE_WIDTH); phase++) {
    // load data from matrices into the shared memory buffer
    if (row < width && (phase * TILE_WIDTH + thread_x)) {
      Ads[thread_y][thread_x] =
          A[row * width + (phase * TILE_WIDTH + thread_x)];
    } else {
      Ads[thread_y][thread_x] = 0;
    }

    if ((phase * TILE_WIDTH + thread_y) < width && col < width) {
      Bds[thread_y][thread_x] =
          B[(phase * TILE_WIDTH + thread_y) * width + col];
    } else {
      Bds[thread_y][thread_x] = 0;
    }
    __syncthreads();

    // compute dot product of row of A and col of B reading from shared memory
    for (int k = 0; k < TILE_WIDTH; k++) {
      val += Ads[thread_y][k] * Bds[k][thread_x];
    }
    __syncthreads();
  }

  // boundary check for output array
  if (row < width && col < width)
    C[row * width + col] = val;
}

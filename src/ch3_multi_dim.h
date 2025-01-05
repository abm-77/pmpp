#pragma once
#include <assert.h>

// greyscale is a multi dimensional grey scale kernel takes an w x h RGB image
// Pin and outputs a greyscale image Pout.
using byte = unsigned char;
constexpr int CHANNELS = 3;
__global__ void greyscale(byte *Pout, byte *Pin, int w, int h) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px < w && py < h) {
    int grey_off = py * w + px;
    int rgb_off = grey_off * CHANNELS;

    byte r = Pin[rgb_off];
    byte g = Pin[rgb_off + 1];
    byte b = Pin[rgb_off + 2];
    Pout[grey_off] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

// matmul_sqr is a multidimensional matrix multiplication kernel for square
// matrices.
__global__ void matmul_sqr(float *A, float *B, float *C, int w) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < w && col < w) {
    int C_rc = 0;
    for (int k = 0; k < w; k++) {
      C_rc += A[row * w + k] * B[row * k + col];
    }
    C[row * w + col] = C_rc;
  }
}

// matmul is a general multidimensional matrix multiplication kernel.
__global__ void matmul_k(float *A, float *B, float *C, int A_r, int A_c,
                         int B_r, int B_c) {
  int row_size = A_r;
  int col_size = B_c;
  int inner_size = A_c;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < row_size && col < col_size) {
    int C_rc = 0;
    for (int k = 0; k < inner_size; k++) {
      C_rc += A[row * inner_size + k] * B[row * k + col];
    }
    C[row * row_size + col] = C_rc;
  }
}

void matmul(float *A_h, float *B_h, float *C_h, int A_r, int A_c, int B_r,
            int B_c) {
  assert(A_c == B_r);
  float *A_d, *B_d, *C_d;
  int A_size = A_r * A_c * sizeof(float);
  int B_size = B_r * B_c * sizeof(float);
  int C_size = A_r * B_c * sizeof(float);

  cudaMalloc((void **)&A_d, A_size);
  cudaMalloc((void **)&B_d, B_size);
  cudaMalloc((void **)&C_d, C_size);

  cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);

  dim3 dim_grid(ceil(A_r / 16.0), ceil(B_c / 16.0), 1);
  dim3 dim_block(16, 16, 1);
  matmul_k<<<dim_grid, dim_block>>>(A_d, B_d, C_d, A_r, A_c, B_r, B_c);

  cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

constexpr int BLUR_SIZE = 1;
__global__ void blur_k(byte *in, byte *out, int w, int h) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockIdx.x + threadIdx.x;
  if (row < h && col < w) {
    int n_pixels = 0;
    int pixel_sum = 0;
    for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; blur_row++) {
      for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; blur_col++) {
        int curr_row = row + blur_row;
        int curr_col = col + blur_col;
        if (curr_row >= 0 && curr_row < h && curr_col >= 0 && curr_col < w) {
          pixel_sum += in[curr_row * w + curr_col];
          n_pixels++;
        }
      }
    }
    out[row * w + col] = (byte)(pixel_sum / n_pixels);
  }
}

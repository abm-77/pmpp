#include "ch2_intro.h"
#include "ch3_multi_dim.h"
#include <iostream>

int main(void) {
  int N = 5;
  float A[5] = {1, 2, 3, 4, 5};
  float B[5] = {1, 2, 3, 4, 5};
  float C[5] = {0};

  // vec_add(A, B, C, N);
  matmul(A, B, C, 1, 5, 5, 1);

  for (int i = 0; i < N; i++) {
    std::cout << C[i] << std::endl;
  }
}

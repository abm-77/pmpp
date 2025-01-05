#include "ch2_vec_add.h"
#include <iostream>

int main(void) {
  int N = 5;
  float A[5] = {1, 2, 3, 4, 5};
  float B[5] = {1, 2, 3, 4, 5};
  float C[5] = {0};

  vec_add(A, B, C, N);

  for (int i = 0; i < N; i++) {
    std::cout << C[i] << std::endl;
  }
}

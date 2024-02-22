//===- matmul.c - Try ScopMatcher -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include "matmul.h"

#ifdef TEST
#define L 5
#define M 7
#define N 9
#else
#define L 160
#define M 128
#define N 256
#endif

// Simulates a LxM = LxN * NxM element mat mat mul

// R by C (RxC) matrix ... R rows , C columns
// row major
// R outer, C inner

TEST_TYPE Z[L][M];
TEST_TYPE X[L][N];
TEST_TYPE XT[N][L];
TEST_TYPE Y[N][M];
TEST_TYPE YT[M][N];
TEST_TYPE R[L][M];

void compare(INT_TYPE m,
             INT_TYPE n,
             TEST_TYPE a[restrict m][n],
             TEST_TYPE b[restrict m][n]) {
  int has_diff = 0;
  for (INT_TYPE i = 0; i < m; i++) {
    for (INT_TYPE j = 0; j < n; j++) {
      TEST_TYPE c = a[i][j] - b[i][j];
      TEST_TYPE tolerance = 0.001;
      has_diff |= c > tolerance || -c > tolerance;
    }
  }
  if (has_diff)
    fprintf(stdout, "OOOOOOOOOOOOOOOO found a difference OOOOOOOOOOOOOOOO\n\n");
}

void init_array_simple(INT_TYPE m,
                       INT_TYPE n,
                       TEST_TYPE c[restrict m][n],
                       TEST_TYPE v,
                       TEST_TYPE inc) {
  for (INT_TYPE i = 0; i < m; i++) {
    for (INT_TYPE j = 0; j < n; j++) {
      c[i][j]= v;
      v += inc;
      if (v > 123.5) {
        v -= 123.5;
      }
    }
  }
}

void print_array(INT_TYPE m,
                 INT_TYPE n,
                 TEST_TYPE c[restrict m][n]) {
  for (INT_TYPE i = 0; i < m; i++) {
    for (INT_TYPE j = 0; j < n; j++) {
      fprintf(stdout, FORMAT, c[i][j]);
    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "\n");
}

void transpose(INT_TYPE m,
               INT_TYPE n,
               TEST_TYPE in[restrict m][n],
               TEST_TYPE out[restrict n][m]) {
  for (INT_TYPE i = 0; i < m; i++) {
    for (INT_TYPE j = 0; j < n; j++) {
      out[j][i] = in[i][j];
    }
  }
}

void kernel_matmul_init_outside(INT_TYPE l,
                                INT_TYPE m,
                                INT_TYPE n,
                                TEST_TYPE c[restrict l][m],
                                TEST_TYPE a[restrict l][n],
                                TEST_TYPE b[restrict n][m]) {
  for (INT_TYPE i = 0; i < l; i++) {
     for (INT_TYPE j = 0; j < m; j++) {
       c[i][j] = 0;
     }
  }
  for (INT_TYPE i = 0; i < l; i++) {
    for (INT_TYPE j = 0; j < m; j++) {
      for (INT_TYPE k = 0; k < n; k++) {
        c[i][j] = c[i][j] + a[i][k] * b[k][j];
      }
    }
  }
}

void kernel_matmul_init_inside(INT_TYPE l,
                               INT_TYPE m,
                               INT_TYPE n,
                               TEST_TYPE c[restrict l][m],
                               TEST_TYPE a[restrict l][n],
                               TEST_TYPE b[restrict n][m]) {
  for (INT_TYPE i = 0; i < l; i++) {
    for (INT_TYPE j = 0; j < m; j++) {
      c[i][j] = 0;
      for (INT_TYPE k = 0; k < n; k++) {
        c[i][j] = c[i][j] + a[i][k] * b[k][j];
      }
    }
  }
}

void outer_product(INT_TYPE l,
                   INT_TYPE m,
                   TEST_TYPE  c[restrict l][m],
                   TEST_TYPE at[restrict l],
                   TEST_TYPE  b[restrict m]) {
  for (INT_TYPE i = 0; i < l; i++) {
    for (INT_TYPE j = 0; j < m; j++) {
      c[i][j] += at[i] * b[j];
    }
  }
}

void left_transposed_matmul(INT_TYPE l,
                            INT_TYPE m,
                            INT_TYPE n,
                            TEST_TYPE c[restrict l][m],
                            TEST_TYPE at[restrict n][l],
                            TEST_TYPE b[restrict n][m]) {
  for (INT_TYPE i = 0; i < l; i++) {
    for (INT_TYPE j = 0; j < m; j++) {
      c[i][j] = 0;
    }
  }
  for (INT_TYPE k = 0; k < n; k++) {
    for (INT_TYPE i = 0; i < l; i++) {
      for (INT_TYPE j = 0; j < m; j++) {
        c[i][j] += at[k][i] * b[k][j];
      }
    }
  }
}

void kernel_transposed_matmul_init_outside(INT_TYPE l,
                                           INT_TYPE m,
                                           INT_TYPE n,
                                           TEST_TYPE c[restrict l][m],
                                           TEST_TYPE a[restrict l][n],
                                           TEST_TYPE bt[restrict m][n]) {
  for (INT_TYPE i = 0; i < l; i++) {
    for (INT_TYPE j = 0; j < m; j++) {
      c[i][j] = 0;
    }
  }
  for (INT_TYPE i = 0; i < l; i++) {
    for (INT_TYPE j = 0; j < m; j++) {
      for (INT_TYPE k = 0; k < n; k += 1) {
        c[i][j] += a[i][k] * bt[j][k];
      }
    }
  }
}

void kernel_transposed_matmul_init_inside(INT_TYPE l,
                                          INT_TYPE m,
                                          INT_TYPE n,
                                          TEST_TYPE c[restrict l][m],
                                          TEST_TYPE a[restrict l][n],
                                          TEST_TYPE bt[restrict m][n]) {
  for (INT_TYPE i = 0; i < l; i++) {
    for (INT_TYPE j = 0; j < m; j++) {
      c[i][j] = 0;
      for (INT_TYPE k = 0; k < n; k += 1) {
        c[i][j] += a[i][k] * bt[j][k];
      }
    }
  }
}

void impl_matmul_lmn(INT_TYPE l,
                     INT_TYPE m,
                     INT_TYPE n,
                     TEST_TYPE * restrict C,
                     TEST_TYPE * restrict A,
                     TEST_TYPE * restrict B,
                     INT_TYPE stride_a,
                     INT_TYPE stride_b,
                     TEST_TYPE alpha);

void cblas_matmul(INT_TYPE l,
                  INT_TYPE m,
                  INT_TYPE n,
                  TEST_TYPE c[restrict l][m],
                  TEST_TYPE a[restrict l][n],
                  TEST_TYPE b[restrict n][m]) {
  impl_matmul_lmn(l, m, n, c, a, b, n, m, 1.0);
}

void try_axpy() {
  int n = 16;
  TEST_TYPE C[1][n];
  init_array_simple(1, n, C, 1, 2);
  print_array(1, n, C);
  TEST_TYPE A[1][n];
  init_array_simple(1, n, A, 2, 2);
  print_array(1, n, A);
  TEST_TYPE B = 5;
  impl_axpy_n(n, C, A, &B);
  print_array(1, n, C);
}

int main() {
  init_array_simple(L, N, X, INIT_VAL_1, INC_VAL);
  init_array_simple(N, M, Y, INIT_VAL_2, INC_VAL);
  init_array_simple(L, M, Z, ZERO, ZERO);

  init_array_simple(L, M, R, ZERO, ZERO);

  // print_array(L, N, X);
  // print_array(N, M, Y);
  // print_array(L, M, Z);
  // print_array(L, M, R);

  cblas_matmul(L, M, N, R, X, Y);
  // print_array(L, M, R);

  kernel_matmul_init_outside(L, M, N, Z, X, Y);
  // print_array(L, M, Z);
  compare(L, M, R, Z);
  init_array_simple(L, M, Z, ZERO, ZERO);

  kernel_matmul_init_inside(L, M, N, Z, X, Y);
  // print_array(L, M, Z);
  compare(L, M, R, Z);
  init_array_simple(L, M, Z, ZERO, ZERO);

  init_array_simple(M, N, YT, ZERO, ZERO);
  transpose(N, M, Y, YT);
  // print_array(M, N, YT);

  kernel_transposed_matmul_init_outside(L, M, N, Z, X, YT);
  // print_array(L, M, Z);
  compare(L, M, R, Z);
  init_array_simple(L, M, Z, ZERO, ZERO);

  kernel_transposed_matmul_init_inside(L, M, N, Z, X, YT);
  // print_array(L, M, Z);
  compare(L, M, R, Z);
  init_array_simple(L, M, Z, ZERO, ZERO);

  init_array_simple(N, L, XT, ZERO, ZERO);
  transpose(L, N, X, XT);
  // print_array(N, L, XT);

  left_transposed_matmul(L, M, N, Z, XT, Y);
  // print_array(L, M, Z);
  compare(L, M, R, Z);
  init_array_simple(L, M, Z, ZERO, ZERO);

  try_axpy();

  return 0;
}

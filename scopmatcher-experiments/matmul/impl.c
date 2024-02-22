//===- impl.c - Implementations for the ScopMatcherEmitter ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Use optimized target implementations for the experiments, e.g., OpenBLAS
//
//===----------------------------------------------------------------------===//

// #include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "matmul.h"

// cblas_sgemm example
// http://gnu.ist.utl.pt/software/gsl/manual/html_node/GSL-CBLAS-Examples.html

// void cblas_sgemm(
//   OPENBLAS_CONST enum CBLAS_ORDER Order,
//   OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
//   OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
//   OPENBLAS_CONST blasint M,
//   OPENBLAS_CONST blasint N,
//   OPENBLAS_CONST blasint K,
//   OPENBLAS_CONST float alpha,
//   OPENBLAS_CONST float *A,
//   OPENBLAS_CONST blasint lda,
//   OPENBLAS_CONST float *B,
//   OPENBLAS_CONST blasint ldb,
//   OPENBLAS_CONST float beta,
//   float *C,
//   OPENBLAS_CONST blasint ldc);

void cblas_sgemm(
  unsigned Order,
  unsigned TransA,
  unsigned TransB,
  long M, // Number of rows in matrices A and C
  long N, // Number of columns in matrices B and C
  long K, // Number of columns in matrix A; number of rows in matrix B
  float alpha,
  float *A,
  long lda,   // If not transposed and row major: number of columns of A
  float *B,
  long ldb,   // If not transposed and row major: number of columns of B
  float beta,
  float *C,
  long ldc);  // If not transposed and row major: number of columns of C

// Simulates a LxM = LxN * NxM element mat mat mul
void impl_matmul(TEST_TYPE * restrict C,
                 TEST_TYPE * restrict A,
                 TEST_TYPE * restrict B,
                 INT_TYPE stride_a,
                 INT_TYPE stride_b) {
  cblas_sgemm(101, 111, 111, SL, SM, SN, 1.0, A, stride_a, B, stride_b, 1.0, C, stride_b);
}

// Simulates a LxM = LxN * NxM element mat mat mul
// stride_a is n
// stride_b is m
void impl_matmul_lmn(INT_TYPE l,
                     INT_TYPE m,
                     INT_TYPE n,
                     TEST_TYPE * restrict C,
                     TEST_TYPE * restrict A,
                     TEST_TYPE * restrict B,
                     INT_TYPE stride_a,
                     INT_TYPE stride_b,
                     TEST_TYPE alpha) {
  cblas_sgemm(101, 111, 111, l, m, n, alpha, A, stride_a, B, stride_b, 1.0, C, stride_b);
}

float cblas_sdot(long n, float *x, long incx, float *y, long incy);

void impl_dotprod_n(INT_TYPE n,
                    TEST_TYPE * restrict c,
                    TEST_TYPE * restrict a,
                    TEST_TYPE * restrict b) {
  float cblas_result = 0.0;
  cblas_result = cblas_sdot(n, a, 1, b, 1);
  *c = *c + cblas_result;
}

void cblas_saxpy(long n, float alpha, float *x, long incx, float *y, long incy);

void impl_axpy_n(INT_TYPE n,
                 TEST_TYPE * restrict c,
                 TEST_TYPE * restrict a,
                 TEST_TYPE * restrict b) {
  cblas_saxpy(n, *b, a, 1, c, 1);
}

// void cblas_sgemv(
//   OPENBLAS_CONST enum CBLAS_ORDER order,
//   OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,
//   OPENBLAS_CONST blasint m,
//   OPENBLAS_CONST blasint n,
//   OPENBLAS_CONST float alpha,
//   OPENBLAS_CONST float *a,
//   OPENBLAS_CONST blasint lda,
//   OPENBLAS_CONST float *x,
//   OPENBLAS_CONST blasint incx,
//   OPENBLAS_CONST float beta,
//   float *y,
//   OPENBLAS_CONST blasint incy
// );

void cblas_sgemv(
  unsigned order,
  unsigned trans,
  long r,
  long c,
  float alpha,
  float *a,
  long lda,
  float *x,
  long incx,
  float beta,
  float *y,
  long incy
);

// stride_b is m in the parameterized case
void impl_vect_matmul_mn(INT_TYPE m,
                         INT_TYPE n,
                         TEST_TYPE * restrict C,
                         TEST_TYPE * restrict A,
                         TEST_TYPE * restrict B,
                         INT_TYPE stride_b,
                         TEST_TYPE Alpha) {
  cblas_sgemv(101, 112, n, m, Alpha, B, stride_b, A, 1, 1.0, C, 1);
}

#ifdef USE_AMX
typedef struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __tilecfg;

// Simulates a LxM = LxN * NxM element mat mat mul
void impl_matmul_amx(
    unsigned l,
    unsigned m,
    unsigned n,
    uint32_t * restrict C,
    uint32_t * restrict A,
    uint32_t * restrict B,
    unsigned stride_a,
    unsigned stride_b) {
  __tilecfg tileinfo = {0};
  tileinfo.palette_id = 1;
  tileinfo.start_row = 0;
  // FIXME I do not know why
  //       we initialize tile 0.
  tileinfo.rows[0] =  16;
  tileinfo.colsb[0] = 16;
  // C
  tileinfo.rows[1] = l;
  tileinfo.colsb[1] = m*4;
  // A
  tileinfo.rows[2] =  l;
  tileinfo.colsb[2] = n*4;
  // B
  tileinfo.rows[3] = n;
  tileinfo.colsb[3] = m*4;
  _tile_loadconfig (&tileinfo);

  _tile_loadd(1, C, stride_b*4);
  _tile_loadd(2, A, stride_a*4);
  _tile_loadd(3, B, stride_b*4);
  _tile_dpbuud(1, 2, 3);
  _tile_stored(1, C, stride_b*4);
  _tile_release();
}
#endif

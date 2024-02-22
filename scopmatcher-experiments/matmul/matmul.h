//===- matmul.h - Config for the ScopMatcher experiments -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #define TEST 1

#define SL 64
#define SM 64
#define SN 64

// #define USE_INT 1

#ifdef USE_INT
#define TEST_TYPE unsigned
#define INIT_VAL_1 1
#define INIT_VAL_2 2
#define INC_VAL 2
#define ZERO 0
#define FORMAT "%6u"
#else
#define TEST_TYPE float
#define INIT_VAL_1 1.0
#define INIT_VAL_2 2.0
#define INC_VAL 2.0
#define ZERO 0.0
#define FORMAT "%6.2f "
#endif

#define INT_TYPE long

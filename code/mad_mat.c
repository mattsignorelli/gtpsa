/*
 o-----------------------------------------------------------------------------o
 |
 | Matrix module implementation
 |
 | Methodical Accelerator Design - Copyright (c) 2016+
 | Support: http://cern.ch/mad  - mad at cern.ch
 | Authors: L. Deniau, laurent.deniau at cern.ch
 | Contrib: -
 |
 o-----------------------------------------------------------------------------o
 | You can redistribute this file and/or modify it under the terms of the GNU
 | General Public License GPLv3 (or later), as published by the Free Software
 | Foundation. This file is distributed in the hope that it will be useful, but
 | WITHOUT ANY WARRANTY OF ANY KIND. See http://gnu.org/licenses for details.
 o-----------------------------------------------------------------------------o
*/

#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <assert.h>

#include "mad_log.h"
#include "mad_mem.h"
#include "mad_vec.h"
#include "mad_mat.h"

#define MAD_USE_MADX 0

// --- helpers for debug ------------------------------------------------------o

#if 0
#include <stdio.h>

static inline void __attribute__((unused))
mprint(str_t name, const num_t a[], ssz_t m, ssz_t n)
{
  printf("%s[%dx%d]=\n", name, m, n);
  FOR(i,m) { FOR(j,n) printf("% -.5e ", a[i*n+j]); printf("\n"); }
}

static inline void __attribute__((unused))
iprint(str_t name, const idx_t a[], ssz_t m, ssz_t n)
{
  printf("%s[%dx%d]=\n", name, m, n);
  FOR(i,m) { FOR(j,n) printf("% 3d ", a[i*n+j]); printf("\n"); }
}
#endif

// --- implementation ---------------------------------------------------------o

#define CHKR   assert( r )
#define CHKX   assert( x )
#define CHKXY  assert( x && y )
#define CHKXR  assert( x && r )
#define CHKYR  assert( y && r )
#define CHKXYR assert( x && y && r )
#define CHKXRX assert( x && r && x != r)

#define CPX(re,im) (* (cpx_t*) & (num_t[2]) { re, im })

// --- matrix, cmatrix, imatrix

struct  matrix { ssz_t nr, nc; num_t data[]; };
struct cmatrix { ssz_t nr, nc; cpx_t data[]; };
struct imatrix { ssz_t nr, nc; idx_t data[]; };

// Note: matrix of zero size are forbidden

void mad_mat_reshape (struct matrix *x, ssz_t m, ssz_t n)
{ CHKX; x->nr = MAX(1,m); x->nc = MAX(1,n); }

void mad_cmat_reshape (struct cmatrix *x, ssz_t m, ssz_t n)
{ CHKX; x->nr = MAX(1,m); x->nc = MAX(1,n); }

void mad_imat_reshape (struct imatrix *x, ssz_t m, ssz_t n)
{ CHKX; x->nr = MAX(1,m); x->nc = MAX(1,n); }

// -----

#if 1
// r[m x n] = x[m x p] * y[p x n]
// naive implementation (more efficient on recent superscalar arch!)
#define MMUL() /* mat * mat */ \
  FOR(i,m*n) r[i] = 0; \
  FOR(i,m) FOR(k,p) FOR(j,n) r[i*n+j] += x[i*p+k] * y[k*n+j];
#else
// [m x n] = [m x p] * [p x n]
// portable vectorized general matrix-matrix multiplication
// loop unroll + vectorized on SSE2 (x2), AVX & AVX2 (x4), AVX-512 (x8)
#define MMUL() { /* mat * mat */ \
  assert(m>0 && n>0 && p>0); \
  if (n & ~7) { \
    FOR(i,m) FOR(j,0,n-7,8) { \
      r[i*n+j  ] = r[i*n+j+1] = \
      r[i*n+j+2] = r[i*n+j+3] = \
      r[i*n+j+4] = r[i*n+j+5] = \
      r[i*n+j+6] = r[i*n+j+7] = 0; \
      FOR(k,p) { \
        r[i*n+j  ] += x[i*p+k] * y[k*n+j  ]; \
        r[i*n+j+1] += x[i*p+k] * y[k*n+j+1]; \
        r[i*n+j+2] += x[i*p+k] * y[k*n+j+2]; \
        r[i*n+j+3] += x[i*p+k] * y[k*n+j+3]; \
        r[i*n+j+4] += x[i*p+k] * y[k*n+j+4]; \
        r[i*n+j+5] += x[i*p+k] * y[k*n+j+5]; \
        r[i*n+j+6] += x[i*p+k] * y[k*n+j+6]; \
        r[i*n+j+7] += x[i*p+k] * y[k*n+j+7]; \
      } \
    } \
  } \
  if (n & 4) { \
    idx_t j = n - (n & 7); \
    FOR(i,m) { \
      r[i*n+j  ] = r[i*n+j+1] = \
      r[i*n+j+2] = r[i*n+j+3] = 0; \
      FOR(k,p) { \
        r[i*n+j  ] += x[i*p+k] * y[k*n+j  ]; \
        r[i*n+j+1] += x[i*p+k] * y[k*n+j+1]; \
        r[i*n+j+2] += x[i*p+k] * y[k*n+j+2]; \
        r[i*n+j+3] += x[i*p+k] * y[k*n+j+3]; \
      } \
    } \
  } \
  if (n & 2) { \
    idx_t j = n - (n & 3); \
    FOR(i,m) { \
      r[i*n+j] = r[i*n+j+1] = 0; \
      FOR(k,p) { \
        r[i*n+j  ] += x[i*p+k] * y[k*n+j  ]; \
        r[i*n+j+1] += x[i*p+k] * y[k*n+j+1]; \
      } \
    } \
  } \
  if (n & 1) { \
    idx_t j = n - 1; \
    FOR(i,m) { \
      r[i*n+j] = 0; \
      FOR(k,p) r[i*n+j] += x[i*p+k] * y[k*n+j]; \
    } \
  } \
}
#endif

// n=1: [m x 1] = [m x p] * [p x 1]
#define MULV() /* mat * vec */ \
  FOR(i,m) r[i] = 0; \
  FOR(i,m) FOR(k,p) r[i] += x[i*p+k] * y[k];

// m=1: [1 x n] = [1 x p] * [p x n]
#define VMUL() /* vec * mat */ \
  FOR(j,n) r[j] = 0; \
  FOR(k,p) FOR(j,n) r[j] += y[k*n+j] * x[k];

// m=1, n=1: [1 x 1] = [1 x p] * [p x 1]
#define IMUL() /* vec * vec */ \
  r[0] = 0; \
  FOR(k,p) r[0] += x[k] * y[k];

// [m x n] = [m x p] * [p x n]
#define MUL() \
  switch(((m == 1) << 1) & (n == 1)) { \
    case 0: MMUL(); break; \
    case 1: MULV(); break; \
    case 2: VMUL(); break; \
    case 3: IMUL(); break; \
  }

// -----

#if 1
// [m x n] = [p x m]' * [p x n]
// naive implementation (more efficient on recent superscalar arch!)
#define TMMUL(C) /* mat' * mat */ \
  FOR(i,m*n) r[i] = 0; \
  FOR(i,m) FOR(k,p) FOR(j,n) r[i*n+j] += C(x[k*m+i]) * y[k*n+j];
#else
// [m x n] = [p x m]' * [p x n]
// portable vectorized general transpose matrix-matrix multiplication
// loop unroll + vectorized on SSE2 (x2), AVX & AVX2 (x4), AVX-512 (x8)
#define TMMUL(C) { /* mat' * mat */ \
  assert(m>0 && n>0 && p>0); \
  if (n & ~7) { \
    FOR(i,m) FOR(j,0,n-7,8) { \
      r[i*n+j  ] = r[i*n+j+1] = \
      r[i*n+j+2] = r[i*n+j+3] = \
      r[i*n+j+4] = r[i*n+j+5] = \
      r[i*n+j+6] = r[i*n+j+7] = 0; \
      FOR(k,p) { \
        r[i*n+j  ] += C(x[k*m+i]) * y[k*n+j  ]; \
        r[i*n+j+1] += C(x[k*m+i]) * y[k*n+j+1]; \
        r[i*n+j+2] += C(x[k*m+i]) * y[k*n+j+2]; \
        r[i*n+j+3] += C(x[k*m+i]) * y[k*n+j+3]; \
        r[i*n+j+4] += C(x[k*m+i]) * y[k*n+j+4]; \
        r[i*n+j+5] += C(x[k*m+i]) * y[k*n+j+5]; \
        r[i*n+j+6] += C(x[k*m+i]) * y[k*n+j+6]; \
        r[i*n+j+7] += C(x[k*m+i]) * y[k*n+j+7]; \
      } \
    } \
  } \
  if (n & 4) { \
    idx_t j = n - (n & 7); \
    FOR(i,m) { \
      r[i*n+j  ] = r[i*n+j+1] = \
      r[i*n+j+2] = r[i*n+j+3] = 0; \
      FOR(k,p) { \
        r[i*n+j  ] += C(x[k*m+i]) * y[k*n+j  ]; \
        r[i*n+j+1] += C(x[k*m+i]) * y[k*n+j+1]; \
        r[i*n+j+2] += C(x[k*m+i]) * y[k*n+j+2]; \
        r[i*n+j+3] += C(x[k*m+i]) * y[k*n+j+3]; \
      } \
    } \
  } \
  if (n & 2) { \
    idx_t j = n - (n & 3); \
    FOR(i,m) { \
      r[i*n+j] = r[i*n+j+1] = 0; \
      FOR(k,p) { \
        r[i*n+j  ] += C(x[k*m+i]) * y[k*n+j  ]; \
        r[i*n+j+1] += C(x[k*m+i]) * y[k*n+j+1]; \
      } \
    } \
  } \
  if (n & 1) { \
    idx_t j = n - 1; \
    FOR(i,m) { \
      r[i*n+j] = 0; \
      FOR(k,p) r[i*n+j] += C(x[k*m+i]) * y[k*n+j]; \
    } \
  } \
}
#endif

// n=1: [m x 1] = [p x m]' * [p x 1]
#define TMULV(C) /* mat' * vec */ \
  FOR(i,m) r[i] = 0; \
  FOR(k,p) FOR(i,m) r[i] += C(x[k*m+i]) * y[k];

// m=1: [1 x n] = [p x 1]' * [p x n]
#define TVMUL(C) /* vec' * mat */ \
  FOR(j,n) r[j] = 0; \
  FOR(k,p) FOR(j,n) r[j] += C(x[k]) * y[k*n+j];

// m=1, n=1: [1 x 1] = [p x 1]' * [p x 1]
#define TIMUL(C) /* vec' * vec */ \
  r[0]= 0; FOR(k,p) r[0] += C(x[k]) * y[k];

// [m x n] = [p x m]' * [p x n]
#define TMUL(C) \
  switch(((m == 1) << 1) & (n == 1)) { \
    case 0: TMMUL(C); break; \
    case 1: TMULV(C); break; \
    case 2: TVMUL(C); break; \
    case 3: TIMUL(C); break; \
  }

// -----

#if 1
// [m x n] = [m x p] * [n x p]'
// naive implementation (more efficient on recent superscalar arch!)
#define MMULT(C) /* mat * mat' */ \
  FOR(i,m*n) r[i] = 0; \
  FOR(i,m) FOR(j,n) FOR(k,p) r[i*n+j] += x[i*p+k] * C(y[j*p+k]);
#else
// [m x n] = [m x p] * [n x p]'
// portable vectorized general transpose matrix-matrix multiplication
// loop unroll + vectorized on SSE2 (x2), AVX & AVX2 (x4), AVX-512 (x8)
#define MMULT(C) { /* mat * mat' */ \
  assert(m>0 && n>0 && p>0); \
  FOR(i,m*n) r[i] = 0; \
  if (n & ~7) { \
    FOR(i,m) FOR(j,n) FOR(k,0,p-7,8) { \
      r[i*n+j] += x[i*p+k  ] * C(y[j*p+k  ]); \
      r[i*n+j] += x[i*p+k+1] * C(y[j*p+k+1]); \
      r[i*n+j] += x[i*p+k+2] * C(y[j*p+k+2]); \
      r[i*n+j] += x[i*p+k+3] * C(y[j*p+k+3]); \
      r[i*n+j] += x[i*p+k+4] * C(y[j*p+k+4]); \
      r[i*n+j] += x[i*p+k+5] * C(y[j*p+k+5]); \
      r[i*n+j] += x[i*p+k+6] * C(y[j*p+k+6]); \
      r[i*n+j] += x[i*p+k+7] * C(y[j*p+k+7]); \
    } \
  } \
  if (p & 4) { \
    idx_t k = p - (p & 7); \
    FOR(i,m) FOR(j,n) { \
      r[i*n+j] += x[i*p+k  ] * C(y[j*p+k  ]); \
      r[i*n+j] += x[i*p+k+1] * C(y[j*p+k+1]); \
      r[i*n+j] += x[i*p+k+2] * C(y[j*p+k+2]); \
      r[i*n+j] += x[i*p+k+3] * C(y[j*p+k+3]); \
    } \
  } \
  if (p & 2) { \
    idx_t k = p - (p & 3); \
    FOR(i,m) FOR(j,n) { \
      r[i*n+j] += x[i*p+k  ] * C(y[j*p+k  ]); \
      r[i*n+j] += x[i*p+k+1] * C(y[j*p+k+1]); \
    } \
  } \
  if (p & 1) { \
    idx_t k = p - 1; \
    FOR(i,m) FOR(j,n) r[i*n+j] += x[i*p+k] * C(y[j*p+k]); \
  } \
}
#endif

// n=1: [m x 1] = [m x p] * [1 x p]'
#define MULVT(C) /* mat * vec' */ \
  FOR(i,m) r[i] = 0; \
  FOR(i,m) FOR(k,p) r[i] += x[i*p+k] * C(y[k]);

// m=1: [1 x n] = [1 x p]' * [n x p]'
#define VMULT(C) /* vec * mat' */ \
  FOR(j,n) r[j] = 0; \
  FOR(j,n) FOR(k,p) r[j] += x[k] * C(y[j*p+k]);

// m=1, n=1: [1 x 1] = [1 x p] * [1 x p]'
#define IMULT(C) /* vec * vec' */ \
  r[0]= 0; FOR(k,p) r[0] += x[k] * C(y[k]);

// [m x n] = [m x p] * [n x p]'
#define MULT(C) \
  switch(((m == 1) << 1) & (n == 1)) { \
    case 0: MMULT(C); break; \
    case 1: MULVT(C); break; \
    case 2: VMULT(C); break; \
    case 3: IMULT(C); break; \
  }

// -----

// r[m x n] = diag(x[m x p]) * y[p x n]
// naive implementation (more efficient on recent superscalar arch!)
#define DMUL() /* diag(mat) * mat */ \
  if (p == 1) { \
    FOR(i,m) FOR(j,n) r[i*n+j] = x[i] * y[i*n+j]; \
  } else { \
    FOR(i,m*n) r[i] = 0; \
    FOR(i,MIN(m,p)) FOR(j,n) r[i*n+j] = x[i*p+i] * y[i*n+j]; \
  }

// r[m x n] = x[m x p] * diag(y[p x n])
// naive implementation (more efficient on recent superscalar arch!)
#define MULD() /* mat * diag(mat) */ \
  if (p == 1) { \
    FOR(i,m) FOR(j,n) r[i*n+j] = x[i*n+j] * y[j]; \
  } else { \
    FOR(i,m*n) r[i] = 0; \
    FOR(i,m) FOR(j,MIN(n,p)) r[i*n+j] = x[i*p+j] * y[j*n+j]; \
  }

// -----

// [m x n] transpose
#define TRANS(T,C) \
  if (m == 1 || n == 1) { \
    if (x != r || I != C(I)) \
      FOR(i,m*n) r[i] = C(x[i]); \
  } else if ((const void*)x != (const void*)r) { \
    FOR(i,m) FOR(j,n) r[j*m+i] = C(x[i*n+j]); \
  } else if (m == n) { \
    FOR(i,m) FOR(j,i,n) { \
      T t = C(r[j*m+i]); \
      r[j*m+i] = C(r[i*n+j]); \
      r[i*n+j] = t; \
    } \
  } else { \
    mad_alloc_tmp(T, t, m*n); \
    FOR(i,m) FOR(j,n) t[j*m+i] = C(x[i*n+j]); \
    memcpy(r, t, m*n*sizeof(T)); \
    mad_free_tmp(t); \
  };

// -----

// [m x n] copy [+ op]
#define CPY(OP) FOR(i,m) FOR(j,n) r[i*ldr+j] OP##= x[i*ldx+j]

// [m x n] set [+ op]
#define SET(OP) FOR(i,m) FOR(j,n) r[i*ldr+j] OP##= x

// [m x n] sequence [+ op]
#define SEQ(OP) FOR(i,m) FOR(j,n) r[i*ldr+j] OP##= (i*ldr+j)+x

// [m x n] diagonal [+ op]
#define DIAG(OP) FOR(i, MIN(m,n)) r[i*ldr+i] OP##= x

// --- mat

void mad_mat_eye (num_t r[], num_t v, ssz_t m, ssz_t n, ssz_t ldr)
{ CHKR; num_t x = 0; SET(); x = v; DIAG(); }

void mad_mat_copy (const num_t x[], num_t r[], ssz_t m, ssz_t n, ssz_t ldx, ssz_t ldr)
{ CHKXRX; CPY(); }

void mad_mat_copym (const num_t x[], cpx_t r[], ssz_t m, ssz_t n, ssz_t ldx, ssz_t ldr)
{ CHKXR; CPY(); }

void mad_mat_trans (const num_t x[], num_t r[], ssz_t m, ssz_t n)
{ CHKXR; TRANS(num_t,); }

void mad_mat_mul (const num_t x[], const num_t y[], num_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { MUL(); return; }
  mad_alloc_tmp(num_t, r_, m*n);
  num_t *t = r; r = r_;
  MUL();
  mad_vec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_mulm (const num_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (y != r) { MUL(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MUL();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_tmul (const num_t x[], const num_t y[], num_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { TMUL(); return; }
  mad_alloc_tmp(num_t, r_, m*n);
  num_t *t = r; r = r_;
  TMUL();
  mad_vec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_tmulm (const num_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (y != r) { TMUL(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  TMUL();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_mult (const num_t x[], const num_t y[], num_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { MULT(); return; }
  mad_alloc_tmp(num_t, r_, m*n);
  num_t *t = r; r = r_;
  MULT();
  mad_vec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_multm (const num_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (y != r) { MULT(conj); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MULT(conj);
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_dmul (const num_t x[], const num_t y[], num_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { DMUL(); return; }
  mad_alloc_tmp(num_t, r_, m*n);
  num_t *t = r; r = r_;
  DMUL();
  mad_vec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_dmulm (const num_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (y != r) { DMUL(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  DMUL();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_muld (const num_t x[], const num_t y[], num_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { MULD(); return; }
  mad_alloc_tmp(num_t, r_, m*n);
  num_t *t = r; r = r_;
  MULD();
  mad_vec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_muldm (const num_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (y != r) { MULD(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MULD();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_mat_center (num_t x[], ssz_t m, ssz_t n, int d)
{ CHKX; num_t mu;
  switch(d) { // 0=vec, 1=row, 2=col, 3=diag
  case 0:
    mu = 0;
    FOR(k,m*n) mu += x[k];
    mu /= m*n;
    FOR(k,m*n) x[k] -= mu;
    break;
  case 1:
    FOR(i,m) { mu = 0;
      FOR(j,n) mu += x[i*n+j];
      mu /= n;
      FOR(j,n) x[i*n+j] -= mu;
    } break;
  case 2:
    FOR(j,n) { mu = 0;
      FOR(i,m) mu += x[i*n+j];
      mu /= m;
      FOR(i,m) x[i*n+j] -= mu;
    } break;
  case 3:
    mu = 0;
    FOR(k,MIN(m,n)) mu += x[k*n+k];
    mu /= MIN(m,n);
    FOR(k,MIN(m,n)) x[k*n+k] -= mu;
    break;
  default: error("invalid direction");
  }
}

void mad_mat_rev (num_t x[], ssz_t m, ssz_t n, int d)
{ CHKX; num_t t;
  switch(d) { // 0=vec, 1=row, 2=col, 3=diag
  case 0: FOR(i,(m*n)/2)        SWAP(x[i    ], x[         m*n-1-i]    , t); break;
  case 1: FOR(i,m  ) FOR(j,n/2) SWAP(x[i*n+j], x[     i *n +n-1-j]    , t); break;
  case 2: FOR(i,m/2) FOR(j,n  ) SWAP(x[i*n+j], x[(m-1-i)*n +    j]    , t); break;
  case 3: FOR(i,MIN(m,n)/2)     SWAP(x[i*n+i], x[(MIN(m,n)-1-i)*(n+1)], t); break;
  default: error("invalid direction");
  }
}

void
mad_mat_roll (num_t x[], ssz_t m, ssz_t n, int mroll, int nroll)
{ CHKX; mroll %= m; nroll %= n;
  ssz_t nm = n*m, msz = n*abs(mroll), nsz = abs(nroll);
  ssz_t sz = msz > nsz ? msz : nsz;
  mad_alloc_tmp(num_t, a, sz);
  if (mroll > 0) {
    mad_vec_copy(x+nm-msz, a    ,    msz); // end of x to a
    mad_vec_copy(x       , x+msz, nm-msz); // shift x down
    mad_vec_copy(a       , x    ,    msz); // a to beginning of x
  } else
  if (mroll < 0) {
    mad_vec_copy(x    , a       ,    msz); // beginning of x to a
    mad_vec_copy(x+msz, x       , nm-msz); // shift x up
    mad_vec_copy(a    , x+nm-msz,    msz); // a to end of x
  }
  if (nroll > 0) FOR(i,0,nm,n) {
    mad_vec_copy(x+i+n-nsz, a      ,   nsz); // end of x to a
    mad_vec_copy(x+i      , x+i+nsz, n-nsz); // shift x right
    mad_vec_copy(a        , x+i    ,   nsz); // a to beginning of x
  } else
  if (nroll < 0) FOR(i,0,nm,n) {
    mad_vec_copy(x+i    , a        ,   nsz); // beginning of x to a
    mad_vec_copy(x+i+nsz, x+i      , n-nsz); // shift x left
    mad_vec_copy(a      , x+i+n-nsz,   nsz); // a to end of x
  }
  mad_free_tmp(a);
}

// -- cmat

void mad_cmat_eye (cpx_t r[], cpx_t v, ssz_t m, ssz_t n, ssz_t ldr)
{ CHKR; cpx_t x = 0; SET(); x = v; DIAG(); }

void mad_cmat_eye_r (cpx_t r[], num_t v_re, num_t v_im, ssz_t m, ssz_t n, ssz_t ldr)
{ CHKR; mad_cmat_eye(r, CPX(v_re,v_im), m, n, ldr); }

void mad_cmat_copy (const cpx_t x[], cpx_t r[], ssz_t m, ssz_t n, ssz_t ldx, ssz_t ldr)
{ CHKXRX; CPY(); }

void mad_cmat_trans (const cpx_t x[], cpx_t r[], ssz_t m, ssz_t n)
{ CHKXR; TRANS(cpx_t,); }

void mad_cmat_ctrans (const cpx_t x[], cpx_t r[], ssz_t m, ssz_t n)
{ CHKXR; TRANS(cpx_t,conj); }

void mad_cmat_mul (const cpx_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { MUL(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MUL();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_mulm (const cpx_t x[], const num_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r) { MUL(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MUL();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_tmul (const cpx_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { TMUL(conj); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  TMUL(conj);
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_tmulm (const cpx_t x[], const num_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r) { TMUL(conj); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  TMUL(conj);
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_mult (const cpx_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { MULT(conj); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MULT(conj);
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_multm (const cpx_t x[], const num_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r) { MULT(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MULT();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_dmul (const cpx_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { DMUL(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  DMUL();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_dmulm (const cpx_t x[], const num_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r) { DMUL(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  DMUL();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_muld (const cpx_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r && y != r) { MULD(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MULD();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_muldm (const cpx_t x[], const num_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p)
{ CHKXYR;
  if (x != r) { MULD(); return; }
  mad_alloc_tmp(cpx_t, r_, m*n);
  cpx_t *t = r; r = r_;
  MULD();
  mad_cvec_copy(r_, t, m*n);
  mad_free_tmp(r_);
}

void mad_cmat_center (cpx_t x[], ssz_t m, ssz_t n, int d)
{ CHKX; cpx_t mu;
  switch(d) { // 0=vec, 1=row, 2=col, 3=diag
  case 0:
    mu = 0;
    FOR(k,m*n) mu += x[k];
    mu /= m*n;
    FOR(k,m*n) x[k] -= mu;
    break;
  case 1:
    FOR(i,m) { mu = 0;
      FOR(j,n) mu += x[i*n+j];
      mu /= n;
      FOR(j,n) x[i*n+j] -= mu;
    } break;
  case 2:
    FOR(j,n) { mu = 0;
      FOR(i,m) mu += x[i*n+j];
      mu /= m;
      FOR(i,m) x[i*n+j] -= mu;
    } break;
  case 3:
    mu = 0;
    FOR(k,MIN(m,n)) mu += x[k*n+k];
    mu /= MIN(m,n);
    FOR(k,MIN(m,n)) x[k*n+k] -= mu;
    break;
  default: error("invalid direction");
  }
}

void mad_cmat_rev (cpx_t x[], ssz_t m, ssz_t n, int d)
{ CHKX; cpx_t t;
  switch(d) { // 0=vec, 1=row, 2=col, 3=diag
  case 0: FOR(i,(m*n)/2)        SWAP(x[i     ], x[         m*n-1-i], t); break;
  case 1: FOR(i,m  ) FOR(j,n/2) SWAP(x[i*n +j], x[     i *n +n-1-j], t); break;
  case 2: FOR(i,m/2) FOR(j,n  ) SWAP(x[i*n +j], x[(m-1-i)*n +    j], t); break;
  case 3: FOR(i,MIN(m,n)/2)     SWAP(x[i*n +i], x[(m-1-i)*n +    i], t); break;
  default: error("invalid direction");
  }
}

void mad_cmat_roll (cpx_t x[], ssz_t m, ssz_t n, int mroll, int nroll)
{ mad_mat_roll((num_t*)x, m, 2*n, mroll, 2*nroll); }

// --- imat

void mad_imat_eye (idx_t r[], idx_t v, ssz_t m, ssz_t n, ssz_t ldr)
{ CHKR; idx_t x = 0; SET(); x = v; DIAG(); }

void mad_imat_copy (const idx_t x[], idx_t r[], ssz_t m, ssz_t n, ssz_t ldx, ssz_t ldr)
{ CHKXRX; CPY(); }

void mad_imat_copym (const idx_t x[], num_t r[], ssz_t m, ssz_t n, ssz_t ldx, ssz_t ldr)
{ CHKXR; CPY(); }

void mad_imat_trans (const idx_t x[], idx_t r[], ssz_t m, ssz_t n)
{ CHKXR; TRANS(idx_t,); }

void mad_imat_rev (idx_t x[], ssz_t m, ssz_t n, int d)
{ CHKX; idx_t t;
  switch(d) { // 0=vec, 1=row, 2=col, 3=diag
  case 0: FOR(i,(m*n)/2)        SWAP(x[i     ], x[         m*n-1-i], t); break;
  case 1: FOR(i,m  ) FOR(j,n/2) SWAP(x[i*n +j], x[     i *n +n-1-j], t); break;
  case 2: FOR(i,m/2) FOR(j,n  ) SWAP(x[i*n +j], x[(m-1-i)*n +    j], t); break;
  case 3: FOR(i,MIN(m,n)/2)     SWAP(x[i*n +i], x[(m-1-i)*n +    i], t); break;
  default: error("invalid direction");
  }
}

void
mad_imat_roll (idx_t x[], ssz_t m, ssz_t n, int mroll, int nroll)
{ CHKX; mroll %= m; nroll %= n;
  ssz_t nm = n*m, msz = n*abs(mroll), nsz = abs(nroll);
  ssz_t sz = msz > nsz ? msz : nsz;
  mad_alloc_tmp(idx_t, a, sz);
  if (mroll > 0) { // roll rows
    mad_ivec_copy(x+nm-msz, a    ,    msz); // end of x to a
    mad_ivec_copy(x       , x+msz, nm-msz); // shift x down
    mad_ivec_copy(a       , x    ,    msz); // a to beginning of x
  } else
  if (mroll < 0) {
    mad_ivec_copy(x    , a       ,    msz); // beginning of x to a
    mad_ivec_copy(x+msz, x       , nm-msz); // shift x up
    mad_ivec_copy(a    , x+nm-msz,    msz); // a to end of x
  }
  if (nroll > 0) FOR(i,0,nm,n) {
    mad_ivec_copy(x+i+n-nsz, a      ,   nsz); // end of x to a
    mad_ivec_copy(x+i      , x+i+nsz, n-nsz); // shift x right
    mad_ivec_copy(a        , x+i    ,   nsz); // a to beginning of x
  } else
  if (nroll < 0) FOR(i,0,nm,n) {
    mad_ivec_copy(x+i    , a        ,   nsz); // beginning of x to a
    mad_ivec_copy(x+i+nsz, x+i      , n-nsz); // shift x left
    mad_ivec_copy(a      , x+i+n-nsz,   nsz); // a to end of x
  }
  mad_free_tmp(a);
}

// -- Symplectic matrices -----------------------------------------------------o

// M[2n x 2n] accessed as n blocks of [a b ; c d]

#define a_(x,i,j) x[ i   *n+j  ]
#define b_(x,i,j) x[ i   *n+j+1]
#define c_(x,i,j) x[(i+1)*n+j  ]
#define d_(x,i,j) x[(i+1)*n+j+1]

// -- Symplecticity error, compute M' J M - J ---------------------------------o

num_t mad_mat_symperr (const num_t x[], num_t r_[], ssz_t n, num_t *tol_)
{ CHKX; assert(!(n & 1));
  num_t s=0, s0, s1, s2, s3;
  ssz_t nn = n*n;
  mad_alloc_tmp(num_t, r, nn);
  for (idx_t i = 0; i < n-1; i += 2) {
    // i == j
    s1 = -1, s2 = 1;
    for (idx_t k = 0; k < n-1; k += 2) {
      s1 += a_(x,k,i) * d_(x,k,i) - b_(x,k,i) * c_(x,k,i);
      s2 += b_(x,k,i) * c_(x,k,i) - a_(x,k,i) * d_(x,k,i);
    }
    s += s1*s1 + s2*s2;
    b_(r,i,i) = s1, c_(r,i,i) = s2, a_(r,i,i) = d_(r,i,i) = 0;
    // i < j
    for (idx_t j = i+2; j < n-1; j += 2) {
      s0 = s1 = s2 = s3 = 0;
      for (idx_t k = 0; k < n-1; k += 2) {
        s0 += a_(x,k,i) * c_(x,k,j) - a_(x,k,j) * c_(x,k,i);
        s1 += a_(x,k,i) * d_(x,k,j) - b_(x,k,j) * c_(x,k,i);
        s2 += b_(x,k,i) * c_(x,k,j) - a_(x,k,j) * d_(x,k,i);
        s3 += b_(x,k,i) * d_(x,k,j) - b_(x,k,j) * d_(x,k,i);
      }
      s += 2*(s0*s0 + s1*s1 + s2*s2 + s3*s3);
      a_(r,i,j) =  s0, b_(r,i,j) =  s1, c_(r,i,j) =  s2, d_(r,i,j) =  s3;
      a_(r,j,i) = -s0, b_(r,j,i) = -s2, c_(r,j,i) = -s1, d_(r,j,i) = -s3;
    }
  }
  if (tol_) {
    num_t tol = MAX(0,*tol_) ; *tol_ = 1; // is_symp = true
    for (idx_t i = 0; i < nn; i++) if (fabs(r[i]) > tol) {*tol_ = 0; break;}
  }
  if (r_) mad_vec_copy(r, r_, nn);
  mad_free_tmp(r);
  return sqrt(s);
}

num_t mad_cmat_symperr (const cpx_t x[], cpx_t r_[], ssz_t n, num_t *tol_)
{ CHKX; assert(!(n & 1));
  cpx_t s=0, s0, s1, s2, s3;
  ssz_t nn = n*n;
  mad_alloc_tmp(cpx_t, r, nn);
  for (idx_t i = 0; i < n-1; i += 2) {
    // i == j
    s1 = -1, s2 = 1;
    for (idx_t k = 0; k < n-1; k += 2) {
      s1 += conj(a_(x,k,i)) * d_(x,k,i) - b_(x,k,i) * conj(c_(x,k,i));
      s2 += conj(b_(x,k,i)) * c_(x,k,i) - a_(x,k,i) * conj(d_(x,k,i));
    }
    s += s1*s1 + s2*s2;
    b_(r,i,i) = s1, c_(r,i,i) = s2, a_(r,i,i) = d_(r,i,i) = 0;
    // i < j
    for (idx_t j = i+2; j < n-1; j += 2) {
      s0 = s1 = s2 = s3 = 0;
      for (idx_t k = 0; k < n-1; k += 2) {
        s0 += conj(a_(x,k,i)) * c_(x,k,j) - a_(x,k,j) * conj(c_(x,k,i));
        s1 += conj(a_(x,k,i)) * d_(x,k,j) - b_(x,k,j) * conj(c_(x,k,i));
        s2 += conj(b_(x,k,i)) * c_(x,k,j) - a_(x,k,j) * conj(d_(x,k,i));
        s3 += conj(b_(x,k,i)) * d_(x,k,j) - b_(x,k,j) * conj(d_(x,k,i));
      }
      s += 2*(s0*s0 + s1*s1 + s2*s2 + s3*s3);
      a_(r,i,j) =  s0, b_(r,i,j) =  s1, c_(r,i,j) =  s2, d_(r,i,j) =  s3;
      a_(r,j,i) = -s0, b_(r,j,i) = -s2, c_(r,j,i) = -s1, d_(r,j,i) = -s3;
    }
  }
  if (tol_) {
    num_t tol = MAX(0,*tol_) ; *tol_ = 1; // is_symp = true
    for (idx_t i = 0; i < nn; i++) if (cabs(r[i]) > tol) {*tol_ = 0; break;}
  }
  if (r_) mad_cvec_copy(r, r_, nn);
  mad_free_tmp(r);
  return sqrt(cabs(s));
}

// -- Symplectic conjugate, compute \bar{M} = -J M' J -------------------------o

void mad_mat_sympconj (const num_t x[], num_t r[], ssz_t n)
{ CHKXR; assert(!(n & 1));
  num_t t;
  for (idx_t i = 0; i < n-1; i += 2) {     // 2x2 blocks on diagonal
    t = a_(x,i,i),  a_(r,i,i) =  d_(x,i,i),  d_(r,i,i) = t;
    b_(r,i,i) = -b_(x,i,i),  c_(r,i,i) = -c_(x,i,i);

    for (idx_t j = i+2; j < n-1; j += 2) { // 2x2 blocks off diagonal
      t = a_(x,i,j),  a_(r,i,j) =  d_(x,j,i),  d_(r,j,i) =  t;
      t = b_(x,i,j),  b_(r,i,j) = -b_(x,j,i),  b_(r,j,i) = -t;
      t = c_(x,i,j),  c_(r,i,j) = -c_(x,j,i),  c_(r,j,i) = -t;
      t = d_(x,i,j),  d_(r,i,j) =  a_(x,j,i),  a_(r,j,i) =  t;
    }
  }
}

void mad_cmat_sympconj (const cpx_t x[], cpx_t r[], ssz_t n)
{ CHKXR; assert(!(n & 1));
  cpx_t t;
  for (idx_t i = 0; i < n-1; i += 2) {     // 2x2 blocks on diagonal
    t = a_(x,i,i),  a_(r,i,i) =  conj(d_(x,i,i)),  d_(r,i,i) = conj(t);
    b_(r,i,i) = -conj(b_(x,i,i)),  c_(r,i,i) = -conj(c_(x,i,i));

    for (idx_t j = i+2; j < n-1; j += 2) {   // 2x2 blocks off diagonal
      t = a_(x,i,j),  a_(r,i,j) =  conj(d_(x,j,i)),  d_(r,j,i) =  conj(t);
      t = b_(x,i,j),  b_(r,i,j) = -conj(b_(x,j,i)),  b_(r,j,i) = -conj(t);
      t = c_(x,i,j),  c_(r,i,j) = -conj(c_(x,j,i)),  c_(r,j,i) = -conj(t);
      t = d_(x,i,j),  d_(r,i,j) =  conj(a_(x,j,i)),  a_(r,j,i) =  conj(t);
    }
  }
}

#undef a_
#undef b_
#undef c_
#undef d_

// -- lapack ------------------------------------------------------------------o

/*
LAPACK is the default method for computing LU decomposition. When matrix is
square and nonsinguler the routines dgetrf and zgetrf.

LAPACK is the default method for solving linear least squares problems for dense
matrices. When the matrix is square and nonsingular the routines dgesv and zgesv
are used otherwise routines dgelsy and zgelsy are used. Alternate methods dgelsd
and zgelsd based on SVD can be used too. The general least squares problems for
dense matrices uses dgglse and zgglse for the first kind, and dggglm and zggglm
for the second kind.

LAPACK is the default method for computing the entire set of singluar values
and singular vectors. For generalized SVD the routines dgesdd and zgesdd are
used.

LAPACK is the default method for computing the entire set of eigenvalues and
eigenvectors. For simple eigenvalues the routines dgeev and zgeev are used. For
generalized eigenvalues the routines dggev and zggev are used.

LAPACK C examples using F77 interface can be found at:
https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/
LAPACK F77 examples with data can be found at:
https://github.com/numericalalgorithmsgroup/LAPACK_Examples/tree/master/examples
*/

// -----
// Decompose A = LU with A[m x n] (generalized)
// -----
void dgetrf_ (const int *m, const int *n, num_t A[], const int *lda,
              int *IPIV, int *info);
void zgetrf_ (const int *m, const int *n, cpx_t A[], const int *lda,
              int *IPIV, int *info);

// -----
// Solve A * X = B with A[n x n], B[n x nrhs] and X[n x nrhs]: search min |} b - Ax ||_2 using LU
// -----
void dgesv_ (const int *n, const int *nrhs, num_t A[], const int *lda,
                                 int *IPIV, num_t B[], const int *ldb, int *info);
void zgesv_ (const int *n, const int *nrhs, cpx_t A[], const int *lda,
                                 int *IPIV, cpx_t B[], const int *ldb, int *info);

// -----
// Solve A * X = B with A[m x n], B[m x nrhs] and X[m x nrhs]: search min || b - Ax ||_2 using QR
// -----
void dgelsy_ (const int *m, const int *n, const int *nrhs,
              num_t A[], const int *lda, num_t B[], const int *ldb,
              int jpvt[], const num_t *rcond, int *rank,
              num_t work[], const int lwork[], int *info);
void zgelsy_ (const int *m, const int *n, const int *nrhs,
              cpx_t A[], const int *lda, cpx_t B[], const int *ldb,
              int jpvt[], const num_t *rcond, int *rank,
              cpx_t work[], const int lwork[], num_t rwork[], int *info);

// -----
// Solve A * X = B with A[m x n], B[m x nrhs] and X[m x nrhs]: search min || b - Ax ||_2 using SVD
// -----
void dgelsd_ (const int *m, const int *n, const int *nrhs,
              num_t A[], const int *lda, num_t B[], const int *ldb,
              num_t S[], const num_t *rcond, int *rank,
              num_t work[], int *lwork, int iwork[], int *info);
void zgelsd_ (const int *m, const int *n, const int *nrhs,
              cpx_t A[], const int *lda, cpx_t B[], const int *ldb,
               num_t S[], const num_t *rcond, int *rank,
              cpx_t work[], int *lwork, num_t rwork[], int iwork[], int *info);

// -----
// LS minimization: min_x || c - A*x ||_2 subject to B*x = d using QR
// -----

void dgglse_ (const int *m, const int *n, const int *p,
              num_t A[], const int *lda, num_t B[], const int *ldb,
              num_t C[], num_t D[], num_t X[],
              num_t work[], int *lwork, int *info);
void zgglse_ (const int *m, const int *n, const int *p,
              cpx_t A[], const int *lda, cpx_t B[], const int *ldb,
              cpx_t C[], cpx_t D[], cpx_t X[],
              cpx_t work[], int *lwork, int *info);

// -----
// LS minimization: min_x || y ||_2 subject to A*x + B*y = d using QR
// -----

void dggglm_ (const int *m, const int *n, const int *p,
              num_t A[], const int *lda, num_t B[], const int *ldb,
              num_t D[], num_t X[], num_t Y[],
              num_t work[], int *lwork, int *info);
void zggglm_ (const int *m, const int *n, const int *p,
              cpx_t A[], const int *lda, cpx_t B[], const int *ldb,
              cpx_t D[], cpx_t X[], cpx_t Y[],
              cpx_t work[], int *lwork, int *info);

// -----
// SVD A[m x n]
// -----
void dgesdd_ (str_t jobz, const int *m, const int *n, num_t A[], const int *lda,
              num_t S[], num_t U[], const int *ldu, num_t VT[], const int *ldvt,
              num_t work[], int *lwork, int iwork[], int *info);
void zgesdd_ (str_t jobz, const int *m, const int *n, cpx_t A[], const int *lda,
              num_t S[], cpx_t U[], const int *ldu, cpx_t VT[], const int *ldvt,
              cpx_t work[], int *lwork, num_t rwork[], int iwork[], int *info);

// -----
// Eigen values/vectors A[n x n]
// -----
void dgeev_ (str_t jobvl, str_t jobvr, const int *n, num_t A[], const int *lda,
             num_t WR[], num_t WI[],
             num_t VL[], const int *ldvl, num_t VR[], const int *ldvr,
             num_t work[], int *lwork, int *info);
void zgeev_ (str_t jobvl, str_t jobvr, const int *n, cpx_t A[], const int *lda,
             cpx_t W[], cpx_t VL[], const int *ldvl, cpx_t VR[], const int *ldvr,
             cpx_t work[], int *lwork, num_t rwork[], int *info);

// -- determinant -------------------------------------------------------------o

int
mad_mat_det (const num_t x[], num_t *r, ssz_t n)
{
  CHKX;
  const int nn=n;
  int info=0, ipiv[n];
  mad_alloc_tmp(num_t, a, n*n);
  mad_vec_copy(x, a, n*n);
  dgetrf_(&nn, &nn, a, &nn, ipiv, &info);

  if (info < 0) error("Det: invalid input argument");

  int perm = 0;
  num_t det = 1;
  for (int i=0, j=0; i < n; i++, j+=n+1)
    det *= a[j], perm += ipiv[i] != i+1;
  mad_free_tmp(a);
  *r = perm & 1 ? -det : det;
  return info;
}

int
mad_cmat_det (const cpx_t x[], cpx_t *r, ssz_t n)
{
  CHKX;
  const int nn=n;
  int info=0, ipiv[n];
  mad_alloc_tmp(cpx_t, a, n*n);
  mad_cvec_copy(x, a, n*n);
  zgetrf_(&nn, &nn, a, &nn, ipiv, &info);

  if (info < 0) error("Det: invalid input argument");

  int perm = 0;
  cpx_t det = 1;
  for (int i=0, j=0; i < n; i++, j+=n+1)
    det *= a[j], perm += ipiv[i] != i+1;
  mad_free_tmp(a);
  *r = perm & 1 ? -det : det;
  return info;
}

// -- inverse -----------------------------------------------------------------o

int
mad_mat_invn (const num_t y[], num_t x, num_t r[], ssz_t m, ssz_t n, num_t rcond)
{
  CHKYR; // compute U:[n x n]/Y:[m x n]
  mad_alloc_tmp(num_t, u, n*n);
  mad_mat_eye(u, 1, n, n, n);
#pragma GCC diagnostic push // remove false-positive
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
  int rank = mad_mat_div(u, y, r, n, m, n, rcond);
#pragma GCC diagnostic pop
  mad_free_tmp(u);
  if (x != 1) mad_vec_muln(r, x, r, m*n);
  return rank;
}

int // without complex-by-value version
mad_mat_invc_r (const num_t y[], num_t x_re, num_t x_im, cpx_t r[], ssz_t m, ssz_t n, num_t rcond)
{ return mad_mat_invc(y, CPX(x_re,x_im), r, m, n, rcond); }

int
mad_mat_invc (const num_t y[], cpx_t x, cpx_t r[], ssz_t m, ssz_t n, num_t rcond)
{
  CHKYR; // compute U:[n x n]/Y:[m x n]
  mad_alloc_tmp(num_t, t, m*n);
  mad_alloc_tmp(num_t, u, n*n);
  mad_mat_eye(u, 1, n, n, n);
  int rank = mad_mat_div(u, y, t, n, m, n, rcond);
  mad_free_tmp(u);
  if (x != 1) mad_vec_mulc(t, x, r, m*n);
  mad_free_tmp(t);
  return rank;
}

int
mad_cmat_invn (const cpx_t y[], num_t x, cpx_t r[], ssz_t m, ssz_t n, num_t rcond)
{
  CHKYR; // compute U:[n x n]/Y:[m x n]
  mad_alloc_tmp(cpx_t, u, n*n);
  mad_cmat_eye(u, 1, n, n, n);
#pragma GCC diagnostic push // remove false-positive
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
  int rank = mad_cmat_div(u, y, r, n, m, n, rcond);
#pragma GCC diagnostic pop
  mad_free_tmp(u);
  if (x != 1) mad_cvec_muln(r, x, r, m*n);
  return rank;
}

int
mad_cmat_invc (const cpx_t y[], cpx_t x, cpx_t r[], ssz_t m, ssz_t n, num_t rcond)
{
  CHKYR; // compute U:[n x n]/Y:[m x n]
  mad_alloc_tmp(cpx_t, u, n*n);
  mad_cmat_eye(u, 1, n, n, n);
#pragma GCC diagnostic push // remove false-positive
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
  int rank = mad_cmat_div(u, y, r, n, m, n, rcond);
#pragma GCC diagnostic pop
  mad_free_tmp(u);
  if (x != 1) mad_cvec_mulc(r, x, r, m*n);
  return rank;
}

int
mad_cmat_invc_r (const cpx_t y[], num_t x_re, num_t x_im, cpx_t r[], ssz_t m, ssz_t n, num_t rcond)
{ return mad_cmat_invc(y, CPX(x_re,x_im), r, m, n, rcond); }

// -- pseudo-inverse ----------------------------------------------------------o

// SVD decomposition A = U * S * V.t()
// A:[m x n], U:[m x m], S:[min(m,n)], V:[n x n]

int
mad_mat_pinvn (const num_t y[], num_t x, num_t r[], ssz_t m, ssz_t n, num_t rcond, int ncond)
{
  CHKYR; // compute x/Y[m x n]
  ssz_t mn = MIN(m,n);
  mad_alloc_tmp(num_t, s, mn );
  mad_alloc_tmp(num_t, U, m*m);
  mad_alloc_tmp(num_t, V, n*n);
  mad_alloc_tmp(num_t, S, n*m); mad_vec_fill(0, S, n*m);

  int rank = 0;
  int info = mad_mat_svd(y, U, s, V, m, n);
  if (info != 0) goto finalize;

  // Remove ncond (largest) singular values
  idx_t k = 0;
  FOR(i,MIN(mn,-ncond)) s[k++] = 0;

  // Tolerance on keeping singular values.
  rcond = MAX(fabs(rcond), DBL_EPSILON);

  // Keep relevant singular values and reject ncond (smallest) singular values
  FOR(i,k,mn)
    if (mn-i >= ncond && s[i] >= rcond*s[k]) S[i*m+i] = (++rank, 1/s[i]);
    else break;

  mad_mat_muld(V, S, r, n, m, n);
  mad_mat_mult(r, U, r, n, m, m);

  if (x != 1) mad_vec_muln(r, x, r, m*n);

finalize:
  mad_free_tmp(s);
  mad_free_tmp(U);
  mad_free_tmp(V);
  mad_free_tmp(S);

  return rank;
}

int // without complex-by-value version
mad_mat_pinvc_r (const num_t y[], num_t x_re, num_t x_im, cpx_t r[], ssz_t m, ssz_t n, num_t rcond, int ncond)
{ return mad_mat_pinvc(y, CPX(x_re,x_im), r, m, n, rcond, ncond); }

int
mad_mat_pinvc (const num_t y[], cpx_t x, cpx_t r[], ssz_t m, ssz_t n, num_t rcond, int ncond)
{
  CHKYR; // compute x/Y[m x n]
  mad_alloc_tmp(num_t, rr, m*n);
  int rank = mad_mat_pinvn(y, 1, rr, m, n, rcond, ncond);
  mad_vec_mulc(rr, x, r, m*n);
  mad_free_tmp(rr);
  return rank;
}

int
mad_cmat_pinvc_r (const cpx_t y[], num_t x_re, num_t x_im, cpx_t r[], ssz_t m, ssz_t n, num_t rcond, int ncond)
{ return mad_cmat_pinvc(y, CPX(x_re,x_im), r, m, n, rcond, ncond); }

int
mad_cmat_pinvn (const cpx_t y[], num_t x, cpx_t r[], ssz_t m, ssz_t n, num_t rcond, int ncond)
{ return mad_cmat_pinvc(y, CPX(x,0), r, m, n, rcond, ncond); }

int
mad_cmat_pinvc (const cpx_t y[], cpx_t x, cpx_t r[], ssz_t m, ssz_t n, num_t rcond, int ncond)
{
  CHKYR; // compute x/Y[m x n]
  ssz_t mn = MIN(m,n);
  mad_alloc_tmp(num_t, s, mn );
  mad_alloc_tmp(cpx_t, U, m*m);
  mad_alloc_tmp(cpx_t, V, n*n);
  mad_alloc_tmp(num_t, S, n*m); mad_vec_fill(0, S, n*m);

  int rank = 0;
  int info = mad_cmat_svd(y, U, s, V, m, n);
  if (info != 0) goto finalize;

  // Remove ncond (largest) singular values
  idx_t k = 0;
  FOR(i,MIN(mn,-ncond)) s[k++] = 0;

  // Tolerance on keeping singular values.
  rcond = MAX(fabs(rcond), DBL_EPSILON);

  // Keep relevant singular values and reject ncond (smallest) singular values
  FOR(i,k,mn)
    if (mn-i >= ncond && s[i] >= rcond*s[k]) S[i*m+i] = (++rank, 1/s[i]);
    else break;

  mad_cmat_muldm(V, S, r, n, m, n);
  mad_cmat_mult (r, U, r, n, m, m);

  if (x != 1) mad_cvec_mulc(r, x, r, m*n);

finalize:
  mad_free_tmp(s);
  mad_free_tmp(U);
  mad_free_tmp(V);
  mad_free_tmp(S);

  return rank;
}

// -- divide ------------------------------------------------------------------o

// note:
// X/Y => X * Y^-1 => [m x p] * [p x n] => X:[m x p], Y:[n x p]
// X/Y => X * Y^-1 => (Y'^-1 * X')' => A=Y' and B=X'
// Solving A*X=B => X = A^-1 B = (B'/A')' (col-major!)
//    with Y':[p x n] = A:[M=p x N=n],
//    and  X':[p x m] = B:[M=p x NRHS=m], ipiv:[N]

int
mad_mat_div (const num_t x[], const num_t y[], num_t r[], ssz_t m, ssz_t n, ssz_t p, num_t rcond)
{
  CHKXYR;
  int info=0;
  const int nm=m, nn=n, np=p;
  mad_alloc_tmp(num_t, a, n*p);
  mad_vec_copy(y, a, n*p);

  // square system (y is square, n == p), use LU decomposition
  if (n == p) {
    int ipiv[n];
    mad_vec_copy(x, r, m*p);
    dgesv_(&np, &nm, a, &np, ipiv, r, &np, &info);
    if (!info) return mad_free_tmp(a), n;
    if (info > 0) warn("Div: singular matrix, no solution found");
  }

  // non-square system or singular square system, use QR or LQ factorization
  num_t sz;
  int rank, ldb=MAX(nn,np), lwork=-1; // query for optimal size
  int JPVT[nn]; memset(JPVT, 0, sizeof JPVT);
  mad_alloc_tmp(num_t, rr, ldb*nm);
  mad_mat_copy(x, rr, m, p, p, ldb); // input strided copy [M x NRHS]
  dgelsy_(&np, &nn, &nm, a, &np, rr, &ldb, JPVT, &rcond, &rank, &sz, &lwork, &info); // query
  lwork=sz;
  mad_alloc_tmp(num_t, wk, lwork);
  dgelsy_(&np, &nn, &nm, a, &np, rr, &ldb, JPVT, &rcond, &rank,  wk, &lwork, &info); // compute
  mad_mat_copy(rr, r, m, n, ldb, n); // output strided copy [N x NRHS]
  mad_free_tmp(wk); mad_free_tmp(rr); mad_free_tmp(a);

  if (info < 0) error("Div: invalid input argument");
  if (info > 0) error("Div: unexpected lapack error");

  return rank;
}

int
mad_mat_divm (const num_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p, num_t rcond)
{
  CHKXYR;
  int info=0;
  const int nm=m, nn=n, np=p;
  mad_alloc_tmp(cpx_t, a, n*p);
  mad_cvec_copy(y, a, n*p);

  // square system (y is square, n == p), use LU decomposition
  if (n == p) {
    int ipiv[n];
    mad_vec_copyv(x, r, m*p);
    zgesv_(&np, &nm, a, &np, ipiv, r, &np, &info);
    if (!info) return mad_free_tmp(a), n;
    if (info > 0) warn("Div: singular matrix, no solution found");
  }

  // non-square system or singular square system, use QR or LQ factorization
  cpx_t sz;
  num_t rwk[2*nn];
  int rank, ldb=MAX(nn,np), lwork=-1; // query for optimal size
  int JPVT[nn]; memset(JPVT, 0, sizeof JPVT);
  mad_alloc_tmp(cpx_t, rr, ldb*nm);
  mad_mat_copym(x, rr, m, p, p, ldb); // input strided copy [M x NRHS]
  zgelsy_(&np, &nn, &nm, a, &np, rr, &ldb, JPVT, &rcond, &rank, &sz, &lwork, rwk, &info); // query
  lwork=creal(sz);
  mad_alloc_tmp(cpx_t, wk, lwork);
  zgelsy_(&np, &nn, &nm, a, &np, rr, &ldb, JPVT, &rcond, &rank,  wk, &lwork, rwk, &info); // compute
  mad_cmat_copy(rr, r, m, n, ldb, n); // output strided copy [N x NRHS]
  mad_free_tmp(wk); mad_free_tmp(rr); mad_free_tmp(a);

  if (info < 0) error("Div: invalid input argument");
  if (info > 0) error("Div: unexpected lapack error");

  return rank;
}

int
mad_cmat_div (const cpx_t x[], const cpx_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p, num_t rcond)
{
  CHKXYR;
  int info=0;
  const int nm=m, nn=n, np=p;
  mad_alloc_tmp(cpx_t, a, n*p);
  mad_cvec_copy(y, a, n*p);

  // square system (y is square, n == p), use LU decomposition
  if (n == p) {
    int ipiv[n];
    mad_cvec_copy(x, r, m*p);
    zgesv_(&np, &nm, a, &np, ipiv, r, &np, &info);
    if (!info) return mad_free_tmp(a), n;
    if (info > 0) warn("Div: singular matrix, no solution found");
  }

  // non-square system or singular square system, use QR or LQ factorization
  cpx_t sz;
  num_t rwk[2*nn];
  int rank, ldb=MAX(nn,np), lwork=-1; // query for optimal size
  int JPVT[nn]; memset(JPVT, 0, sizeof JPVT);
  mad_alloc_tmp(cpx_t, rr, ldb*nm);
  mad_cmat_copy(x, rr, m, p, p, ldb); // input strided copy [M x NRHS]
  zgelsy_(&np, &nn, &nm, a, &np, rr, &ldb, JPVT, &rcond, &rank, &sz, &lwork, rwk, &info); // query
  lwork=creal(sz);
  mad_alloc_tmp(cpx_t, wk, lwork);
  zgelsy_(&np, &nn, &nm, a, &np, rr, &ldb, JPVT, &rcond, &rank,  wk, &lwork, rwk, &info); // compute
  mad_cmat_copy(rr, r, m, n, ldb, n); // output strided copy [N x NRHS]
  mad_free_tmp(wk); mad_free_tmp(rr); mad_free_tmp(a);

  if (info < 0) error("Div: invalid input argument");
  if (info > 0) error("Div: unexpected lapack error");

  return rank;
}

int
mad_cmat_divm (const cpx_t x[], const num_t y[], cpx_t r[], ssz_t m, ssz_t n, ssz_t p, num_t rcond)
{
  CHKXYR;
  int info=0;
  const int nm=m, nn=n, np=p;
  mad_alloc_tmp(cpx_t, a, n*p);
  mad_vec_copyv(y, a, n*p);

  // square system (y is square, n == p), use LU decomposition
  if (n == p) {
    int ipiv[n];
    mad_cvec_copy(x, r, m*p);
    zgesv_(&np, &nm, a, &np, ipiv, r, &np, &info);
    if (!info) return mad_free_tmp(a), n;
    if (info > 0) warn("Div: singular matrix, no solution found");
  }

  // non-square system or singular square system, use QR or LQ factorization
  cpx_t sz;
  num_t rwk[2*nn];
  int rank, ldb=MAX(nn,np), lwork=-1; // query for optimal size
  int JPVT[nn]; memset(JPVT, 0, sizeof JPVT);
  mad_alloc_tmp(cpx_t, rr, ldb*nm);
  mad_cmat_copy(x, rr, m, p, p, ldb); // input strided copy [M x NRHS]
  zgelsy_(&np, &nn, &nm, a, &np, rr, &ldb, JPVT, &rcond, &rank, &sz, &lwork, rwk, &info); // query
  lwork=creal(sz);
  mad_alloc_tmp(cpx_t, wk, lwork);
  zgelsy_(&np, &nn, &nm, a, &np, rr, &ldb, JPVT, &rcond, &rank,  wk, &lwork, rwk, &info); // compute
  mad_cmat_copy(rr, r, m, n, ldb, n); // output strided copy [N x NRHS]
  mad_free_tmp(wk); mad_free_tmp(rr); mad_free_tmp(a);

  if (info < 0) error("Div: invalid input argument");
  if (info > 0) error("Div: unexpected lapack error");

  return rank;
}

// -- SVD ---------------------------------------------------------------------o

// SVD decomposition A = U * S * V.t()
// A:[m x n], U:[m x m], S:[min(m,n)], V:[n x n]

int
mad_mat_svd (const num_t x[], num_t u[], num_t s[], num_t v[], ssz_t m, ssz_t n)
{
  assert( x && u && s && v );
  int info=0;
  const int nm=m, nn=n, mn=MIN(m,n);

  num_t sz;
  int lwork=-1;
  int iwk[8*mn];
  mad_alloc_tmp(num_t, ra, m*n);
  mad_mat_trans(x, ra, m, n);
  dgesdd_("A", &nm, &nn, ra, &nm, s, u, &nm, v, &nn, &sz, &lwork, iwk, &info); // query
  lwork=sz;
  mad_alloc_tmp(num_t, wk, lwork);
  dgesdd_("A", &nm, &nn, ra, &nm, s, u, &nm, v, &nn,  wk, &lwork, iwk, &info); // compute
  mad_free_tmp(wk); mad_free_tmp(ra);
  mad_mat_trans(u, u, m, m);

  if (info < 0) error("SVD: invalid input argument");
  if (info > 0) warn ("SVD: failed to converge");

  return info;
}

int
mad_cmat_svd (const cpx_t x[], cpx_t u[], num_t s[], cpx_t v[], ssz_t m, ssz_t n)
{
  assert( x && u && s && v );
  int info=0;
  const int nm=m, nn=n, mn=MIN(m,n);

  cpx_t sz;
  int lwork=-1;
  int iwk[8*mn];
  ssz_t rwk_sz = mn * MAX(5*mn+7, 2*MAX(m,n)+2*mn+1);
  mad_alloc_tmp(num_t, rwk, rwk_sz);
  mad_alloc_tmp(cpx_t, ra, m*n);
  mad_cmat_trans(x, ra, m, n);
  zgesdd_("A", &nm, &nn, ra, &nm, s, u, &nm, v, &nn, &sz, &lwork, rwk, iwk, &info); // query
  lwork=creal(sz);
  mad_alloc_tmp(cpx_t, wk, lwork);
  zgesdd_("A", &nm, &nn, ra, &nm, s, u, &nm, v, &nn,  wk, &lwork, rwk, iwk, &info); // compute
  mad_free_tmp(wk); mad_free_tmp(ra); mad_free_tmp(rwk);
  mad_cmat_trans(u, u, m, m);
  mad_cvec_conj (v, v, n*n);

  if (info < 0) error("SVD: invalid input argument");
  if (info > 0) warn ("SVD: failed to converge");

  return info;
}

// -- LEAST SQUARE Solvers ----------------------------------------------------o

int
mad_mat_solve (const num_t a[], const num_t b[], num_t x[], ssz_t m, ssz_t n, ssz_t p, num_t rcond)
{
  assert( a && b && x );
  int info=0;
  const int nm=m, nn=n, np=p, mn=MAX(m,n);

  num_t sz;
  int lwork=-1, rank;
  int pvt[nn]; memset(pvt, 0, sizeof pvt);
  mad_alloc_tmp(num_t, ta, m*n);
  mad_alloc_tmp(num_t, tb, mn*p); mad_vec_fill(0, tb+m*p, (mn-m)*p);
  mad_vec_copy (b , tb, m*p);
  mad_mat_trans(tb, tb, mn, p);
  mad_mat_trans(a , ta, m , n);
  dgelsy_(&nm, &nn, &np, ta, &nm, tb, &mn, pvt, &rcond, &rank, &sz, &lwork, &info); // query
  lwork=sz;
  mad_alloc_tmp(num_t, wk, lwork);
  dgelsy_(&nm, &nn, &np, ta, &nm, tb, &mn, pvt, &rcond, &rank,  wk, &lwork, &info); // compute
  mad_mat_trans(tb, tb, p, mn);
  mad_vec_copy (tb,  x, n*p);

  mad_free_tmp(wk); mad_free_tmp(ta); mad_free_tmp(tb);

  if (info < 0) error("Solve: invalid input argument");
  if (info > 0) warn ("Solve: unexpected lapack error");

  return rank;
}

int
mad_cmat_solve (const cpx_t a[], const cpx_t b[], cpx_t x[], ssz_t m, ssz_t n, ssz_t p, num_t rcond)
{
  assert( a && b && x );
  int info=0;
  const int nm=m, nn=n, np=p, mn=MAX(m,n);

  cpx_t sz;
  num_t rwk[2*nn];
  int lwork=-1, rank;
  int pvt[nn]; memset(pvt, 0, sizeof pvt);
  mad_alloc_tmp(cpx_t, ta, m*n);
  mad_alloc_tmp(cpx_t, tb, mn*p); mad_cvec_fill(0, tb+m*p, (mn-m)*p);
  mad_cvec_copy (b , tb, m*p);
  mad_cmat_trans(tb, tb, mn, p);
  mad_cmat_trans(a , ta, m , n);
  zgelsy_(&nm, &nn, &np, ta, &nm, tb, &mn, pvt, &rcond, &rank, &sz, &lwork, rwk, &info); // query
  lwork=creal(sz);
  mad_alloc_tmp(cpx_t, wk, lwork);
  zgelsy_(&nm, &nn, &np, ta, &nm, tb, &mn, pvt, &rcond, &rank,  wk, &lwork, rwk, &info); // compute
  mad_cmat_trans(tb, tb, p, mn);
  mad_cvec_copy (tb,  x, n*p);

  mad_free_tmp(wk); mad_free_tmp(ta); mad_free_tmp(tb);

  if (info < 0) error("Solve: invalid input argument");
  if (info > 0) warn ("Solve: unexpected lapack error");

  return rank;
}

int
mad_mat_ssolve (const num_t a[], const num_t b[], num_t x[], ssz_t m, ssz_t n, ssz_t p, num_t rcond, int ncond, num_t s_[])
{
  assert( a && b && x );
  int info=0;
  const int nm=m, nn=n, np=p, mn=MAX(m,n);

  if (ncond) {
    mad_alloc_tmp(num_t, ai, m*n);
    int rank = mad_mat_pinvn(a, 1, ai, m, n, rcond, ncond);
    mad_mat_mul(ai, b, x, n, p, m);
    mad_free_tmp(ai);
    return rank;
  }

  num_t sz;
  int lwork=-1, rank, isz;
  mad_alloc_tmp(num_t, ta, m *n);
  mad_alloc_tmp(num_t, tb, mn*p);
  mad_alloc_tmp(num_t, ts, MIN(m,n));
  mad_vec_copy (b , tb, m*p);
  mad_vec_fill (0 , tb +m*p, (mn-m)*p);
  mad_mat_trans(tb, tb, mn, p);
  mad_mat_trans(a , ta, m , n);
  dgelsd_(&nm, &nn, &np, ta, &nm, tb, &mn, ts, &rcond, &rank, &sz, &lwork, &isz, &info); // query
  lwork=sz;
  mad_alloc_tmp(num_t,  wk, lwork);
  mad_alloc_tmp(int  , iwk, isz);
  dgelsd_(&nm, &nn, &np, ta, &nm, tb, &mn, ts, &rcond, &rank,  wk, &lwork,  iwk, &info); // compute
  mad_mat_trans(tb, tb, p, mn);
  mad_vec_copy (tb,  x, n*p);

  if (s_) mad_vec_copy(ts, s_, MIN(m,n));

  mad_free_tmp(wk); mad_free_tmp(iwk);
  mad_free_tmp(ta); mad_free_tmp(tb); mad_free_tmp(ts);

  if (info < 0) error("SSolve: invalid input argument");
  if (info > 0) warn ("SSolve: failed to converge");

  return rank;
}

int
mad_cmat_ssolve (const cpx_t a[], const cpx_t b[], cpx_t x[], ssz_t m, ssz_t n, ssz_t p, num_t rcond, int ncond, num_t s_[])
{
  assert( a && b && x );
  int info=0;
  const int nm=m, nn=n, np=p, mn=MAX(m,n);

  if (ncond) {
    mad_alloc_tmp(cpx_t, ai, m*n);
    int rank = mad_cmat_pinvc(a, 1, ai, m, n, rcond, ncond);
    mad_cmat_mul(ai, b, x, n, p, m);
    mad_free_tmp(ai);
    return rank;
  }

  num_t rsz;
  cpx_t sz;
  int lwork=-1, rank, isz;
  mad_alloc_tmp(cpx_t, ta, m*n);
  mad_alloc_tmp(cpx_t, tb, mn*p);
  mad_alloc_tmp(num_t, ts, MIN(m,n));
  mad_cvec_copy (b , tb, m*p);
  mad_cvec_fill (0 , tb +m*p, (mn-m)*p);
  mad_cmat_trans(tb, tb, mn, p);
  mad_cmat_trans(a , ta, m , n);
  zgelsd_(&nm, &nn, &np, ta, &nm, tb, &mn, ts, &rcond, &rank, &sz, &lwork, &rsz, &isz, &info); // query
  lwork=creal(sz);
  mad_alloc_tmp(cpx_t,  wk, lwork);
  mad_alloc_tmp( num_t, rwk, (int)rsz);
  mad_alloc_tmp( int  , iwk, isz);
  zgelsd_(&nm, &nn, &np, ta, &nm, tb, &mn, ts, &rcond, &rank,  wk, &lwork,  rwk,  iwk, &info); // compute
  mad_cmat_trans(tb, tb, p, mn);
  mad_cvec_copy (tb,  x, n*p);

  if (s_) mad_vec_copy(ts, s_, MIN(m,n));

  mad_free_tmp(wk); mad_free_tmp(rwk); mad_free_tmp(iwk);
  mad_free_tmp(ta); mad_free_tmp(tb);  mad_free_tmp(ts);

  if (info < 0) error("SSolve: invalid input argument");
  if (info > 0) warn ("SSolve: failed to converge");

  return rank;
}

// -- Generalized LS Solvers --------------------------------------------------o

int
mad_mat_gsolve (const num_t a[], const num_t b[], const num_t c[], const num_t d[],
                num_t x[], ssz_t m, ssz_t n, ssz_t p, num_t *nrm_)
{
  assert( a && b && x );
  ensure( 0 <= p && p <= n && n <= m+p, "invalid system sizes" );
  int info=0;
  const int nm=m, nn=n, np=p;

  num_t sz;
  int lwork=-1;
  mad_alloc_tmp(num_t, ta, m*n);
  mad_alloc_tmp(num_t, tb, p*n);
  mad_alloc_tmp(num_t, tc, m);
  mad_alloc_tmp(num_t, td, p);
  mad_mat_trans(a, ta, m, n);
  mad_mat_trans(b, tb, p, n);
  mad_vec_copy (c, tc, m);
  mad_vec_copy (d, td, p);
  dgglse_(&nm, &nn, &np, ta, &nm, tb, &np, tc, td, x, &sz, &lwork, &info); // query
  lwork=sz;
  mad_alloc_tmp(num_t, wk, lwork);
  dgglse_(&nm, &nn, &np, ta, &nm, tb, &np, tc, td, x,  wk, &lwork, &info); // compute

  if (nrm_) *nrm_ = mad_vec_nrm(tc+(n-p), m-(n-p)); // residues

  mad_free_tmp(wk);
  mad_free_tmp(ta); mad_free_tmp(tb); mad_free_tmp(tc); mad_free_tmp(td);

  if (info < 0) error("GSolve: invalid input argument");
  if (info > 0) warn ("GSolve: [B A] is singular, no solution found");

  return info;
}

int
mad_cmat_gsolve (const cpx_t a[], const cpx_t b[], const cpx_t c[], const cpx_t d[],
                 cpx_t x[], ssz_t m, ssz_t n, ssz_t p, num_t *nrm_)
{
  assert( a && b && x );
  ensure( 0 <= p && p <= n && n <= m+p, "invalid system sizes" );
  int info=0;
  const int nm=m, nn=n, np=p;

  cpx_t sz;
  int lwork=-1;
  mad_alloc_tmp(cpx_t, ta, m*n);
  mad_alloc_tmp(cpx_t, tb, p*n);
  mad_alloc_tmp(cpx_t, tc, m);
  mad_alloc_tmp(cpx_t, td, p);
  mad_cmat_trans(a, ta, m, n);
  mad_cmat_trans(b, tb, p, n);
  mad_cvec_copy (c, tc, m);
  mad_cvec_copy (d, td, p);
  zgglse_(&nm, &nn, &np, ta, &nm, tb, &np, tc, td, x, &sz, &lwork, &info); // query
  lwork=sz;
  mad_alloc_tmp(cpx_t, wk, lwork);
  zgglse_(&nm, &nn, &np, ta, &nm, tb, &np, tc, td, x,  wk, &lwork, &info); // compute

  if (nrm_) *nrm_ = mad_cvec_nrm(tc+(n-p), m-(n-p)); // residues

  mad_free_tmp(wk);
  mad_free_tmp(ta); mad_free_tmp(tb); mad_free_tmp(tc); mad_free_tmp(td);

  if (info < 0) error("GSolve: invalid input argument");
  if (info > 0) warn ("GSolve: [B A] is singular, no solution found");

  return info;
}

int
mad_mat_gmsolve (const num_t a[], const num_t b[], const num_t d[],
                 num_t x[], num_t y[], ssz_t m, ssz_t n, ssz_t p)
{
  assert( a && b && x );
  ensure( 0 <= p && n <= m && m <= n+p, "invalid system sizes" );
  int info=0;
  const int nm=m, nn=n, np=p;

  num_t sz;
  int lwork=-1;
  mad_alloc_tmp(num_t, ta, m*n);
  mad_alloc_tmp(num_t, tb, m*p);
  mad_alloc_tmp(num_t, td, m);
  mad_mat_trans(a, ta, m, n);
  mad_mat_trans(b, tb, m, p);
  mad_vec_copy (d, td, m);
  dggglm_(&nm, &nn, &np, ta, &nm, tb, &nm, td, x, y, &sz, &lwork, &info); // query
  lwork=sz;
  mad_alloc_tmp(num_t, wk, lwork);
  dggglm_(&nm, &nn, &np, ta, &nm, tb, &nm, td, x, y,  wk, &lwork, &info); // compute

  mad_free_tmp(wk);
  mad_free_tmp(ta); mad_free_tmp(tb); mad_free_tmp(td);

  if (info < 0) error("GMSolve: invalid input argument");
  if (info > 0) warn ("GMSolve: [A B] is singular, no solution found");

  return info;
}

int
mad_cmat_gmsolve (const cpx_t a[], const cpx_t b[], const cpx_t d[],
                  cpx_t x[], cpx_t y[], ssz_t m, ssz_t n, ssz_t p)
{
  assert( a && b && x );
  ensure( 0 <= p && n <= m && m <= n+p, "invalid system sizes" );
  int info=0;
  const int nm=m, nn=n, np=p;

  cpx_t sz;
  int lwork=-1;
  mad_alloc_tmp(cpx_t, ta, m*n);
  mad_alloc_tmp(cpx_t, tb, m*p);
  mad_alloc_tmp(cpx_t, td, m);
  mad_cmat_trans(a, ta, m, n);
  mad_cmat_trans(b, tb, m, p);
  mad_cvec_copy (d, td, m);
  zggglm_(&nm, &nn, &np, ta, &nm, tb, &nm, td, x, y, &sz, &lwork, &info); // query
  lwork=sz;
  mad_alloc_tmp(cpx_t, wk, lwork);
  zggglm_(&nm, &nn, &np, ta, &nm, tb, &nm, td, x, y,  wk, &lwork, &info); // compute

  mad_free_tmp(wk);
  mad_free_tmp(ta); mad_free_tmp(tb); mad_free_tmp(td);

  if (info < 0) error("GMSolve: invalid input argument");
  if (info > 0) warn ("GMSolve: [A B] is singular, no solution found");

  return info;
}

// -- EIGEN -------------------------------------------------------------------o

// Eigen values and vectors
// A:[n x n], U:[m x m], S:[min(m,n)], V:[n x n]

int
mad_mat_eigen (const num_t x[], cpx_t w[], num_t vl_[], num_t vr_[], ssz_t n)
{
  assert( x && w );
  int info=0;
  const int nn=n;
  const str_t vls = vl_ ? "V" : "N";
  const str_t vrs = vr_ ? "V" : "N";

  num_t sz;
  int lwork=-1;
  mad_alloc_tmp(num_t, wr, n);
  mad_alloc_tmp(num_t, wi, n);
  mad_alloc_tmp(num_t, ra, n*n);
  mad_mat_trans(x, ra, n, n);
  dgeev_(vls, vrs, &nn, ra, &nn, wr, wi, vl_, &nn, vr_, &nn, &sz, &lwork, &info); // query
  lwork=sz;
  mad_alloc_tmp(num_t, wk, lwork);
  dgeev_(vls, vrs, &nn, ra, &nn, wr, wi, vl_, &nn, vr_, &nn,  wk, &lwork, &info); // compute
  mad_vec_cplx(wr, wi, w, n);
  mad_free_tmp(wk); mad_free_tmp(ra);
  mad_free_tmp(wi); mad_free_tmp(wr);
//if (vl_) mad_mat_trans(vl_, vl_, n, n);
  if (vr_) mad_mat_trans(vr_, vr_, n, n);

  if (info < 0) error("Eigen: invalid input argument");
  if (info > 0) warn ("Eigen: failed to compute all eigenvalues");

  return info;
}

int
mad_cmat_eigen (const cpx_t x[], cpx_t w[], cpx_t vl_[], cpx_t vr_[], ssz_t n)
{
  assert( x && w );
  int info=0;
  const int nn=n;
  const str_t vls = vl_ ? "V" : "N";
  const str_t vrs = vr_ ? "V" : "N";

  cpx_t sz;
  int lwork=-1;
  mad_alloc_tmp(num_t, rwk, 2*n);
  mad_alloc_tmp(cpx_t, ra, n*n);
  mad_cmat_trans(x, ra, n, n);
  zgeev_(vls, vrs, &nn, ra, &nn, w, vl_, &nn, vr_, &nn, &sz, &lwork, rwk, &info); // query
  lwork=creal(sz);
  mad_alloc_tmp(cpx_t, wk, lwork);
  zgeev_(vls, vrs, &nn, ra, &nn, w, vl_, &nn, vr_, &nn,  wk, &lwork, rwk, &info); // compute
  mad_free_tmp(wk); mad_free_tmp(ra); mad_free_tmp(rwk);
//if (vl_) mad_cmat_trans(vl_, vl_, n, n);
  if (vr_) mad_cmat_trans(vr_, vr_, n, n);

  if (info < 0) error("Eigen: invalid input argument");
  if (info > 0) warn ("Eigen: failed to compute all eigenvalues");

  return info;
}

// -- GEOMETRY ----------------------------------------------------------------o

// -- Helpers -----------------------------------------------------------------o

#define NN            (N*N)
#define X(i,j)        x[(i-1)*N+(j-1)]
#define VCPY(src,dst) for(idx_t i=0; i<N ; dst[i]=src[i], ++i)
#define MCPY(src,dst) for(idx_t i=0; i<NN; dst[i]=src[i], ++i)

// -- 2D geometry -------------------------------------------------------------o

#define N 2

// 2D rotation

void mad_mat_rot (num_t x[NN], num_t a) // R
{
  CHKX;
  num_t ca = cos(a), sa = sin(a);
  num_t r[NN] = {ca,-sa,
                 sa, ca};
  MCPY(r,x);
}

#undef N

// -- 3D geometry -------------------------------------------------------------o

#define N 3

// 3D rotations (one axis)

void mad_mat_rotx (num_t x[NN], num_t ax) // Rx
{
  CHKX;
  num_t cx = cos(ax), sx = sin(ax);
  num_t r[NN] = {1,  0,  0,
                 0, cx,-sx,
                 0, sx, cx};
  MCPY(r,x);
}

void mad_mat_roty (num_t x[NN], num_t ay) // Ry
{
  CHKX;
  num_t cy = cos(ay), sy = sin(ay);
  num_t r[NN] = { cy, 0, sy,
                   0, 1,  0,
                 -sy, 0, cy};
  MCPY(r,x);
}

void mad_mat_rotz (num_t x[NN], num_t az) // Rz
{
  CHKX;
  num_t cz = cos(az), sz = sin(az);
  num_t r[NN] = {cz,-sz, 0,
                 sz, cz, 0,
                  0,  0, 1};
  MCPY(r,x);
}

// 3D rotations (two axis)

void mad_mat_rotxy (num_t x[NN], num_t ax, num_t ay, log_t inv) // Ry.Rx
{
  CHKX;
  num_t cx = cos(ax), sx = sin(ax);
  num_t cy = cos(ay), sy = sin(ay);

  if (!inv) {  // normal
    num_t r[NN] = { cy, sx*sy, cx*sy,
                     0,    cx,   -sx,
                   -sy, sx*cy, cx*cy};
    MCPY(r,x);
  } else {     // transposed
    num_t r[NN] = {   cy,   0,   -sy,
                   sx*sy,  cx, sx*cy,
                   cx*sy, -sx, cx*cy};
    MCPY(r,x);
  }
}

void mad_mat_rotxz (num_t x[NN], num_t ax, num_t az, log_t inv) // Rz.Rx
{
  CHKX;
  num_t cx = cos(ax), sx = sin(ax);
  num_t cz = cos(az), sz = sin(az);

  if (!inv) {  // normal
    num_t r[NN] = {cz,-cx*sz, sx*sz,
                   sz, cx*cz,-sx*cz,
                    0,    sx,    cx};
    MCPY(r,x);
  } else {     // transposed
    num_t r[NN] = {    cz,    sz,  0,
                   -cx*sz, cx*cz, sx,
                    sx*sz,-sx*cz, cx};
    MCPY(r,x);
  }
}

void mad_mat_rotyz (num_t x[NN], num_t ay, num_t az, log_t inv) // Rz.Ry
{
  CHKX;
  num_t cy = cos(ay), sy = sin(ay);
  num_t cz = cos(az), sz = sin(az);

  if (!inv) {  // normal
    num_t r[NN] = {cy*cz,-sz, sy*cz,
                   cy*sz, cz, sy*sz,
                     -sy,  0,    cy};
    MCPY(r,x);
  } else {     // transposed
    num_t r[NN] = {cy*cz,cy*sz, -sy,
                     -sz,   cz,   0,
                   sy*cz,sy*sz,  cy};
    MCPY(r,x);
  }
}

// 3D rotations (three axis)

void mad_mat_rotxyz (num_t x[NN], num_t ax, num_t ay, num_t az, log_t inv)
{ // Rz.Ry.Rx
  CHKX;
  num_t cx = cos(ax), sx = sin(ax);
  num_t cy = cos(ay), sy = sin(ay);
  num_t cz = cos(az), sz = sin(az);

  if (!inv) {  // normal
    num_t r[NN] = {cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz,
                   cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - cz*sx,
                     -sy,            cy*sx,            cx*cy};
    MCPY(r,x);
  } else {     // transposed
    num_t r[NN] = {           cy*cz,            cy*sz,   -sy,
                   cz*sx*sy - cx*sz, sx*sy*sz + cx*cz, cy*sx,
                   cx*cz*sy + sx*sz, cx*sy*sz - cz*sx, cx*cy};
    MCPY(r,x);
  }
}

void mad_mat_rotxzy (num_t x[NN], num_t ax, num_t ay, num_t az, log_t inv)
{ // Ry.Rz.Rx
  CHKX;
  num_t cx = cos(ax), sx = sin(ax);
  num_t cy = cos(ay), sy = sin(ay);
  num_t cz = cos(az), sz = sin(az);

  if (!inv) {  // normal
    num_t r[NN] = { cy*cz, sx*sy - cx*cy*sz,  cx*sy + cy*sx*sz,
                       sz, cx*cz           , -cz*sx           ,
                   -cz*sy, cy*sx + cx*sy*sz,  cx*cy - sx*sy*sz};
    MCPY(r,x);
  } else {     // transposed
    num_t r[NN] = {cy*cz           ,     sz, -cz*sy           ,
                   sx*sy - cx*cy*sz,  cx*cz,  cy*sx + cx*sy*sz,
                   cx*sy + cy*sx*sz, -cz*sx,  cx*cy - sx*sy*sz};
    MCPY(r,x);
  }
}

void mad_mat_rotyxz (num_t x[NN], num_t ax, num_t ay, num_t az, log_t inv)
{ // Rz.Rx.Ry
  CHKX;
  num_t cx = cos(ax), sx = sin(ax);
  num_t cy = cos(ay), sy = sin(ay);
  num_t cz = cos(az), sz = sin(az);

  if (!inv) {  // normal
    num_t r[NN] = { cy*cz - sx*sy*sz, -cx*sz, cz*sy + cy*sx*sz,
                    cy*sz + cz*sx*sy,  cx*cz, sy*sz - cy*cz*sx,
                   -cx*sy           ,     sx, cx*cy           };
    MCPY(r,x);
  } else {     // transposed
    num_t r[NN] = { cy*cz - sx*sy*sz, cy*sz + cz*sx*sy, -cx*sy,
                   -cx*sz           , cx*cz           ,     sx,
                    cz*sy + cy*sx*sz, sy*sz - cy*cz*sx,  cx*cy};
    MCPY(r,x);
  }
}

// 3D angles from rotations

void mad_mat_torotxyz (const num_t x[NN], num_t r[N], log_t inv)
{ // extract ax, ay, az from rotxyz
  CHKXR;

  num_t x11 = X(1,1), x33 = X(3,3), x21, x31, x32;

  if (!inv) x21 = X(2,1), x31 = X(3,1), x32 = X(3,2);
  else      x21 = X(1,2), x31 = X(1,3), x32 = X(2,3);

  r[0] = atan2( x32, x33 );                     // ax
  r[1] = atan2(-x31, sqrt(x32*x32 + x33*x33) ); // ay
  r[2] = atan2( x21, x11 );                     // az
}

void mad_mat_torotxzy (const num_t x[NN], num_t r[N], log_t inv)
{ // extract ax, ay, az from rotxzy
  CHKXR;
  num_t x11 = X(1,1), x22 = X(2,2), x21, x23, x31;

  if (!inv) x21 = X(2,1), x23 = X(2,3), x31 = X(3,1);
  else      x21 = X(1,2), x23 = X(3,2), x31 = X(1,3);

  r[0] = atan2(-x23, x22 );                     // ax
  r[1] = atan2(-x31, x11 );                     // ay
  r[2] = atan2( x21, sqrt(x22*x22 + x23*x23) ); // az
}

void mad_mat_torotyxz (const num_t x[NN], num_t r[N], log_t inv)
{ // extract ax, ay, az from rotyxz
  CHKXR;
  num_t x22 = X(2,2), x33 = X(3,3), x12, x31, x32;

  if (!inv) x12 = X(1,2), x31 = X(3,1), x32 = X(3,2);
  else      x12 = X(2,1), x31 = X(1,3), x32 = X(2,3);

  r[0] = atan2( x32, sqrt(x12*x12 + x22*x22) ); // ax
  r[1] = atan2(-x31, x33 );                     // ay
  r[2] = atan2(-x12, x22 );                     // az
}

// 3D vector rotation

void mad_mat_rotv (num_t x[NN], const num_t v[N], num_t a, log_t inv)
{
  assert(x && v);

  num_t vx = v[0], vy = v[1], vz = v[2];
  num_t n = vx*vx + vy*vy + vz*vz;

  if (n == 0) {
    mad_mat_eye(x, 1, N, N, N);
    return;
  }

  if (n != 1) {
    n = 1/sqrt(n);
    vx *= n, vy *= n, vz *= n;
  }

  num_t xx = vx*vx,  yy = vy*vy,  zz = vz*vz;
  num_t xy = vx*vy,  xz = vx*vz,  yz = vy*vz;
  num_t ca = cos(a), sa = sin(a), C  = 1-ca;

  if (!inv) {  // normal
    num_t r[NN] = {xx*C +    ca, xy*C - vz*sa, xz*C + vy*sa,
                   xy*C + vz*sa, yy*C +    ca, yz*C - vx*sa,
                   xz*C - vy*sa, yz*C + vx*sa, zz*C +    ca};
    MCPY(r,x);
  } else {     // transposed
    num_t r[NN] = {xx*C +    ca, xy*C + vz*sa, xz*C - vy*sa,
                   xy*C - vz*sa, yy*C +    ca, yz*C + vx*sa,
                   xz*C + vy*sa, yz*C - vx*sa, zz*C +    ca};
    MCPY(r,x);
  }
}

num_t mad_mat_torotv (const num_t x[NN], num_t v_[N], log_t inv)
{
  CHKX;
  num_t vx, vy, vz;

  if (!inv) {
    vx = X(3,2) - X(2,3);
    vy = X(1,3) - X(3,1);
    vz = X(2,1) - X(1,2);
  } else {
    vx = X(2,3) - X(3,2);
    vy = X(3,1) - X(1,3);
    vz = X(1,2) - X(2,1);
  }

  num_t n = sqrt(vx*vx + vy*vy + vz*vz);
  num_t t = X(1,1) + X(2,2) + X(3,3);
  num_t a = atan2(n, t-1);

  if (v_) {
    n = n != 0 ? 1/n : 0;
    v_[0] = n*vx, v_[1] = n*vy, v_[2] = n*vz;
  }
  return a;
}

// Quaternion

void mad_mat_rotq (num_t x[NN], const num_t q[4], log_t inv)
{
  assert(x && q);

  num_t qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  num_t n = qw*qw + qx*qx + qy*qy + qz*qz;
  num_t s = n != 0 ? 2/n : 0;
  num_t wx = s*qw*qx, wy = s*qw*qy, wz = s*qw*qz;
  num_t xx = s*qx*qx, xy = s*qx*qy, xz = s*qx*qz;
  num_t yy = s*qy*qy, yz = s*qy*qz, zz = s*qz*qz;

  if (!inv) {  // normal
    num_t r[NN] = {1-(yy+zz),    xy-wz ,    xz+wy,
                      xy+wz , 1-(xx+zz),    yz-wx,
                      xz-wy ,    yz+wx , 1-(xx+yy)};
    MCPY(r,x);
  } else {     // transposed
    num_t r[NN] = {1-(yy+zz),    xy+wz ,    xz-wy,
                      xy-wz , 1-(xx+zz),    yz+wx,
                      xz+wy ,    yz-wx , 1-(xx+yy)};
    MCPY(r,x);
  }
}

void mad_mat_torotq (const num_t x[NN], num_t q[4], log_t inv)
{
  CHKX;
  num_t xx = X(1,1), yy = X(2,2), zz = X(3,3);
  num_t tt = xx+yy+zz, rr, ss;

  // stable trace
  if (tt > -0.99999) {
    rr = sqrt(1+tt), ss = 0.5/rr;
    q[0] = 0.5*rr;
    if (!inv) {  // normal
      q[1] = (X(3,2) - X(2,3)) * ss;
      q[2] = (X(1,3) - X(3,1)) * ss;
      q[3] = (X(2,1) - X(1,2)) * ss;
    } else {     // transposed
      q[1] = (X(2,3) - X(3,2)) * ss;
      q[2] = (X(3,1) - X(1,3)) * ss;
      q[3] = (X(1,2) - X(2,1)) * ss;
    }
    return;
  }

  // look for more stable trace
  num_t m = MAX(xx, yy, zz);
  if (!inv) {  // normal
    if (m == xx) {
      rr = sqrt(1+xx-yy-zz), ss = 0.5/rr;
      q[1] = 0.5*rr;
      q[0] = (X(3,2) - X(2,3)) * ss;
      q[2] = (X(1,3) + X(3,1)) * ss;
      q[3] = (X(2,1) + X(1,2)) * ss;
    } else if (m == yy) {
      rr = sqrt(1+yy-xx-zz), ss = 0.5/rr;
      q[2] = 0.5*rr;
      q[0] = (X(3,2) - X(2,3)) * ss;
      q[1] = (X(1,3) - X(3,1)) * ss;
      q[3] = (X(2,1) + X(1,2)) * ss;
    } else {
      rr = sqrt(1+zz-xx-yy), ss = 0.5/rr;
      q[3] = 0.5*rr;
      q[0] = (X(3,2) - X(2,3)) * ss;
      q[1] = (X(1,3) - X(3,1)) * ss;
      q[2] = (X(2,1) - X(1,2)) * ss;
    }
  } else {     // transposed
    if (m == xx) {
      rr = sqrt(1+xx-yy-zz), ss = 0.5/rr;
      q[1] = 0.5*rr;
      q[0] = (X(2,3) - X(3,2)) * ss;
      q[2] = (X(3,1) + X(1,3)) * ss;
      q[3] = (X(1,2) + X(2,1)) * ss;
    } else if (m == yy) {
      rr = sqrt(1+yy-xx-zz), ss = 0.5/rr;
      q[2] = 0.5*rr;
      q[0] = (X(2,3) - X(3,2)) * ss;
      q[1] = (X(3,1) - X(1,3)) * ss;
      q[3] = (X(1,2) + X(2,1)) * ss;
    } else {
      rr = sqrt(1+zz-xx-yy), ss = 0.5/rr;
      q[3] = 0.5*rr;
      q[0] = (X(2,3) - X(3,2)) * ss;
      q[1] = (X(3,1) - X(1,3)) * ss;
      q[2] = (X(1,2) - X(2,1)) * ss;
    }
  }
}

#undef N
#undef X

#undef N

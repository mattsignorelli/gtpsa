/*
 o-----------------------------------------------------------------------------o
 |
 | TPSA map inversion module implementation
 |
 | Methodical Accelerator Design - Copyright (c) 2016+
 | Support: http://cern.ch/mad  - mad at cern.ch
 | Authors: L. Deniau, laurent.deniau at cern.ch
 |          C. Tomoiaga
 | Contrib: -
 |
 o-----------------------------------------------------------------------------o
 | You can redistribute this file and/or modify it under the terms of the GNU
 | General Public License GPLv3 (or later), as published by the Free Software
 | Foundation. This file is distributed in the hope that it will be useful, but
 | WITHOUT ANY WARRANTY OF ANY KIND. See http://gnu.org/licenses for details.
 o-----------------------------------------------------------------------------o
*/

#include "mad_mem.h"
#include "mad_vec.h"
#include "mad_mat.h"
#ifdef    MAD_CTPSA_IMPL
#include "mad_ctpsa_impl.h"
#else
#include "mad_tpsa_impl.h"
#endif

#define DEBUG_MINV 0
#define TC (const T**)

// --- local ------------------------------------------------------------------o

static inline void
check_same_desc(ssz_t na, const T *ma[na])
{
  assert(ma);
  FOR(i,1,na)
    ensure(ma[i]->d == ma[i-1]->d, "inconsistent GTPSAs (descriptors differ)");
}

static inline void
check_minv(ssz_t na, const T *ma[na], ssz_t nb, T *mc[na], idx_t select[na])
{
  ensure(na <= ma[0]->d->nn, "invalid na > #vars+#params");
  ensure(nb <= ma[0]->d->nv, "invalid nb > #vars");
  check_same_desc(na,   ma);
  check_same_desc(na,TC mc);
  ensure(IS_COMPAT(*ma,*mc), "incompatibles GTPSA (descriptors differ)");

  FOR(i,na) if (!select || select[i]) {
    ensure(ma[i]->lo == 1 && ma[i]->hi != 0,
           "invalid rank-deficient map (1st order has row(s) full of zeros)"); }

  FOR(i,nb,na) if (!select || select[i]) {
    ensure(ma[i]->lo == 1 && ma[i]->hi == 1,
           "invalid parameter orders (order > 1)"); }
}

static void      // nn                         nv
split_and_inv(ssz_t na, const T *ma[na], ssz_t nb, T *lininv[na], T *nonlin[na])
{
  ssz_t nn = na, nv = nb, np = nn-nv;  // #vars+#params, #vars, #params
  mad_alloc_tmp(NUM, mat_var , nv*nv); // canonical vars
  mad_alloc_tmp(NUM, mat_vari, nv*nv); // inverse of vars
  mad_alloc_tmp(NUM, mat_par , nv*np); // parameters
  mad_alloc_tmp(NUM, mat_pari, nv*np); // 'inverse' of parameters

  // split linear, (-1 * nonlinear)
  FOR(i,nv) {
    FUN(getv)(ma[i], 1   , nv, mat_var + i*nv);
    FUN(getv)(ma[i], 1+nv, np, mat_par + i*np);

    T *t = nonlin[i];
    FUN(copy)(ma[i], t);
    FUN(cutord)(t,t,-1); // keep orders 2+ (i.e. cut 0..1)
    FUN(scl)(t,-1,t);
  }

  // invert linear part: mat_vari = mat_var^-1
# ifndef MAD_CTPSA_IMPL
  mad_mat_invn(mat_var, 1, mat_vari, nv, nv, -1);
  if (np) {
    // mat_pari = - mat_vari * mat_par
    mad_mat_mul(mat_vari, mat_par, mat_pari, nv, np, nv);
    mad_vec_muln(mat_pari, -1, mat_pari, nv*np);
  }
# else
  mad_cmat_invn(mat_var, 1, mat_vari, nv, nv, -1);
  if (np) {
    // mat_pari = - mat_vari * mat_par
    mad_cmat_mul(mat_vari, mat_par, mat_pari, nv, np, nv);
    mad_cvec_muln(mat_pari, -1, mat_pari, nv*np);
  }
# endif

  // copy result into TPSA
  FOR(i,nv) {
    FUN(setv)(lininv[i], 1   , nv, mat_vari + i*nv);
    FUN(setv)(lininv[i], 1+nv, np, mat_pari + i*np);
  }

  mad_free_tmp(mat_var );
  mad_free_tmp(mat_vari);
  mad_free_tmp(mat_par );
  mad_free_tmp(mat_pari);
}

// --- public -----------------------------------------------------------------o

void          // nn                         nv
FUN(minv) (ssz_t na, const T *ma[na], ssz_t nb, T *mc[na])
{
  assert(ma && mc); DBGFUN(->);
  ensure(na >= nb, "invalid subtitution ranks, na >= nb expected");
  check_minv(na, ma, nb, mc, NULL);

  const D *d = ma[0]->d;
  T *lininv[na], *nonlin[na], *tmp[na];
  FOR(i,nb) { // vars
    lininv[i] = FUN(newd)(d,1);
    nonlin[i] = FUN(new)(ma[i], mad_tpsa_same);
    tmp[i]    = FUN(new)(mc[i], mad_tpsa_same);
  }
  FOR(i,nb,na) { // params
    lininv[i] = (T*)ma[i];
    nonlin[i] = (T*)ma[i];
    tmp[i]    = (T*)ma[i];
  }

  split_and_inv(na, ma, nb, lininv, nonlin);

#if DEBUG_MINV
  printf("\nminv lininv:\n"); FOR(i,nb) FUN(print)(lininv[i],0,-1,0,0);
  printf("\nminv nonlin:\n"); FOR(i,nb) FUN(print)(nonlin[i],0,-1,0,0);
#endif

  // iteratively compute higher orders of the inverse:
  // al  = mc[ord=1]
  // anl = mc[ord>1]
  // mc[ord=1] = al^-1
  // mc[ord=i] = al^-1 o ( anl o mc[ord=i-1] + id ) ; i > 1

  log_t isnul = TRUE;
  FOR(i,nb) {
    FUN(copy)(lininv[i], mc[i]);
    isnul &= FUN(isnul)(nonlin[i]);
  }

  if (!isnul) {
    ord_t mo[nb], hi[nb], to=FUN(mord)(nb, TC mc, FALSE), dbgo=mad_tpsa_dbgo;
    FOR(i,nb) mo[i] = FUN(ord)(mc    [i], FALSE); // backup mo[i]
    FOR(i,nb) hi[i] = FUN(ord)(nonlin[i], TRUE ); // backup hi[i]
    for (ord_t o=2; o <= to; ++o) {
      mad_tpsa_dbgo = o;                         // for debug purpose only
      FOR(i,nb) FUN(mo)(    mc[i],MIN(o,mo[i])); // truncate mo to order o
      FOR(i,nb) FUN(mo)(   tmp[i],MIN(o,mo[i]));
      FOR(i,nb) FUN(mo)(nonlin[i],MIN(o,hi[i])), nonlin[i]->hi = MIN(o,hi[i]);
      FUN(compose)(nb, TC nonlin, na, TC mc, tmp);
      FOR(v,nb) FUN(seti)(tmp[v], v+1, 1,1); // add identity
      FUN(compose)(nb, TC lininv, na, TC tmp, mc);

#if DEBUG_MINV
      printf("\nminv tmp[o=%d]:\n",o); FOR(i,nb) FUN(print)(tmp[i],0,-1,0,0);
      printf("\nminv mc[o=%d]:\n" ,o); FOR(i,nb) FUN(print)(mc [i],0,-1,0,0);
#endif
    }
    FOR(i,nb) FUN(mo)(mc[i], mo[i]); // restore mo[i]
    mad_tpsa_dbgo = dbgo;
  }

  // cleanup
  FOR(i,nb) {
    FUN(del)(lininv[i]);
    FUN(del)(nonlin[i]);
    FUN(del)(tmp[i]);
  }
  DBGFUN(<-);
}

void           // na                         nv
FUN(pminv) (ssz_t na, const T *ma[na], ssz_t nb, T *mc[na], idx_t select[na])
{
  assert(ma && mc && select); DBGFUN(->);
  ensure(na >= nb, "invalid subtitution ranks, na >= nb expected");
  check_minv(na, ma, nb, mc, select);

  // split input map into rows that are inverted and rows that are not

  const D *d = ma[0]->d;
  T *mUsed[na], *mUnused[na], *mInv[na];
  FOR(i,nb) { // vars
    if (select[i]) { // used
      mUsed  [i] = FUN(new) (ma[i], mad_tpsa_same);
      mInv   [i] = FUN(new) (ma[i], mad_tpsa_same);
      mUnused[i] = FUN(newd)(d,1);
      FUN(copy)(ma[i],mUsed[i]);
      FUN(seti)(mUnused[i], i+1,  0,1); // set identity
    }
    else {           // unused
      mUsed  [i] = FUN(newd)(d,1);
      mInv   [i] = FUN(newd)(d,1);
      mUnused[i] = FUN(new) (ma[i], mad_tpsa_same);
      FUN(copy)(ma[i],mUnused[i]);
      FUN(seti)(mUsed[i], i+1, 0,1); // set identity
    }
    FUN(seti)(mUsed  [i], 0,0,0);
    FUN(seti)(mUnused[i], 0,0,0);
  }
  FOR(i,nb,na) { // params
    mUsed  [i] = (T*)ma[i];
    mInv   [i] = (T*)ma[i];
    mUnused[i] = (T*)ma[i];
  }

  FUN(minv)   (na, TC mUsed  , nb,    mInv);
  FUN(compose)(nb, TC mUnused, na, TC mInv, mc);

  FOR(i,nb) {
    FUN(del)(mUsed[i]);
    FUN(del)(mUnused[i]);
    FUN(del)(mInv[i]);
  }
  DBGFUN(<-);
}

// --- end --------------------------------------------------------------------o

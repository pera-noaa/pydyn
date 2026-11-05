/* convec (kernels.convection_matrix, fastminv, convec): one thread per column, the matrix,
   its inverse and the update held in thread-local arrays of PD_LMAXC levels.

   Two inverses, both open, both bitwise between this file's host and device builds:
     0  Crout without pivoting, kernels.fastminv's loops (the build without with_lapack)
     1  reference LAPACK's DGETRF + DGETRI as lapack.LapackInverse calls them (lwork = n, so
        DGETRF2's recursion and DGETRI's unblocked column sweep, reference BLAS loops), the
        same functions the host build exports as dgetrf_ / dgetri_ for the CPU reference */
#ifndef PD_CONVEC
#define PD_CONVEC
#include "pd.h"

#ifndef PD_LMAXC
#define PD_LMAXC 40
#endif

/* column-major n x n (lda) as LAPACK sees it */
#define CM(a, lda, i, j) (a)[(i) + (i64)(lda) * (j)]

/* ------------------------------------------------------------------ reference BLAS / LAPACK */
PD_FN_NOINLINE int pd_idamax(int n, PD_L const real *x) /* 0-based */
{
    int im = 0;
    real dmax = pd_fabs(x[0]);
    for (int i = 1; i < n; i++)
        if (pd_fabs(x[i]) > dmax) {
            im = i;
            dmax = pd_fabs(x[i]);
        }
    return im;
}

/* DLASWP(n, a, lda, k1, k2, ipiv, 1): k1, k2 and ipiv 1-based */
PD_FN_NOINLINE void pd_dlaswp(int n, PD_L real *a, int lda, int k1, int k2, PD_L const int *ipiv)
{
    for (int i = k1; i <= k2; i++) {
        int ip = ipiv[i - 1];
        if (ip != i)
            for (int c = 0; c < n; c++) {
                real t = CM(a, lda, i - 1, c);
                CM(a, lda, i - 1, c) = CM(a, lda, ip - 1, c);
                CM(a, lda, ip - 1, c) = t;
            }
    }
}

/* DTRSM('L', 'L', 'N', 'U', m, n, one, a, lda, b, ldb) */
PD_FN_NOINLINE void pd_dtrsm_llnu(int m, int n, PD_L const real *a, int lda, PD_L real *b, int ldb)
{
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++)
            if (CM(b, ldb, k, j) != PD_R(0.0))
                for (int i = k + 1; i < m; i++) CM(b, ldb, i, j) = CM(b, ldb, i, j) - CM(b, ldb, k, j) * CM(a, lda, i, k);
}

/* DGEMM('N', 'N', m, n, k, alpha, a, lda, b, ldb, one, c, ldc) */
PD_FN_NOINLINE void pd_dgemm_nn(int m, int n, int k, real alpha, PD_L const real *a, int lda, PD_L const real *b,
                                int ldb, PD_L real *c, int ldc)
{
    for (int j = 0; j < n; j++)
        for (int l = 0; l < k; l++) {
            real temp = alpha * CM(b, ldb, l, j);
            for (int i = 0; i < m; i++) CM(c, ldc, i, j) = CM(c, ldc, i, j) + temp * CM(a, lda, i, l);
        }
}

/* DGETRF2(m, n, a, lda, ipiv, info), recursive; ipiv 1-based */
PD_FN_NOINLINE int pd_dgetrf2(int m, int n, PD_L real *a, int lda, PD_L int *ipiv)
{
    int info = 0;
    if (m == 0 || n == 0) return 0;
    if (m == 1) {
        ipiv[0] = 1;
        if (CM(a, lda, 0, 0) == PD_R(0.0)) info = 1;
    } else if (n == 1) {
        const real sfmin = PD_REAL_TINY;
        int i = pd_idamax(m, a);
        ipiv[0] = i + 1;
        if (CM(a, lda, i, 0) != PD_R(0.0)) {
            if (i != 0) {
                real t = CM(a, lda, 0, 0);
                CM(a, lda, 0, 0) = CM(a, lda, i, 0);
                CM(a, lda, i, 0) = t;
            }
            if (pd_fabs(CM(a, lda, 0, 0)) >= sfmin) {
                real r = PD_R(1.0) / CM(a, lda, 0, 0);
                for (int q = 1; q < m; q++) CM(a, lda, q, 0) = r * CM(a, lda, q, 0); /* DSCAL */
            } else {
                for (int q = 1; q < m; q++) CM(a, lda, q, 0) = CM(a, lda, q, 0) / CM(a, lda, 0, 0);
            }
        } else {
            info = 1;
        }
    } else {
#ifdef PD_METAL
        info = -1; /* unreachable: pd_dgetrf2_iter calls only the leaves */
#else
        int mn = m < n ? m : n, n1 = mn / 2, n2 = n - n1;
        int iinfo = pd_dgetrf2(m, n1, a, lda, ipiv);
        if (info == 0 && iinfo > 0) info = iinfo;
        pd_dlaswp(n2, &CM(a, lda, 0, n1), lda, 1, n1, ipiv);
        pd_dtrsm_llnu(n1, n2, a, lda, &CM(a, lda, 0, n1), lda);
        pd_dgemm_nn(m - n1, n2, n1, -PD_R(1.0), &CM(a, lda, n1, 0), lda, &CM(a, lda, 0, n1), lda, &CM(a, lda, n1, n1), lda);
        iinfo = pd_dgetrf2(m - n1, n2, &CM(a, lda, n1, n1), lda, ipiv + n1);
        if (info == 0 && iinfo > 0) info = iinfo + n1;
        for (int i = n1; i < mn; i++) ipiv[i] = ipiv[i] + n1;
        pd_dlaswp(n1, a, lda, n1 + 1, mn, ipiv);
#endif
    }
    return info;
}

/* pd_dgetrf2 without recursion (MSL has none): the same calls in the same order, its frames on
   an explicit stack (depth log2(n) + 2).  The leaves are pd_dgetrf2's m == 1 and n == 1. */
struct pd_lu_frame {
    int m, n, ao, po, st, info, n1, n2, mn;
};

PD_FN int pd_dgetrf2_iter(int m0, int n0, PD_L real *a0, int lda, PD_L int *ipiv0)
{
    pd_lu_frame S[16];
    int sp = 0, ret = 0;
    S[0].m = m0, S[0].n = n0, S[0].ao = 0, S[0].po = 0, S[0].st = 0;
    while (sp >= 0) {
        PD_L pd_lu_frame *f = &S[sp];
        PD_L real *a = a0 + f->ao;
        PD_L int *ipiv = ipiv0 + f->po;
        if (f->st == 0) {
            if (f->m == 0 || f->n == 0 || f->m == 1 || f->n == 1) {
                ret = pd_dgetrf2(f->m, f->n, a, lda, ipiv); /* a leaf: no recursion */
                sp--;
                continue;
            }
            f->mn = f->m < f->n ? f->m : f->n;
            f->n1 = f->mn / 2;
            f->n2 = f->n - f->n1;
            f->info = 0;
            f->st = 1;
            PD_L pd_lu_frame *c = &S[sp + 1];
            c->m = f->m, c->n = f->n1, c->ao = f->ao, c->po = f->po, c->st = 0;
            sp++;
            continue;
        }
        int n1 = f->n1, n2 = f->n2;
        if (f->st == 1) {
            if (f->info == 0 && ret > 0) f->info = ret;
            pd_dlaswp(n2, &CM(a, lda, 0, n1), lda, 1, n1, ipiv);
            pd_dtrsm_llnu(n1, n2, a, lda, &CM(a, lda, 0, n1), lda);
            pd_dgemm_nn(f->m - n1, n2, n1, PD_R(-1.0), &CM(a, lda, n1, 0), lda, &CM(a, lda, 0, n1), lda,
                        &CM(a, lda, n1, n1), lda);
            f->st = 2;
            PD_L pd_lu_frame *c = &S[sp + 1];
            c->m = f->m - n1, c->n = n2, c->ao = f->ao + n1 + lda * n1, c->po = f->po + n1, c->st = 0;
            sp++;
            continue;
        }
        if (f->info == 0 && ret > 0) f->info = ret + n1;
        for (int i = n1; i < f->mn; i++) ipiv[i] = ipiv[i] + n1;
        pd_dlaswp(n1, a, lda, n1 + 1, f->mn, ipiv);
        ret = f->info;
        sp--;
    }
    return ret;
}

/* DGETRI(n, a, lda, ipiv, work, lwork = n): DTRTRI -> DTRTI2 (upper, non-unit), then the
   unblocked solve of inv(A) L = inv(U), then the column interchanges */
PD_FN_NOINLINE int pd_dgetri(int n, PD_L real *a, int lda, PD_L const int *ipiv, PD_L real *work)
{
    if (n == 0) return 0;
    for (int i = 0; i < n; i++)
        if (CM(a, lda, i, i) == PD_R(0.0)) return i + 1;
    for (int j = 0; j < n; j++) { /* DTRTI2 */
        CM(a, lda, j, j) = PD_R(1.0) / CM(a, lda, j, j);
        real ajj = -CM(a, lda, j, j);
        PD_L real *x = &CM(a, lda, 0, j); /* DTRMV('U', 'N', 'N', j, a, lda, x, 1) */
        for (int jj = 0; jj < j; jj++)
            if (x[jj] != PD_R(0.0)) {
                real temp = x[jj];
                for (int i = 0; i < jj; i++) x[i] = x[i] + temp * CM(a, lda, i, jj);
                x[jj] = x[jj] * CM(a, lda, jj, jj);
            }
        for (int i = 0; i < j; i++) x[i] = ajj * x[i]; /* DSCAL */
    }
    for (int j = n - 1; j >= 0; j--) {
        for (int i = j + 1; i < n; i++) {
            work[i] = CM(a, lda, i, j);
            CM(a, lda, i, j) = PD_R(0.0);
        }
        if (j < n - 1) /* DGEMV('N', n, n-1-j, -one, a(1, j+1), lda, work(j+1), 1, one, a(1, j), 1) */
            for (int jj = 0; jj < n - 1 - j; jj++) {
                real temp = -PD_R(1.0) * work[j + 1 + jj];
                for (int i = 0; i < n; i++) CM(a, lda, i, j) = CM(a, lda, i, j) + temp * CM(a, lda, i, j + 1 + jj);
            }
    }
    for (int j = n - 2; j >= 0; j--) {
        int jp = ipiv[j] - 1;
        if (jp != j)
            for (int i = 0; i < n; i++) {
                real t = CM(a, lda, i, j);
                CM(a, lda, i, j) = CM(a, lda, i, jp);
                CM(a, lda, i, jp) = t;
            }
    }
    return 0;
}

/* kernels.fastminv on one matrix: lu in place in a, the inverse into y */
PD_FN void pd_crout(int n, PD_L real *a, PD_L real *y)
{
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < j; k++)
            for (int r = k + 1; r < n; r++) CM(a, n, r, j) = CM(a, n, r, j) - CM(a, n, r, k) * CM(a, n, k, j);
        for (int r = j + 1; r < n; r++) CM(a, n, r, j) = CM(a, n, r, j) / CM(a, n, j, j);
    }
    for (int i = 0; i < n; i++)
        for (int c = 0; c < n; c++) CM(y, n, i, c) = i == c ? PD_R(1.0) : PD_R(0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            for (int c = 0; c < n; c++) CM(y, n, i, c) = CM(y, n, i, c) - CM(a, n, i, j) * CM(y, n, j, c);
    for (int i = n - 1; i >= 0; i--) { /* inv(i, :) = y(i, :), in place */
        for (int j = i + 1; j < n; j++)
            for (int c = 0; c < n; c++) CM(y, n, i, c) = CM(y, n, i, c) - CM(a, n, i, j) * CM(y, n, j, c);
        for (int c = 0; c < n; c++) CM(y, n, i, c) = CM(y, n, i, c) / CM(a, n, i, i);
    }
}

/* the LAPACK kind's inverse alone, for its tests: ncol row-major n x n matrices A to Y */
PD_GLOBAL void pd_inverse_rows(i64 ncol, int n, PD_G const real *A, PD_G real *Y, PD_G int *info)
{
    PD_LOOP(q, ncol)
    {
        real a[(PD_LMAXC + 1) * PD_LMAXC], y[(PD_LMAXC + 1) * PD_LMAXC];
        int ipiv[PD_LMAXC];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++) CM(a, n, r, c) = A[q * n * n + r * n + c];
#if defined(PD_METAL) || defined(PD_DGETRF2_ITER)
        int e = pd_dgetrf2_iter(n, n, a, n, ipiv);
#else
        int e = pd_dgetrf2(n, n, a, n, ipiv);
#endif
        if (e == 0) e = pd_dgetri(n, a, n, ipiv, y);
        if (e != 0) pd_or_i(info, 1);
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++) Y[q * n * n + r * n + c] = CM(a, n, r, c);
    }
}

/* ------------------------------------------------------------------ convec, one column */
#ifdef PD_F32
/* float only: the column's tracer before minus after, each summed exactly enough (TwoSum, in
   level order), added to its largest level.  conv1's diagonal is 1 + a small outflow whose low
   bits a float drops while the off-diagonal inflows keep theirs, so its column sums miss 1 by
   an error of one sign, which the inverse kept for the 3-h interval applies every step: -1.2e-8
   of the tracer a call, under an ulp of every level, so only one level can carry it back. */
PD_FN void pd_two_sum(PD_L real *s, PD_L real *c, real x)
{
    real t = *s + x, bp = t - *s;
    *c = *c + ((*s - (t - bp)) + (x - bp));
    *s = t;
}
PD_FN void pd_column_deficit(int n, PD_L const real *before, PD_L real *after)
{
    real sb = PD_R(0.0), cb = PD_R(0.0), sa = PD_R(0.0), ca = PD_R(0.0);
    for (int k = 0; k < n; k++) pd_two_sum(&sb, &cb, before[k]);
    for (int l = 0; l < n; l++) pd_two_sum(&sa, &ca, after[l]);
    int lx = 0;
    for (int l = 1; l < n; l++)
        if (after[l] > after[lx]) lx = l;
    after[lx] = after[lx] + ((sb - sa) + (cb - ca));
}
#endif
/* kernels.convection_matrix for one column into a (column-major n x n); fu is scratch of
   (n+1) n.  fd/f/conv1 share `a` as rows of stride n until the closing transpose. */
PD_FN void pd_convection_matrix(int n, PD_G const real *m, i64 sm, PD_G const real *entu, PD_G const real *detu,
                                PD_G const real *entd, PD_G const real *detd, i64 se, PD_G const real *dkg, i64 sd,
                                int ctop, int clfs, real dt, PD_L real *a, PD_L real *fu)
{
#define FU(k, p) fu[(i64)(k) * n + (p)]
#define FD(k, p) a[(i64)(k) * n + (p)]
    real amu[PD_LMAXC + 1], amd[PD_LMAXC + 1];
    for (i64 q = 0; q < (i64)(n + 1) * n; q++) fu[q] = a[q] = PD_R(0.0);
    for (int k = 0; k <= n; k++) amu[k] = amd[k] = PD_R(0.0);
    for (int k = 1; k <= n; k++) { /* updraft */
        int act = ctop >= k;
        real eu = entu[(k - 1) * se], du = detu[(k - 1) * se];
        real av = amu[k - 1] + eu - du;
        int up = av > PD_R(0.0);
        real zxi = up ? np_maximum(PD_R(0.0), PD_R(1.0) - du / (amu[k - 1] + eu)) : PD_R(0.0);
        amu[k] = act && up ? av : PD_R(0.0);
        for (int p = 0; p < k - 1; p++) FU(k, p) = act ? FU(k - 1, p) * zxi : PD_R(0.0);
        FU(k, k - 1) = act ? eu / m[(k - 1) * sm] * zxi : PD_R(0.0);
    }
    for (int k = n; k > 1; k--) { /* downdraft (amd negative downward) */
        int act = clfs >= k;
        real ed = entd[(k - 1) * se], dd = detd[(k - 1) * se];
        real av = amd[k] - ed + dd;
        int dn = av < PD_R(0.0);
        real zxi = dn ? np_maximum(PD_R(0.0), PD_R(1.0) + dd / (amd[k] - ed)) : PD_R(0.0);
        amd[k - 1] = act && dn ? av : PD_R(0.0);
        for (int p = k; p < n; p++) FD(k - 1, p) = act ? FD(k, p) * zxi : FD(k - 1, p);
        FD(k - 1, k - 1) = act ? -ed / m[(k - 1) * sm] * zxi : FD(k - 1, k - 1);
    }
    for (int k = 1; k < n; k++) { /* combine, compensating subsidence, diffusion */
        real mk = m[k * sm], mk1 = m[(k - 1) * sm], dg = dkg[(k - 1) * sd];
        for (int p = 0; p < n; p++) FD(k, p) = FU(k, p) + FD(k, p);
        FD(k, k) = FD(k, k) - amu[k] / mk;
        FD(k, k - 1) = FD(k, k - 1) - amd[k] / mk1;
        FD(k, k - 1) = FD(k, k - 1) + dg / mk1;
        FD(k, k) = FD(k, k) - dg / mk;
    }
    for (int k = 1; k <= n; k++) { /* conv1, in place: row k-1 needs rows k-1 and k of f */
        for (int p = 0; p < n; p++) FD(k - 1, p) = -dt * (FD(k - 1, p) - FD(k, p));
        FD(k - 1, k - 1) = FD(k - 1, k - 1) + PD_R(1.0);
    }
    for (int i = 0; i < n; i++) /* rows of stride n -> column-major */
        for (int j = i + 1; j < n; j++) {
            real t = a[(i64)i * n + j];
            a[(i64)i * n + j] = a[(i64)j * n + i];
            a[(i64)j * n + i] = t;
        }
#undef FU
#undef FD
}

/* convec on the listed columns (flat i * jm + j of the (im, jm) views): matrix, inverse, the
   slopes scaled by its diagonal, rm(l) = sum_k C(l, k) rm(k) accumulated in k. */
PD_GLOBAL void pd_convec(i64 ncol, PD_G const int *cols, i64 jm, i64 nt, int lmax, real dt, int kind, PD_G real *m,
                         i64 sm0, i64 sm1, i64 sm2, PD_G real *rm, PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 sq0,
                         i64 sq1, i64 sq2, i64 sq3, PD_G const real *entu, PD_G const real *detu, PD_G const real *entd,
                         PD_G const real *detd, i64 se0, i64 se1, i64 se2, PD_G const real *dkg, i64 sd0, i64 sd1,
                         i64 sd2, PD_G const int *ctop, PD_G const int *clfs, i64 sc0, i64 sc1, PD_G int *info)
{
    PD_LOOP(q, ncol)
    {
        i64 col = cols[q], i = col / jm, j = col % jm;
        int n = lmax;
        real a[(PD_LMAXC + 1) * PD_LMAXC], y[(PD_LMAXC + 1) * PD_LMAXC];
        int ipiv[PD_LMAXC];
        i64 oe = i * se0 + j * se1;
        pd_convection_matrix(n, m + i * sm0 + j * sm1, sm2, entu + oe, detu + oe, entd + oe, detd + oe, se2,
                             dkg + i * sd0 + j * sd1, sd2, ctop[i * sc0 + j * sc1], clfs[i * sc0 + j * sc1], dt, a, y);
        PD_L real *c;
        if (kind == 0) {
            pd_crout(n, a, y);
            c = y;
        } else {
#if defined(PD_METAL) || defined(PD_DGETRF2_ITER)
            int e = pd_dgetrf2_iter(n, n, a, n, ipiv);
#else
            int e = pd_dgetrf2(n, n, a, n, ipiv);
#endif
            if (e == 0) e = pd_dgetri(n, a, n, ipiv, y);
            if (e != 0) pd_or_i(info, 1);
            c = a;
        }
        PD_G real *qs[3] = {rxm, rym, rzm};
        for (int s = 0; s < 3; s++)
            for (int l = 0; l < n; l++)
                for (i64 t = 0; t < nt; t++) {
                    PD_G real *p = qs[s] + i * sq0 + j * sq1 + l * sq2 + t * sq3;
                    *p = CM(c, n, l, l) * *p;
                }
        for (i64 t = 0; t < nt; t++) {
            PD_G real *col_rm = rm + i * sq0 + j * sq1 + t * sq3;
            real srm[PD_LMAXC];
            for (int k = 0; k < n; k++) srm[k] = col_rm[k * sq2];
#ifdef PD_F32
            real zs[PD_LMAXC];
            for (int l = 0; l < n; l++) {
                real z = PD_R(0.0);
                for (int k = 0; k < n; k++) z = z + CM(c, n, l, k) * srm[k];
                zs[l] = z;
            }
            pd_column_deficit(n, srm, zs);
            for (int l = 0; l < n; l++) col_rm[l * sq2] = zs[l];
#else
            for (int l = 0; l < n; l++) {
                real z = PD_R(0.0);
                for (int k = 0; k < n; k++) z = z + CM(c, n, l, k) * srm[k];
                col_rm[l * sq2] = z;
            }
#endif
        }
    }
}

#endif

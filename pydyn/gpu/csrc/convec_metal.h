/* pd_convec on Metal with one SIMD group (32 lanes) a column, for lmax > 32 (convec_gen.py's
   register kernel takes lmax <= 32): the matrix and its inverse in threadgroup memory, every
   loop over independent elements spread over the lanes, every element's own operations in
   convec.cuh's order -- so its bits are pd_convec's (Crout, kind 0; the LAPACK kind stays on
   pd_convec).  Appended by metal.metal_source after the argument structs; the launcher
   dispatches it in place of pd_convec. */
#ifdef PD_METAL
#ifndef PD_CONVEC_CPT
#define PD_CONVEC_CPT 2 /* columns a threadgroup holds */
#endif
#define PD_CW ((PD_LMAXC + 1) * PD_LMAXC)
#define PD_SB simdgroup_barrier(mem_flags::mem_threadgroup)

/* pd_convection_matrix's rows (stride n) into FD (n+1 rows) and FU, lane p a column; then the
   caller's barrier.  amu, amd: n+1 each. */
PD_FN void pd_conv_build(int n, uint ln, PD_G const real *m, i64 sm, PD_G const real *entu, PD_G const real *detu,
                         PD_G const real *entd, PD_G const real *detd, i64 se, PD_G const real *dkg, i64 sd, int ctop,
                         int clfs, real dt, threadgroup real *A, threadgroup real *FU, threadgroup real *amu,
                         threadgroup real *amd)
{
    const int L = 32;
#define FU_(k, p) FU[(i64)(k) * n + (p)]
#define FD_(k, p) A[(i64)(k) * n + (p)]
    for (int t = ln; t < (n + 1) * n; t += L) FU[t] = A[t] = PD_R(0.0);
    PD_SB;
    {
        real amu_prev = PD_R(0.0); /* every lane runs the scalar recurrences; lane 0 keeps them */
        if (ln == 0) amu[0] = PD_R(0.0);
        for (int k = 1; k <= n; k++) { /* updraft: FU(k, p) needs FU(k-1, p), the same lane's */
            int act = ctop >= k;
            real eu = entu[(k - 1) * se], du = detu[(k - 1) * se];
            real av = amu_prev + eu - du;
            int up = av > PD_R(0.0);
            real zxi = up ? np_maximum(PD_R(0.0), PD_R(1.0) - du / (amu_prev + eu)) : PD_R(0.0);
            amu_prev = act && up ? av : PD_R(0.0);
            if (ln == 0) amu[k] = amu_prev;
            for (int p = ln; p < k - 1; p += L) FU_(k, p) = act ? FU_(k - 1, p) * zxi : PD_R(0.0);
            if ((k - 1) % L == (int)ln) FU_(k, k - 1) = act ? eu / m[(k - 1) * sm] * zxi : PD_R(0.0);
        }
        real amd_k = PD_R(0.0);
        if (ln == 0) amd[n] = PD_R(0.0);
        for (int k = n; k > 1; k--) { /* downdraft: FD(k-1, p) needs FD(k, p), the same lane's */
            int act = clfs >= k;
            real ed = entd[(k - 1) * se], dd = detd[(k - 1) * se];
            real av = amd_k - ed + dd;
            int dn = av < PD_R(0.0);
            real zxi = dn ? np_maximum(PD_R(0.0), PD_R(1.0) + dd / (amd_k - ed)) : PD_R(0.0);
            amd_k = act && dn ? av : PD_R(0.0);
            if (ln == 0) amd[k - 1] = amd_k;
            for (int p = k + ln; p < n; p += L) FD_(k - 1, p) = act ? FD_(k, p) * zxi : FD_(k - 1, p);
            if ((k - 1) % L == (int)ln) FD_(k - 1, k - 1) = act ? -ed / m[(k - 1) * sm] * zxi : FD_(k - 1, k - 1);
        }
    }
    PD_SB;
    for (int k = 1; k < n; k++) { /* combine, compensating subsidence, diffusion: lane p */
        real mk = m[k * sm], mk1 = m[(k - 1) * sm], dg = dkg[(k - 1) * sd];
        for (int p = ln; p < n; p += L) {
            FD_(k, p) = FU_(k, p) + FD_(k, p);
            if (p == k) FD_(k, k) = FD_(k, k) - amu[k] / mk;
            if (p == k - 1) {
                FD_(k, k - 1) = FD_(k, k - 1) - amd[k] / mk1;
                FD_(k, k - 1) = FD_(k, k - 1) + dg / mk1;
            }
            if (p == k) FD_(k, k) = FD_(k, k) - dg / mk;
        }
    }
    for (int k = 1; k <= n; k++) /* conv1: row k-1 from rows k-1 and k, lane p */
        for (int p = ln; p < n; p += L) {
            FD_(k - 1, p) = -dt * (FD_(k - 1, p) - FD_(k, p));
            if (p == k - 1) FD_(k - 1, k - 1) = FD_(k - 1, k - 1) + PD_R(1.0);
        }
#undef FU_
#undef FD_
}

kernel void pd_convec_simd(constant pd_convec_args &a [[buffer(0)]], uint gid [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]])
{
    threadgroup real Abuf[PD_CONVEC_CPT][PD_CW], Ybuf[PD_CONVEC_CPT][PD_CW];
    threadgroup real amu_b[PD_CONVEC_CPT][PD_LMAXC + 1], amd_b[PD_CONVEC_CPT][PD_LMAXC + 1];
    threadgroup real srm_b[PD_CONVEC_CPT][PD_LMAXC];
    i64 q = (i64)gid * PD_CONVEC_CPT + sg;
    if (q >= a.ncol) return; /* uniform over the SIMD group; only SIMD-group barriers below */
    threadgroup real *A = Abuf[sg], *Y = Ybuf[sg], *FU = Ybuf[sg]; /* fu is dead before y is written */
    threadgroup real *amu = amu_b[sg], *amd = amd_b[sg], *srm = srm_b[sg];
    const int L = 32;
    i64 col = a.cols[q], i = col / a.jm, j = col % a.jm;
    int n = a.lmax;
    PD_G const real *m = a.m + i * a.sm0 + j * a.sm1;
    i64 sm = a.sm2, oe = i * a.se0 + j * a.se1, se = a.se2, sd = a.sd2;
    PD_G const real *entu = a.entu + oe, *detu = a.detu + oe, *entd = a.entd + oe, *detd = a.detd + oe;
    PD_G const real *dkg = a.dkg + i * a.sd0 + j * a.sd1;
    int ctop = a.ctop[i * a.sc0 + j * a.sc1], clfs = a.clfs[i * a.sc0 + j * a.sc1];
    real dt = a.dt;

    pd_conv_build(n, ln, m, sm, entu, detu, entd, detd, se, dkg, sd, ctop, clfs, dt, A, FU, amu, amd);
    PD_SB;
    for (int r = 0; r < n; r++) /* rows of stride n -> column-major; pairs are disjoint */
        for (int c = r + 1 + ln; c < n; c += L) {
            real t = A[(i64)r * n + c];
            A[(i64)r * n + c] = A[(i64)c * n + r];
            A[(i64)c * n + r] = t;
        }
    PD_SB;

    /* ---- pd_crout: lu in A, the inverse into Y (column-major, CM) */
    for (int jj = 0; jj < n; jj++) {
        for (int k = 0; k < jj; k++) {
            for (int r = k + 1 + ln; r < n; r += L) CM(A, n, r, jj) = CM(A, n, r, jj) - CM(A, n, r, k) * CM(A, n, k, jj);
            PD_SB; /* A(k+1, jj) is final for the next k */
        }
        for (int r = jj + 1 + ln; r < n; r += L) CM(A, n, r, jj) = CM(A, n, r, jj) / CM(A, n, jj, jj);
        PD_SB;
    }
    for (int t = ln; t < n * n; t += L) Y[t] = (t % n) == (t / n) ? PD_R(1.0) : PD_R(0.0);
    PD_SB;
    for (int c = ln; c < n; c += L) { /* column c of y is lane c's alone */
        for (int ii = 0; ii < n; ii++)
            for (int jj = 0; jj < ii; jj++) CM(Y, n, ii, c) = CM(Y, n, ii, c) - CM(A, n, ii, jj) * CM(Y, n, jj, c);
        for (int ii = n - 1; ii >= 0; ii--) {
            for (int jj = ii + 1; jj < n; jj++) CM(Y, n, ii, c) = CM(Y, n, ii, c) - CM(A, n, ii, jj) * CM(Y, n, jj, c);
            CM(Y, n, ii, c) = CM(Y, n, ii, c) / CM(A, n, ii, ii);
        }
    }
    PD_SB;

    /* ---- the update: slopes by the diagonal, rm(l) = sum_k C(l, k) rm(k) */
    PD_G real *qs[3] = {a.rxm, a.rym, a.rzm};
    i64 nt = a.nt, base = i * a.sq0 + j * a.sq1;
    for (int s = 0; s < 3; s++)
        for (int u = ln; u < n * nt; u += L) {
            int l = u / nt;
            i64 t = u % nt;
            PD_G real *p = qs[s] + base + l * a.sq2 + t * a.sq3;
            *p = CM(Y, n, l, l) * *p;
        }
    for (i64 t = 0; t < nt; t++) {
        PD_G real *col_rm = a.rm + base + t * a.sq3;
        for (int k = ln; k < n; k += L) srm[k] = col_rm[k * a.sq2];
        PD_SB;
        for (int l = ln; l < n; l += L) {
            real z = PD_R(0.0);
            for (int k = 0; k < n; k++) z = z + CM(Y, n, l, k) * srm[k];
            amu[l] = z;
        }
        PD_SB;
        real sb = PD_R(0.0), cb = PD_R(0.0), sa = PD_R(0.0), ca = PD_R(0.0); /* convec.cuh pd_column_deficit */
        for (int k = 0; k < n; k++) {
            real t = sb + srm[k], bp = t - sb;
            cb = cb + ((sb - (t - bp)) + (srm[k] - bp));
            sb = t;
        }
        int lx = 0;
        for (int l = 0; l < n; l++) {
            real t = sa + amu[l], bp = t - sa;
            ca = ca + ((sa - (t - bp)) + (amu[l] - bp));
            sa = t;
            if (l > 0 && amu[l] > amu[lx]) lx = l;
        }
        real d = (sb - sa) + (cb - ca);
        for (int l = ln; l < n; l += L) col_rm[l * a.sq2] = l == lx ? amu[l] + d : amu[l];
        PD_SB;
    }
}

#undef PD_SB
#endif

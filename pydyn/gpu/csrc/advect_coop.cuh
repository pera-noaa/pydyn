/* advect.cuh's line kernels with a SIMD group (Metal) / warp (CUDA) of 32 lanes a line: each
   sub-step computes the line's faces from its old cells, lanes striding over them, into group
   memory, one tracer at a time; then the cells, from their two faces; the mass last.  Every
   element's operations are the serial kernel's, in its order -- a face is a function of the old
   cells alone and a cell of its own two faces -- so the bits are advect.cuh's (the host builds
   keep that one).  Bodies only: pydyn/gpu/coop.py writes each dialect's kernel around them
   (`<kernel>_coop`, the serial kernel's arguments), which the launchers dispatch in its place.

   PD_SH qualifies group memory, PD_LINE_SYNC is the group's barrier (memory included),
   PD_COOP_FACES (a define per launch) bounds a line's faces. */
#ifndef PD_ADVECT_COOP
#define PD_ADVECT_COOP
#include "advect.cuh"
#include "child.cuh"

#define PD_LANES 32

/* dynamw's sub-steps of column (i, j) = line: as pd_z_steps */
PD_FN void pd_z_steps_line(i64 line, unsigned ln, PD_SH real *sh, i64 im, i64 jm, i64 lm, i64 nt, PD_G real *m, i64 sm0,
                           i64 sm1, i64 sm2, PD_G real *rm, PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 sq0,
                           i64 sq1, i64 sq2, i64 sq3, PD_G const real *c, i64 sc0, i64 sc1, i64 sc2,
                           PD_G const int *nadv, int limits, int prelimit)
{
    PD_SH real *F = sh, *PF = sh + PD_COOP_FACES, *FT = sh + 2 * PD_COOP_FACES, *FW = sh + 3 * PD_COOP_FACES;
    int n_advz = nadv[0];
    i64 i = line / jm, j = line % jm;
    PD_G real *mc = m + i * sm0 + j * sm1;
    PD_G const real *cc = c + i * sc0 + j * sc1;
    i64 off = i * sq0 + j * sq1;
    PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
    (void)im;
    if (prelimit)
        for (i64 l = ln; l < lm; l += PD_LANES)
            for (i64 t = 0; t < nt; t++) qz[l * sq2 + t * sq3] = pd_limit(qz[l * sq2 + t * sq3], qm[l * sq2 + t * sq3]);
    PD_LINE_SYNC;
    real dn = (real)n_advz;
    for (int s = 0; s < n_advz; s++) {
        for (i64 t = 0; t < nt; t++) {
            for (i64 k = ln; k <= lm; k += PD_LANES) { /* face k between cells k-1 and k; the outer two zero */
                real f = PD_R(0.0), pf = PD_R(0.0), ft = PD_R(0.0), fw = PD_R(0.0);
                if (k > 0 && k < lm) {
                    real cf = cc[k * sc2] / dn;
                    i64 u = cf >= PD_R(0.0) ? k - 1 : k;
                    real gamma = cf / mc[u * sm2];
                    i64 o = u * sq2 + t * sq3;
                    pd_face(cf, gamma, qm[o], qz[o], qx[o], qy[o], &f, &pf, &ft, &fw);
                }
                F[k] = f, PF[k] = pf, FT[k] = ft, FW[k] = fw;
            }
            PD_LINE_SYNC;
            for (i64 l = ln; l < lm; l += PD_LANES) {
                real c_lo = cc[l * sc2] / dn, c_hi = cc[(l + 1) * sc2] / dn;
                real m_old = mc[l * sm2];
                real m_new = m_old + c_lo - c_hi;
                i64 o = l * sq2 + t * sq3;
                real rmn = qm[o] + F[l] - F[l + 1]; /* advectz: rm + f(l-1) - f(l) */
                real rz = pd_slope_update(qz[o], PF[l], PF[l + 1], c_lo, c_hi, rmn, F[l], F[l + 1], m_old, m_new);
                if (limits) rz = pd_limit(rz, rmn);
                qm[o] = rmn;
                qz[o] = rz;
                qx[o] = qx[o] + FT[l] - FT[l + 1];
                qy[o] = qy[o] + FW[l] - FW[l + 1];
            }
            PD_LINE_SYNC;
        }
        for (i64 l = ln; l < lm; l += PD_LANES) {
            real c_lo = cc[l * sc2] / dn, c_hi = cc[(l + 1) * sc2] / dn;
            mc[l * sm2] = mc[l * sm2] + c_lo - c_hi;
        }
        PD_LINE_SYNC;
    }
}

/* ------------------------------------------------------------------ x: dynamu (periodic rows) */
/* pd_x_trial's loop body for face k alone: alive, and its largest |Courant| into *cmax */
PD_FN int pd_x_trial_face(PD_G const real *m, i64 sm0, PD_G const real *a, i64 sa0, i64 n, i64 nl, i64 k,
                          PD_L real *cmax)
{
    real ak = a[k * sa0];
    i64 u = ak >= PD_R(0.0) ? (k == 0 ? n - 1 : k - 1) : (k == n ? 0 : k);
    real alo = a[u * sa0], ahi = a[(u + 1) * sa0];
    real mx = m[u * sm0], c_max = PD_R(0.0);
    for (i64 s = 0; s < nl; s++) {
        real c = pd_fabs(ak / mx);
        if (!(c < PD_R(1.0))) return 0;
        if (c > c_max) c_max = c;
        mx = mx + alo - ahi;
    }
    *cmax = c_max;
    return 1;
}

/* dynamu on row (j, l) = line: as pd_x_rows (its search, a trial per face, the row's verdict
   the lanes' AND and its amax their largest) */
PD_FN void pd_x_rows_line(i64 line, unsigned ln, PD_SH real *sh, i64 jm, i64 lm, i64 nt, PD_G const int *nrow,
                          PD_G real *m, i64 sm0, i64 sm1, i64 sm2, PD_G real *rm, PD_G real *rxm, PD_G real *rym,
                          PD_G real *rzm, i64 sq0, i64 sq1, i64 sq2, i64 sq3, PD_G real *a, i64 sa0, i64 sa1, i64 sa2,
                          int limits, int nmax, PD_G int *nloop_out, PD_G real *amax_out, PD_G int *fail)
{
    PD_SH real *F = sh, *PF = sh + PD_COOP_FACES, *FT = sh + 2 * PD_COOP_FACES, *FW = sh + 3 * PD_COOP_FACES;
    i64 j = line / lm, l = line % lm, n = nrow[j];
    PD_G real *mr = m + j * sm1 + l * sm2, *ar = a + j * sa1 + l * sa2;
    i64 off = j * sq1 + l * sq2;
    PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
    (void)jm;
    if (limits)
        for (i64 i = ln; i < n; i += PD_LANES)
            for (i64 t = 0; t < nt; t++) qx[i * sq0 + t * sq3] = pd_limit(qx[i * sq0 + t * sq3], qm[i * sq0 + t * sq3]);
    PD_LINE_SYNC;
    int nl = 1;
    real amax = PD_R(0.0);
    for (;;) {
        int ok = 1;
        real c_max = PD_R(0.0);
        for (i64 k = ln; k <= n; k += PD_LANES) {
            real c = PD_R(0.0);
            if (!pd_x_trial_face(mr, sm0, ar, sa0, n, nl, k, &c)) ok = 0;
            else if (c > c_max) c_max = c;
        }
        if (PD_LINE_ALL(ok)) {
            amax = PD_LINE_MAX(c_max);
            break;
        }
        nl += 1;
        for (i64 k = ln; k <= n; k += PD_LANES) ar[k * sa0] = ar[k * sa0] * (real)(nl - 1) / (real)nl;
        PD_LINE_SYNC;
        if (nl > nmax) {
            if (ln == 0) {
                pd_or_i(fail, 1);
                nloop_out[line] = nl;
            }
            return;
        }
    }
    for (int s = 0; s < nl; s++) {
        for (i64 t = 0; t < nt; t++) {
            for (i64 k = ln + 1; k <= n; k += PD_LANES) { /* face k of cells k-1, k; face n (the seam) is face 0 */
                real ak = ar[k * sa0];
                i64 lo = k - 1, hi = k == n ? 0 : k;
                i64 u = ak >= PD_R(0.0) ? lo : hi;
                real alpha = ak * (PD_R(1.0) / mr[u * sm0]);
                i64 o = u * sq0 + t * sq3;
                real f, pf, ft, fw;
                pd_face(ak, alpha, qm[o], qx[o], qy[o], qz[o], &f, &pf, &ft, &fw);
                F[k] = f, PF[k] = pf, FT[k] = ft, FW[k] = fw;
                if (k == n) F[0] = f, PF[0] = pf, FT[0] = ft, FW[0] = fw;
            }
            PD_LINE_SYNC;
            for (i64 i = ln; i < n; i += PD_LANES) {
                real a_lo = ar[i * sa0], a_hi = ar[(i + 1) * sa0];
                real m_old = mr[i * sm0];
                real m_new = m_old + a_lo - a_hi;
                i64 o = i * sq0 + t * sq3;
                real rmn = qm[o] - (F[i + 1] - F[i]); /* advectx: rm + (f(i-1) - f(i)) */
                real rx = pd_slope_update(qx[o], PF[i], PF[i + 1], a_lo, a_hi, rmn, F[i], F[i + 1], m_old, m_new);
                if (limits) rx = pd_limit(rx, rmn);
                qm[o] = rmn;
                qx[o] = rx;
                qy[o] = qy[o] - (FT[i + 1] - FT[i]);
                qz[o] = qz[o] - (FW[i + 1] - FW[i]);
            }
            PD_LINE_SYNC;
        }
        for (i64 i = ln; i < n; i += PD_LANES) mr[i * sm0] = mr[i * sm0] + ar[i * sa0] - ar[(i + 1) * sa0];
        PD_LINE_SYNC;
    }
    for (i64 k = ln; k <= n; k += PD_LANES) ar[k * sa0] = ar[k * sa0] * (real)nl; /* "restore old am" */
    if (ln == 0) {
        nloop_out[line] = nl;
        amax_out[line] = amax;
    }
}

/* ------------------------------------------------------------------ y: dynamv (per layer nloop) */
/* dynamv's sub-steps of column (i, l) = line: as pd_y_steps (the largest |beta| per lane, every
   sub-step's faces, into bmax[l]) */
PD_FN void pd_y_steps_line(i64 line, unsigned ln, PD_SH real *sh, i64 im, i64 jm, i64 lm, i64 nt, PD_G real *m, i64 sm0,
                           i64 sm1, i64 sm2, PD_G real *rm, PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 sq0,
                           i64 sq1, i64 sq2, i64 sq3, PD_G const real *b, i64 sb0, i64 sb1, i64 sb2,
                           PD_G const int *nloop, int limits, PD_G real *bmax)
{
    PD_SH real *F = sh, *PF = sh + PD_COOP_FACES, *FT = sh + 2 * PD_COOP_FACES, *FW = sh + 3 * PD_COOP_FACES;
    i64 i = line / lm, l = line % lm;
    PD_G real *mc = m + i * sm0 + l * sm2;
    PD_G const real *bc = b + i * sb0 + l * sb2;
    i64 off = i * sq0 + l * sq2;
    PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
    (void)im;
    if (limits)
        for (i64 jj = ln; jj < jm; jj += PD_LANES)
            for (i64 t = 0; t < nt; t++) qy[jj * sq1 + t * sq3] = pd_limit(qy[jj * sq1 + t * sq3], qm[jj * sq1 + t * sq3]);
    PD_LINE_SYNC;
    real beta_max = PD_R(0.0);
    for (int s = 0; s < nloop[l]; s++) {
        for (i64 k = ln + 1; k < jm; k += PD_LANES) { /* pd_y_face's |beta| */
            real bf = bc[k * sb1];
            i64 u = bf >= PD_R(0.0) ? k - 1 : k;
            real bb = pd_fabs(bf / mc[u * sm1]);
            if (bb > beta_max) beta_max = bb;
        }
        for (i64 t = 0; t < nt; t++) {
            for (i64 k = ln; k <= jm; k += PD_LANES) { /* face k of cells k-1, k; the outer two zero */
                real f = PD_R(0.0), pf = PD_R(0.0), ft = PD_R(0.0), fw = PD_R(0.0);
                if (k > 0 && k < jm) {
                    real bf = bc[k * sb1];
                    i64 u = bf >= PD_R(0.0) ? k - 1 : k;
                    real beta = bf / mc[u * sm1];
                    int pole = (k == 1 && bf >= PD_R(0.0)) || (k == jm - 1 && bf < PD_R(0.0));
                    i64 o = u * sq1 + t * sq3;
                    pd_face(bf, beta, qm[o], qy[o], qx[o], qz[o], &f, &pf, &ft, &fw);
                    if (pole) {
                        f = beta * qm[o];
                        pf = (-PD_R(3.0) * bf) * f;
                        ft = PD_R(0.0);
                        fw = beta * qz[o];
                    }
                }
                F[k] = f, PF[k] = pf, FT[k] = ft, FW[k] = fw;
            }
            PD_LINE_SYNC;
            for (i64 jj = ln; jj < jm; jj += PD_LANES) {
                real b_lo = bc[jj * sb1], b_hi = bc[(jj + 1) * sb1];
                real m_old = mc[jj * sm1];
                real m_new = m_old + b_lo - b_hi;
                real inv_mnew = PD_R(1.0) / m_new;
                real flow_conv = b_lo - b_hi;
                real flow_mean = PD_R(3.0) * (b_lo + b_hi);
                real mass_x3 = PD_R(3.0) * m_old;
                int pole = jj == 0 || jj == jm - 1;
                i64 o = jj * sq1 + t * sq3;
                real rmn = qm[o] - (F[jj + 1] - F[jj]); /* advecty: rm + (f(j-1) - f(j)) */
                real ry = qy[o] + ((PF[jj] - PF[jj + 1]) - flow_conv * qy[o] + flow_mean * rmn -
                                   mass_x3 * (F[jj] + F[jj + 1])) * inv_mnew;
                if (limits) ry = pd_limit(ry, rmn);
                real rx = qx[o] - (FT[jj + 1] - FT[jj]);
                qm[o] = rmn;
                qz[o] = qz[o] - (FW[jj + 1] - FW[jj]);
                if (!pole) {
                    qy[o] = ry;
                    qx[o] = rx;
                }
            }
            PD_LINE_SYNC;
        }
        for (i64 jj = ln; jj < jm; jj += PD_LANES) mc[jj * sm1] = mc[jj * sm1] + bc[jj * sb1] - bc[(jj + 1) * sb1];
        PD_LINE_SYNC;
    }
    pd_max_nonneg(&bmax[l], beta_max);
}

/* ------------------------------------------------------------------ a zoom child's held-ends lines */
/* pd_held_trial's loop body for face k alone */
PD_FN int pd_held_trial_face(PD_G const real *m, i64 sm, PD_G const real *f, i64 sf, i64 n, int nl, i64 k)
{
    real fk = pd_scaled(f[k * sf], nl, 0);
    i64 u = fk >= PD_R(0.0) ? k : k + 1;
    int inner = u > 0 && u < n - 1;
    real flo = inner ? pd_scaled(f[(u - 1) * sf], nl, 0) : PD_R(0.0);
    real fhi = inner ? pd_scaled(f[u * sf], nl, 0) : PD_R(0.0);
    real mx = m[u * sm];
    for (int s = 0; s < nl; s++) {
        if (!(pd_fabs(fk / mx) < PD_R(1.0))) return 0;
        if (inner) mx = mx + flo - fhi;
    }
    return 1;
}

/* _rows_child on row (j, l) = line: as pd_x_child_rows */
PD_FN void pd_x_child_rows_line(i64 line, unsigned ln, PD_SH real *sh, i64 n, i64 J, i64 L, i64 nt, PD_G real *m,
                                i64 sm0, i64 sm1, i64 sm2, PD_G real *rm, PD_G real *rxm, PD_G real *rym,
                                PD_G real *rzm, i64 sq0, i64 sq1, i64 sq2, i64 sq3, PD_G real *a, i64 sa0, i64 sa1,
                                i64 sa2, int limits, int nmax, PD_G int *nloop_out, PD_G real *amax_out,
                                PD_G int *fail)
{
    PD_SH real *F = sh, *PF = sh + PD_COOP_FACES, *FT = sh + 2 * PD_COOP_FACES, *FW = sh + 3 * PD_COOP_FACES;
    i64 j = line / L, l = line % L;
    PD_G real *mr = m + j * sm1 + l * sm2, *ar = a + j * sa1 + l * sa2;
    i64 off = j * sq1 + l * sq2;
    PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
    (void)J;
    int nl = 1;
    for (;;) {
        int ok = 1;
        for (i64 k = ln; k < n - 1; k += PD_LANES)
            if (!pd_held_trial_face(mr, sm0, ar, sa0, n, nl, k)) ok = 0;
        if (PD_LINE_ALL(ok)) break;
        nl += 1;
        for (i64 k = ln; k < n - 1; k += PD_LANES) ar[k * sa0] = ar[k * sa0] * (real)(nl - 1) / (real)nl;
        PD_LINE_SYNC;
        if (nl > nmax) {
            if (ln == 0) {
                pd_or_i(fail, 1);
                nloop_out[line] = nl;
            }
            return;
        }
    }
    {
        real c_max = PD_R(0.0); /* the accepted trial's largest |Courant| (subcycle_rows) */
        for (i64 k = ln; k < n - 1; k += PD_LANES) {
            real fk = ar[k * sa0];
            i64 u = fk >= PD_R(0.0) ? k : k + 1;
            int inner = u > 0 && u < n - 1;
            real mx = mr[u * sm0];
            for (int s = 0; s < nl; s++) {
                real c = pd_fabs(fk / mx);
                if (c > c_max) c_max = c;
                if (inner) mx = mx + ar[(u - 1) * sa0] - ar[u * sa0];
            }
        }
        real amax = PD_LINE_MAX(c_max);
        if (ln == 0) amax_out[line] = amax;
    }
    for (int s = 0; s < nl; s++) {
        for (i64 t = 0; t < nt; t++) {
            for (i64 k = ln; k < n - 1; k += PD_LANES) { /* face k of cells k, k+1 */
                real ak = ar[k * sa0];
                i64 u = ak >= PD_R(0.0) ? k : k + 1;
                real alpha = ak * (PD_R(1.0) / mr[u * sm0]);
                i64 o = u * sq0 + t * sq3;
                real f, pf, ft, fw;
                pd_face(ak, alpha, qm[o], qx[o], qy[o], qz[o], &f, &pf, &ft, &fw);
                F[k] = f, PF[k] = pf, FT[k] = ft, FW[k] = fw;
            }
            PD_LINE_SYNC;
            for (i64 i = 1 + ln; i < n - 1; i += PD_LANES) {
                real a_lo = ar[(i - 1) * sa0], a_hi = ar[i * sa0];
                real m_old = mr[i * sm0];
                real m_new = m_old + a_lo - a_hi;
                i64 o = i * sq0 + t * sq3;
                real rmn = qm[o] - (F[i] - F[i - 1]); /* rm + (f(i-1) - f(i)) */
                real rx = pd_slope_update(qx[o], PF[i - 1], PF[i], a_lo, a_hi, rmn, F[i - 1], F[i], m_old, m_new);
                if (limits) rx = pd_limit(rx, rmn);
                qm[o] = rmn;
                qx[o] = rx;
                qy[o] = qy[o] - (FT[i] - FT[i - 1]);
                qz[o] = qz[o] - (FW[i] - FW[i - 1]);
            }
            PD_LINE_SYNC;
        }
        for (i64 i = 1 + ln; i < n - 1; i += PD_LANES) mr[i * sm0] = mr[i * sm0] + ar[(i - 1) * sa0] - ar[i * sa0];
        PD_LINE_SYNC;
    }
    for (i64 k = ln; k < n - 1; k += PD_LANES) ar[k * sa0] = ar[k * sa0] * (real)nl;
    if (ln == 0) nloop_out[line] = nl;
}

/* _dynamv_child's sub-steps of column (i, l) = line: as pd_y_child_steps */
PD_FN void pd_y_child_steps_line(i64 line, unsigned ln, PD_SH real *sh, i64 I, i64 n, i64 L, i64 nt, PD_G real *m,
                                 i64 sm0, i64 sm1, i64 sm2, PD_G real *rm, PD_G real *rxm, PD_G real *rym,
                                 PD_G real *rzm, i64 sq0, i64 sq1, i64 sq2, i64 sq3, PD_G const real *b, i64 sb0,
                                 i64 sb1, i64 sb2, PD_G const int *nloop, int limits, PD_G real *bmax)
{
    PD_SH real *F = sh, *PF = sh + PD_COOP_FACES, *FT = sh + 2 * PD_COOP_FACES, *FW = sh + 3 * PD_COOP_FACES;
    i64 i = line / L, l = line % L;
    PD_G real *mc = m + i * sm0 + l * sm2;
    PD_G const real *bc = b + i * sb0 + l * sb2;
    i64 off = i * sq0 + l * sq2;
    PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
    (void)I;
    real beta_max = PD_R(0.0);
    for (int s = 0; s < nloop[l]; s++) {
        for (i64 k = ln; k < n - 1; k += PD_LANES) { /* pd_yc_face's |beta| */
            real bf = bc[k * sb1];
            i64 u = bf >= PD_R(0.0) ? k : k + 1;
            real bb = pd_fabs(bf / mc[u * sm1]);
            if (bb > beta_max) beta_max = bb;
        }
        for (i64 t = 0; t < nt; t++) {
            for (i64 k = ln; k < n - 1; k += PD_LANES) { /* face k of cells k, k+1 */
                real bf = bc[k * sb1];
                i64 u = bf >= PD_R(0.0) ? k : k + 1;
                real beta = bf / mc[u * sm1];
                i64 o = u * sq1 + t * sq3;
                real f, pf, ft, fw;
                pd_face(bf, beta, qm[o], qy[o], qx[o], qz[o], &f, &pf, &ft, &fw);
                F[k] = f, PF[k] = pf, FT[k] = ft, FW[k] = fw;
            }
            PD_LINE_SYNC;
            for (i64 jj = 1 + ln; jj < n - 1; jj += PD_LANES) {
                real b_lo = bc[(jj - 1) * sb1], b_hi = bc[jj * sb1];
                real m_old = mc[jj * sm1];
                real m_new = m_old + b_lo - b_hi;
                real inv_mnew = PD_R(1.0) / m_new;
                real flow_conv = b_lo - b_hi;
                real flow_mean = PD_R(3.0) * (b_lo + b_hi);
                real mass_x3 = PD_R(3.0) * m_old;
                i64 o = jj * sq1 + t * sq3;
                real rmn = qm[o] + (F[jj - 1] - F[jj]);
                real f_sum = F[jj - 1] + F[jj];
                real ry = qy[o] + ((PF[jj - 1] - PF[jj]) - flow_conv * qy[o] + flow_mean * rmn - mass_x3 * f_sum) *
                                      inv_mnew;
                qm[o] = rmn;
                qy[o] = limits ? pd_limit(ry, rmn) : ry;
                qx[o] = qx[o] + (FT[jj - 1] - FT[jj]);
                qz[o] = qz[o] + (FW[jj - 1] - FW[jj]);
            }
            PD_LINE_SYNC;
        }
        for (i64 jj = 1 + ln; jj < n - 1; jj += PD_LANES) mc[jj * sm1] = mc[jj * sm1] + bc[(jj - 1) * sb1] - bc[jj * sb1];
        PD_LINE_SYNC;
    }
    pd_max_nonneg(&bmax[l], beta_max);
}

/* ------------------------------------------------------------------ the searches */
/* pd_closed_trial's loop body for face k alone */
PD_FN int pd_closed_trial_face(PD_G const real *m, i64 sm, PD_G const real *b, i64 sb, int nl, int mode, i64 k)
{
    real bk = pd_scaled(b[k * sb], nl, mode);
    i64 u = bk >= PD_R(0.0) ? k - 1 : k;
    real blo = pd_scaled(b[u * sb], nl, mode), bhi = pd_scaled(b[(u + 1) * sb], nl, mode);
    real mx = m[u * sm];
    for (int s = 0; s < nl; s++) {
        if (!(pd_fabs(bk / mx) < PD_R(1.0))) return 0;
        mx = mx + blo - bhi;
    }
    return 1;
}

/* pd_closed_search on one line, its faces over the lanes: the smallest nl >= cand all pass */
PD_FN int pd_closed_search_line(unsigned ln, PD_G const real *m, i64 sm, PD_G const real *b, i64 sb, i64 n, int cand,
                                int nmax, int mode)
{
    for (int nl = cand; nl <= nmax; nl++) {
        int ok = 1;
        for (i64 k = 1 + ln; k < n; k += PD_LANES)
            if (!pd_closed_trial_face(m, sm, b, sb, nl, mode, k)) ok = 0;
        if (PD_LINE_ALL(ok)) return nl;
    }
    return nmax + 1;
}

PD_FN void pd_y_search_line(i64 line, unsigned ln, PD_SH real *sh, i64 im, i64 jm, i64 lm, PD_G const real *m,
                            i64 sm0, i64 sm1, i64 sm2, PD_G const real *b, i64 sb0, i64 sb1, i64 sb2,
                            PD_G const int *cand, int nmax, PD_G int *nxt, PD_G int *bad)
{
    i64 i = line / lm, l = line % lm;
    PD_G const real *mc = m + i * sm0 + l * sm2, *bc = b + i * sb0 + l * sb2;
    (void)sh, (void)im;
    if (ln == 0 && (bc[0] != PD_R(0.0) || bc[jm * sb1] != PD_R(0.0))) pd_or_i(bad, 1);
    int found = pd_closed_search_line(ln, mc, sm1, bc, sb1, jm, cand[l], nmax, 0);
    if (ln == 0) pd_max_i(&nxt[l], found);
}

PD_FN void pd_z_search_line(i64 line, unsigned ln, PD_SH real *sh, i64 im, i64 jm, i64 lm, PD_G const real *m,
                            i64 sm0, i64 sm1, i64 sm2, PD_G const real *c, i64 sc0, i64 sc1, i64 sc2,
                            PD_G const int *cand, int nmax, PD_G int *nxt, PD_G int *bad)
{
    i64 i = line / jm, j = line % jm;
    PD_G const real *mc = m + i * sm0 + j * sm1, *cc = c + i * sc0 + j * sc1;
    (void)sh, (void)im;
    if (ln == 0 && (cc[0] != PD_R(0.0) || cc[lm * sc2] != PD_R(0.0))) pd_or_i(bad, 1);
    int found = pd_closed_search_line(ln, mc, sm2, cc, sc2, lm, cand[0], nmax, 1);
    if (ln == 0) pd_max_i(nxt, found);
}

/* pd_z_scope_search on column (i, j) = line: its trial's faces over the lanes */
PD_FN void pd_z_scope_search_line(i64 line, unsigned ln, PD_SH real *sh, i64 im, i64 jm, i64 lm, PD_G const real *m,
                                  i64 sm0, i64 sm1, i64 sm2, PD_G const real *c, i64 sc0, i64 sc1, i64 sc2, i64 si0,
                                  i64 si1, i64 sj0, i64 sj1, PD_G const int *cand, int nmax, PD_G int *nxt,
                                  PD_G int *bad)
{
    i64 i = line / jm, j = line % jm;
    PD_G const real *mc = m + i * sm0 + j * sm1, *cc = c + i * sc0 + j * sc1;
    (void)sh, (void)im;
    if (ln == 0 && (cc[0] != PD_R(0.0) || cc[lm * sc2] != PD_R(0.0))) pd_or_i(bad, 1);
    int scoped = i >= si0 && i < si1 && j >= sj0 && j < sj1;
    int found = nmax + 1;
    for (int nl = cand[0]; nl <= nmax && found > nmax; nl++) {
        int ok = 1;
        for (i64 k = 1 + ln; k < lm; k += PD_LANES) {
            real ck = cc[k * sc2] / (real)nl;
            i64 u = ck >= PD_R(0.0) ? k - 1 : k;
            real clo = cc[u * sc2] / (real)nl, chi = cc[(u + 1) * sc2] / (real)nl;
            real mx = mc[u * sm2];
            for (int s = 0; s < nl; s++) {
                if (!(pd_fabs(ck / mx) < PD_R(1.0))) {
                    ok = 0;
                    break;
                }
                if (scoped) mx = mx + clo - chi;
            }
        }
        if (PD_LINE_ALL(ok)) found = nl;
    }
    if (ln == 0) pd_max_i(nxt, found);
}

PD_FN void pd_held_search_line(i64 line, unsigned ln, PD_SH real *sh, i64 I, i64 n, i64 L, PD_G const real *m,
                               i64 sm0, i64 sm1, i64 sm2, PD_G const real *b, i64 sb0, i64 sb1, i64 sb2,
                               PD_G const int *cand, int nmax, PD_G int *nxt)
{
    i64 i = line / L, l = line % L;
    PD_G const real *mc = m + i * sm0 + l * sm2, *bc = b + i * sb0 + l * sb2;
    (void)sh, (void)I;
    int found = nmax + 1;
    for (int nl = cand[l]; nl <= nmax; nl++) {
        int ok = 1;
        for (i64 k = ln; k < n - 1; k += PD_LANES)
            if (!pd_held_trial_face(mc, sm1, bc, sb1, n, nl, k)) ok = 0;
        if (PD_LINE_ALL(ok)) {
            found = nl;
            break;
        }
    }
    if (ln == 0) pd_max_i(&nxt[l], found);
}

#endif

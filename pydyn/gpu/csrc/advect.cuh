/* The three slopes advection kernels of pydyn/kernels.py, one thread per line (an x row, a
   y column of a layer, a z column), each sub-step one streaming pass along the line: face k+1
   is computed from the old cells k and k+1, then cell k is updated from faces k and k+1, so
   nothing but the face values of one tracer set is held.  Every statement is the reference's,
   in its order (pd.h has the shared ones).

   Arrays are passed as a pointer to the view's first element and its element strides, so a
   kernel works on the same (halo'd, strided) views the reference does.  Tracer arrays rm, rxm,
   rym, rzm share one set of strides. */
#ifndef PD_ADVECT
#define PD_ADVECT
#include "pd.h"

#define IDX3(s0, s1, s2, i, j, k) ((i64)(i) * (s0) + (i64)(j) * (s1) + (i64)(k) * (s2))

struct pd_faces {
    real f[PD_NT_MAX], pf[PD_NT_MAX], ft[PD_NT_MAX], fw[PD_NT_MAX];
};

PD_FN void pd_faces_zero(PD_L pd_faces *q, i64 nt)
{
    for (i64 t = 0; t < nt; t++) q->f[t] = q->pf[t] = q->ft[t] = q->fw[t] = PD_R(0.0);
}

PD_FN void pd_faces_copy(PD_L pd_faces *dst, PD_L const pd_faces *src, i64 nt)
{
    for (i64 t = 0; t < nt; t++) {
        dst->f[t] = src->f[t];
        dst->pf[t] = src->pf[t];
        dst->ft[t] = src->ft[t];
        dst->fw[t] = src->fw[t];
    }
}

/* ------------------------------------------------------------------ x: dynamu (periodic rows) */
/* subcycle_rows' trial of one periodic row at nloop n: every face's upwind cell carried through
   n mass-only sub-steps; alive iff every |Courant| < 1, amax the largest of them. */
PD_FN int pd_x_trial(PD_G const real *m, i64 sm0, PD_G const real *a, i64 sa0, i64 n, i64 nl, PD_L real *amax)
{
    real mx_max = PD_R(0.0);
    for (i64 k = 0; k <= n; k++) {
        real ak = a[k * sa0];
        i64 u = ak >= PD_R(0.0) ? (k == 0 ? n - 1 : k - 1) : (k == n ? 0 : k);
        real alo = a[u * sa0], ahi = a[(u + 1) * sa0];
        real mx = m[u * sm0];
        for (i64 s = 0; s < nl; s++) {
            real c = pd_fabs(ak / mx);
            if (!(c < PD_R(1.0))) return 0;
            if (c > mx_max) mx_max = c;
            mx = mx + alo - ahi;
        }
    }
    *amax = mx_max;
    return 1;
}

/* face k of a periodic row: alpha = am * m_inv(iup) (the production dynamu spelling) */
PD_FN void pd_x_face(PD_G const real *m, i64 sm0, PD_G const real *rm, PD_G const real *rxm, PD_G const real *rym,
                     PD_G const real *rzm, i64 sq0, i64 sq3, i64 nt, real ak, i64 lo, i64 hi, PD_L pd_faces *q, i64 k)
{
    i64 u = ak >= PD_R(0.0) ? lo : hi;
    real alpha = ak * (PD_R(1.0) / m[u * sm0]);
    for (i64 t = 0; t < nt; t++) {
        i64 o = u * sq0 + t * sq3;
        pd_face(ak, alpha, rm[o], rxm[o], rym[o], rzm[o], &q->f[t], &q->pf[t], &q->ft[t], &q->fw[t]);
    }
    (void)k;
}

/* dynamu on every (j, l) row of a region at once: row t is latitude j = t / lm, level l = t % lm,
   with nrow[j] cells (imred(j) on a reduced row) and faces 0..nrow[j].  In place; the faces
   scaled by the search and multiplied back by nloop, as the reference. */
PD_GLOBAL void pd_x_rows(i64 jm, i64 lm, i64 nt, PD_G const int *nrow, PD_G real *m, i64 sm0, i64 sm1, i64 sm2,
                         PD_G real *rm, PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 sq0, i64 sq1, i64 sq2,
                         i64 sq3, PD_G real *a, i64 sa0, i64 sa1, i64 sa2, int limits, int nmax, PD_G int *nloop_out,
                         PD_G real *amax_out, PD_G int *fail)
{
    PD_LOOP(row, jm * lm)
    {
        i64 j = row / lm, l = row % lm, n = nrow[j];
        PD_G real *mr = m + j * sm1 + l * sm2, *ar = a + j * sa1 + l * sa2;
        i64 off = j * sq1 + l * sq2;
        PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
        if (limits)
            for (i64 i = 0; i < n; i++)
                for (i64 t = 0; t < nt; t++) qx[i * sq0 + t * sq3] = pd_limit(qx[i * sq0 + t * sq3], qm[i * sq0 + t * sq3]);

        int nl = 1, failed = 0;
        real amax = PD_R(0.0);
        while (!pd_x_trial(mr, sm0, ar, sa0, n, nl, &amax)) {
            nl += 1;
            for (i64 k = 0; k <= n; k++) ar[k * sa0] = ar[k * sa0] * (real)(nl - 1) / (real)nl;
            if (nl > nmax) {
                pd_or_i(fail, 1);
                nloop_out[row] = nl;
                failed = 1;
                break;
            }
        }
        if (failed) continue;

        for (int s = 0; s < nl; s++) {
            pd_faces fN, prev, nxt;
            pd_x_face(mr, sm0, qm, qx, qy, qz, sq0, sq3, nt, ar[n * sa0], n - 1, 0, &fN, n);
            pd_faces_copy(&prev, &fN, nt); /* periodic seam: face 0 is face n */
            real a_lo = ar[0];
            for (i64 i = 0; i < n; i++) {
                real a_hi = ar[(i + 1) * sa0];
                if (i < n - 1)
                    pd_x_face(mr, sm0, qm, qx, qy, qz, sq0, sq3, nt, a_hi, i, i + 1, &nxt, i + 1);
                else
                    pd_faces_copy(&nxt, &fN, nt);
                real m_old = mr[i * sm0];
                real m_new = m_old + a_lo - a_hi;
                for (i64 t = 0; t < nt; t++) {
                    i64 o = i * sq0 + t * sq3;
                    real rmn = qm[o] - (nxt.f[t] - prev.f[t]); /* advectx: rm + (f(i-1) - f(i)) */
                    real rx = pd_slope_update(qx[o], prev.pf[t], nxt.pf[t], a_lo, a_hi, rmn, prev.f[t], nxt.f[t],
                                                m_old, m_new);
                    if (limits) rx = pd_limit(rx, rmn);
                    qm[o] = rmn;
                    qx[o] = rx;
                    qy[o] = qy[o] - (nxt.ft[t] - prev.ft[t]);
                    qz[o] = qz[o] - (nxt.fw[t] - prev.fw[t]);
                }
                mr[i * sm0] = m_new;
                pd_faces_copy(&prev, &nxt, nt);
                a_lo = a_hi;
            }
        }
        for (i64 k = 0; k <= n; k++) ar[k * sa0] = ar[k * sa0] * (real)nl; /* "restore old am" */
        nloop_out[row] = nl;
        amax_out[row] = amax;
    }
}

/* ------------------------------------------------------------------ closed lines (y, z) */
/* The trial of one closed line of n cells (faces 0..n, the outer two zero) whose faces are
   b(k) = scale(b0(k)): every interior face's upwind cell carried through nl mass-only
   sub-steps.  `mode` 0 = progressive (dynamv: b scaled by (p-1)/p for p = 2..nl, cumulatively),
   1 = fresh (dynamw: b0 / nl). */
PD_FN real pd_scaled(real b0, int nl, int mode)
{
    if (mode == 1) return b0 / (real)nl;
    real b = b0;
    for (int p = 2; p <= nl; p++) b = b * (real)(p - 1) / (real)p;
    return b;
}

PD_FN int pd_closed_trial(PD_G const real *m, i64 sm, PD_G const real *b, i64 sb, i64 n, int nl, int mode)
{
    for (i64 k = 1; k < n; k++) {
        real bk = pd_scaled(b[k * sb], nl, mode);
        i64 u = bk >= PD_R(0.0) ? k - 1 : k;
        real blo = pd_scaled(b[u * sb], nl, mode), bhi = pd_scaled(b[(u + 1) * sb], nl, mode);
        real mx = m[u * sm];
        for (int s = 0; s < nl; s++) {
            if (!(pd_fabs(bk / mx) < PD_R(1.0))) return 0;
            mx = mx + blo - bhi;
        }
    }
    return 1;
}

/* The smallest nloop >= cand that line passes, nmax + 1 if none up to nmax. */
PD_FN int pd_closed_search(PD_G const real *m, i64 sm, PD_G const real *b, i64 sb, i64 n, int cand, int nmax, int mode)
{
    for (int nl = cand; nl <= nmax; nl++)
        if (pd_closed_trial(m, sm, b, sb, n, nl, mode)) return nl;
    return nmax + 1;
}

/* ------------------------------------------------------------------ y: dynamv (per layer nloop) */
/* One round of the per-layer search: column t = (i = t / lm, l = t % lm) of layer l reports
   the smallest nloop >= cand[l] it passes into nxt[l] (a max over the layer).  Iterated from
   cand = 1 until nxt == cand, this is the layer's sequential search exactly: no n below the
   fixpoint passes every column (the column that raised it fails there). */
PD_GLOBAL void pd_y_search(i64 im, i64 jm, i64 lm, PD_G const real *m, i64 sm0, i64 sm1, i64 sm2, PD_G const real *b,
                           i64 sb0, i64 sb1, i64 sb2, PD_G const int *cand, int nmax, PD_G int *nxt, PD_G int *bad)
{
    PD_LOOP(col, im * lm)
    {
        i64 i = col / lm, l = col % lm;
        PD_G const real *mc = m + i * sm0 + l * sm2, *bc = b + i * sb0 + l * sb2;
        if (bc[0] != PD_R(0.0) || bc[jm * sb1] != PD_R(0.0)) pd_or_i(bad, 1);
        pd_max_i(&nxt[l], pd_closed_search(mc, sm1, bc, sb1, jm, cand[l], nmax, 0));
    }
}

/* dynamv's search scaling of a whole layer, (p-1)/p for p = 2..nloop, on every face of the
   given (halo'd) bm, and its closing multiplication by nloop (restore = 1). */
PD_GLOBAL void pd_layer_scale(i64 n0, i64 n1, i64 nl_off, i64 nlev, PD_G real *b, i64 s0, i64 s1, i64 s2,
                              PD_G const int *nloop, int restore)
{
    PD_LOOP(t, n0 * n1 * nlev)
    {
        i64 l = t % nlev, r = t / nlev, j = r % n1, i = r / n1;
        int nl = nloop[l];
        PD_G real *p = b + IDX3(s0, s1, s2, i, j, l + nl_off);
        if (restore)
            *p = *p * (real)nl;
        else
            for (int q = 2; q <= nl; q++) *p = *p * (real)(q - 1) / (real)q;
    }
}

/* face k (interior, between cells k-1 and k) of a y column; the polar faces are slope-free when
   the flow leaves a pole cell (dynamv's faces 2 and jmr) */
PD_FN real pd_y_face(PD_G const real *m, i64 sm, PD_G const real *rm, PD_G const real *rxm, PD_G const real *rym,
                       PD_G const real *rzm, i64 sq, i64 sq3, i64 nt, PD_G const real *b, i64 sb, i64 jm, i64 k,
                       PD_L pd_faces *q)
{
    real bf = b[k * sb];
    i64 u = bf >= PD_R(0.0) ? k - 1 : k;
    real beta = bf / m[u * sm];
    int pole = (k == 1 && bf >= PD_R(0.0)) || (k == jm - 1 && bf < PD_R(0.0));
    for (i64 t = 0; t < nt; t++) {
        i64 o = u * sq + t * sq3;
        pd_face(bf, beta, rm[o], rym[o], rxm[o], rzm[o], &q->f[t], &q->pf[t], &q->ft[t], &q->fw[t]);
        if (pole) {
            q->f[t] = beta * rm[o];
            q->pf[t] = (-PD_R(3.0) * bf) * q->f[t];
            q->ft[t] = PD_R(0.0);
            q->fw[t] = beta * rzm[o];
        }
    }
    return pd_fabs(beta);
}

/* dynamv's sub-steps of column (i, l) with its layer's nloop: the limiter first (dynamv applies it
   before the search, which does not touch the tracers), then nloop passes; the pole cells' rxm,
   rym are left as they were.  bmax[l] receives the layer's largest |beta|. */
PD_GLOBAL void pd_y_steps(i64 im, i64 jm, i64 lm, i64 nt, PD_G real *m, i64 sm0, i64 sm1, i64 sm2, PD_G real *rm,
                          PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 sq0, i64 sq1, i64 sq2, i64 sq3,
                          PD_G const real *b, i64 sb0, i64 sb1, i64 sb2, PD_G const int *nloop, int limits, PD_G real *bmax)
{
    PD_LOOP(col, im * lm)
    {
        i64 i = col / lm, l = col % lm;
        PD_G real *mc = m + i * sm0 + l * sm2;
        PD_G const real *bc = b + i * sb0 + l * sb2;
        i64 off = i * sq0 + l * sq2;
        PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
        if (limits)
            for (i64 j = 0; j < jm; j++)
                for (i64 t = 0; t < nt; t++) qy[j * sq1 + t * sq3] = pd_limit(qy[j * sq1 + t * sq3], qm[j * sq1 + t * sq3]);
        real beta_max = PD_R(0.0);
        for (int s = 0; s < nloop[l]; s++) {
            pd_faces prev, nxt;
            pd_faces_zero(&prev, nt);
            for (i64 j = 0; j < jm; j++) {
                real b_lo = bc[j * sb1], b_hi = bc[(j + 1) * sb1];
                if (j + 1 < jm) {
                    real bb = pd_y_face(mc, sm1, qm, qx, qy, qz, sq1, sq3, nt, bc, sb1, jm, j + 1, &nxt);
                    if (bb > beta_max) beta_max = bb;
                } else {
                    pd_faces_zero(&nxt, nt);
                }
                real m_old = mc[j * sm1];
                real m_new = m_old + b_lo - b_hi;
                real inv_mnew = PD_R(1.0) / m_new;
                real flow_conv = b_lo - b_hi;
                real flow_mean = PD_R(3.0) * (b_lo + b_hi);
                real mass_x3 = PD_R(3.0) * m_old;
                int pole = j == 0 || j == jm - 1;
                for (i64 t = 0; t < nt; t++) {
                    i64 o = j * sq1 + t * sq3;
                    real rmn = qm[o] - (nxt.f[t] - prev.f[t]); /* advecty: rm + (f(j-1) - f(j)) */
                    real ry = qy[o] + ((prev.pf[t] - nxt.pf[t]) - flow_conv * qy[o] + flow_mean * rmn -
                                         mass_x3 * (prev.f[t] + nxt.f[t])) * inv_mnew;
                    if (limits) ry = pd_limit(ry, rmn);
                    real rx = qx[o] - (nxt.ft[t] - prev.ft[t]);
                    qm[o] = rmn;
                    qz[o] = qz[o] - (nxt.fw[t] - prev.fw[t]);
                    if (!pole) {
                        qy[o] = ry;
                        qx[o] = rx;
                    }
                }
                mc[j * sm1] = m_new;
                pd_faces_copy(&prev, &nxt, nt);
            }
        }
        pd_max_nonneg(&bmax[l], beta_max);
    }
}

/* ------------------------------------------------------------------ z: dynamw (one nloop) */
/* One round of the region-wide search: column t = (i = t / jm, j = t % jm). */
PD_GLOBAL void pd_z_search(i64 im, i64 jm, i64 lm, PD_G const real *m, i64 sm0, i64 sm1, i64 sm2, PD_G const real *c,
                           i64 sc0, i64 sc1, i64 sc2, PD_G const int *cand, int nmax, PD_G int *nxt, PD_G int *bad)
{
    PD_LOOP(col, im * jm)
    {
        i64 i = col / jm, j = col % jm;
        PD_G const real *mc = m + i * sm0 + j * sm1, *cc = c + i * sc0 + j * sc1;
        if (cc[0] != PD_R(0.0) || cc[lm * sc2] != PD_R(0.0)) pd_or_i(bad, 1);
        pd_max_i(nxt, pd_closed_search(mc, sm2, cc, sc2, lm, cand[0], nmax, 1));
    }
}

PD_FN void pd_z_face(PD_G const real *m, i64 sm, PD_G const real *rm, PD_G const real *rxm, PD_G const real *rym,
                     PD_G const real *rzm, i64 sq, i64 sq3, i64 nt, real cf, i64 k, PD_L pd_faces *q)
{
    i64 u = cf >= PD_R(0.0) ? k - 1 : k;
    real gamma = cf / m[u * sm];
    for (i64 t = 0; t < nt; t++) {
        i64 o = u * sq + t * sq3;
        pd_face(cf, gamma, rm[o], rzm[o], rxm[o], rym[o], &q->f[t], &q->pf[t], &q->ft[t], &q->fw[t]);
    }
}

/* advect_z_substeps of column (i, j): the limiter, then n sub-steps with cm / n (n read from the
   device: the search's result); cm itself is not written (the reference restores it exactly
   from a copy). */
PD_GLOBAL void pd_z_steps(i64 im, i64 jm, i64 lm, i64 nt, PD_G real *m, i64 sm0, i64 sm1, i64 sm2, PD_G real *rm,
                          PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 sq0, i64 sq1, i64 sq2, i64 sq3,
                          PD_G const real *c, i64 sc0, i64 sc1, i64 sc2, PD_G const int *nadv, int limits,
                          int prelimit)
{
    int n_advz = nadv[0]; /* the search's, on the device */
    PD_LOOP(col, im * jm)
    {
        i64 i = col / jm, j = col % jm;
        PD_G real *mc = m + i * sm0 + j * sm1;
        PD_G const real *cc = c + i * sc0 + j * sc1;
        i64 off = i * sq0 + j * sq1;
        PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
        if (prelimit)
            for (i64 l = 0; l < lm; l++)
                for (i64 t = 0; t < nt; t++) qz[l * sq2 + t * sq3] = pd_limit(qz[l * sq2 + t * sq3], qm[l * sq2 + t * sq3]);
        real dn = (real)n_advz;
        for (int s = 0; s < n_advz; s++) {
            pd_faces prev, nxt;
            pd_faces_zero(&prev, nt);
            real c_lo = cc[0] / dn;
            for (i64 l = 0; l < lm; l++) {
                real c_hi = cc[(l + 1) * sc2] / dn;
                if (l + 1 < lm)
                    pd_z_face(mc, sm2, qm, qx, qy, qz, sq2, sq3, nt, c_hi, l + 1, &nxt);
                else
                    pd_faces_zero(&nxt, nt);
                real m_old = mc[l * sm2];
                real m_new = m_old + c_lo - c_hi;
                for (i64 t = 0; t < nt; t++) {
                    i64 o = l * sq2 + t * sq3;
                    real rmn = qm[o] + prev.f[t] - nxt.f[t]; /* advectz: rm + f(l-1) - f(l) */
                    real rz = pd_slope_update(qz[o], prev.pf[t], nxt.pf[t], c_lo, c_hi, rmn, prev.f[t], nxt.f[t],
                                                m_old, m_new);
                    if (limits) rz = pd_limit(rz, rmn);
                    qm[o] = rmn;
                    qz[o] = rz;
                    qx[o] = qx[o] + prev.ft[t] - nxt.ft[t];
                    qy[o] = qy[o] + prev.fw[t] - nxt.fw[t];
                }
                mc[l * sm2] = m_new;
                pd_faces_copy(&prev, &nxt, nt);
                c_lo = c_hi;
            }
        }
    }
}

#endif

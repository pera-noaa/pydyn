/* A zoom child's advection (zoom/advect.py _rows_child, _dynamv_child, _dynamw_child) and the
   exchange's arithmetic (zoom/exchange.py mix_edges, update_parent).

   A child's x row and y column are lines of n cells with n-1 interior faces whose two end
   cells are held (they are the parent's, put there by put_x/yedges): face k lies between cells
   k and k+1, and an inner cell i is updated through faces i-1 and i. */
#ifndef PD_CHILD
#define PD_CHILD
#include "pd.h"
#include "advect.cuh"

/* the trial of a held-ends line at nloop nl: the faces scaled as pd_scaled (mode 0 progressive,
   1 fresh); an end cell's mass stays as it is */
PD_FN int pd_held_trial(PD_G const real *m, i64 sm, PD_G const real *f, i64 sf, i64 n, int nl, int mode)
{
    for (i64 k = 0; k < n - 1; k++) {
        real fk = pd_scaled(f[k * sf], nl, mode);
        i64 u = fk >= PD_R(0.0) ? k : k + 1;
        int inner = u > 0 && u < n - 1;
        real flo = inner ? pd_scaled(f[(u - 1) * sf], nl, mode) : PD_R(0.0);
        real fhi = inner ? pd_scaled(f[u * sf], nl, mode) : PD_R(0.0);
        real mx = m[u * sm];
        for (int s = 0; s < nl; s++) {
            if (!(pd_fabs(fk / mx) < PD_R(1.0))) return 0;
            if (inner) mx = mx + flo - fhi;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ x: a child's dynamu rows */
/* face k of a held-ends x row: alpha = am * m_inv(iup) */
PD_FN void pd_xc_face(PD_G const real *m, i64 sm0, PD_G const real *rm, PD_G const real *rxm, PD_G const real *rym,
                      PD_G const real *rzm, i64 sq0, i64 sq3, i64 nt, real ak, i64 k, PD_L pd_faces *q)
{
    pd_x_face(m, sm0, rm, rxm, rym, rzm, sq0, sq3, nt, ak, k, k + 1, q, k);
}

/* _rows_child on every (j, l) row of the scope at once: row t is j = t / nl_, l = t % nl_ of
   the views m (n, J, L), tracers (n, J, L, nt), a (n-1, J, L).  nt = 0 is Check_CFL's
   mass-only dynamum.  The faces are scaled by the search and multiplied back by nloop. */
PD_GLOBAL void pd_x_child_rows(i64 n, i64 J, i64 L, i64 nt, PD_G real *m, i64 sm0, i64 sm1, i64 sm2, PD_G real *rm,
                               PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 sq0, i64 sq1, i64 sq2, i64 sq3, PD_G real *a,
                               i64 sa0, i64 sa1, i64 sa2, int limits, int nmax, PD_G int *nloop_out, PD_G real *amax_out,
                               PD_G int *fail)
{
    PD_LOOP(row, J * L)
    {
        i64 j = row / L, l = row % L;
        PD_G real *mr = m + j * sm1 + l * sm2, *ar = a + j * sa1 + l * sa2;
        i64 off = j * sq1 + l * sq2;
        PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
        int nl = 1, failed = 0;
        while (!pd_held_trial(mr, sm0, ar, sa0, n, nl, 0)) {
            nl += 1;
            for (i64 k = 0; k < n - 1; k++) ar[k * sa0] = ar[k * sa0] * (real)(nl - 1) / (real)nl;
            if (nl > nmax) {
                pd_or_i(fail, 1);
                nloop_out[row] = nl;
                failed = 1;
                break;
            }
        }
        if (failed) continue;
        {
            real amax = PD_R(0.0); /* the accepted trial's largest |Courant| (subcycle_rows) */
            for (i64 k = 0; k < n - 1; k++) {
                real fk = ar[k * sa0];
                i64 u = fk >= PD_R(0.0) ? k : k + 1;
                int inner = u > 0 && u < n - 1;
                real mx = mr[u * sm0];
                for (int s = 0; s < nl; s++) {
                    real c = pd_fabs(fk / mx);
                    if (c > amax) amax = c;
                    if (inner) mx = mx + ar[(u - 1) * sa0] - ar[u * sa0];
                }
            }
            amax_out[row] = amax;
        }
        for (int s = 0; s < nl; s++) {
            pd_faces prev, nxt;
            pd_xc_face(mr, sm0, qm, qx, qy, qz, sq0, sq3, nt, ar[0], 0, &prev);
            for (i64 i = 1; i < n - 1; i++) {
                real a_lo = ar[(i - 1) * sa0], a_hi = ar[i * sa0];
                pd_xc_face(mr, sm0, qm, qx, qy, qz, sq0, sq3, nt, a_hi, i, &nxt);
                real m_old = mr[i * sm0];
                real m_new = m_old + a_lo - a_hi;
                for (i64 t = 0; t < nt; t++) {
                    i64 o = i * sq0 + t * sq3;
                    real rmn = qm[o] - (nxt.f[t] - prev.f[t]); /* rm + (f(i-1) - f(i)) */
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
            }
        }
        for (i64 k = 0; k < n - 1; k++) ar[k * sa0] = ar[k * sa0] * (real)nl;
        nloop_out[row] = nl;
    }
}

/* ------------------------------------------------------------------ y: a child's dynamv */
/* One round of the per-layer search over the scope's columns: column t = (i = t / L, l = t % L)
   of the views m (I, n, L), b (I, n-1, L). */
PD_GLOBAL void pd_held_search(i64 I, i64 n, i64 L, PD_G const real *m, i64 sm0, i64 sm1, i64 sm2, PD_G const real *b,
                              i64 sb0, i64 sb1, i64 sb2, PD_G const int *cand, int nmax, PD_G int *nxt)
{
    PD_LOOP(col, I * L)
    {
        i64 i = col / L, l = col % L;
        PD_G const real *mc = m + i * sm0 + l * sm2, *bc = b + i * sb0 + l * sb2;
        int found = nmax + 1;
        for (int nl = cand[l]; nl <= nmax; nl++)
            if (pd_held_trial(mc, sm1, bc, sb1, n, nl, 0)) {
                found = nl;
                break;
            }
        pd_max_i(&nxt[l], found);
    }
}

/* face k of a child's y column: beta = bm / m(upwind) */
PD_FN real pd_yc_face(PD_G const real *m, i64 sm, PD_G const real *rm, PD_G const real *rxm, PD_G const real *rym,
                        PD_G const real *rzm, i64 sq, i64 sq3, i64 nt, real bf, i64 k, PD_L pd_faces *q)
{
    i64 u = bf >= PD_R(0.0) ? k : k + 1;
    real beta = bf / m[u * sm];
    for (i64 t = 0; t < nt; t++) {
        i64 o = u * sq + t * sq3;
        pd_face(bf, beta, rm[o], rym[o], rxm[o], rzm[o], &q->f[t], &q->pf[t], &q->ft[t], &q->fw[t]);
    }
    return pd_fabs(beta);
}

/* _dynamv_child's sub-steps of column (i, l) of the views m (I, n, L), tracers (I, n, L, nt),
   b (I, n-1, L), its layer's nloop; the limiter is the caller's.  nt = 0 is dynamvm. */
PD_GLOBAL void pd_y_child_steps(i64 I, i64 n, i64 L, i64 nt, PD_G real *m, i64 sm0, i64 sm1, i64 sm2, PD_G real *rm,
                                PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 sq0, i64 sq1, i64 sq2, i64 sq3,
                                PD_G const real *b, i64 sb0, i64 sb1, i64 sb2, PD_G const int *nloop, int limits,
                                PD_G real *bmax)
{
    PD_LOOP(col, I * L)
    {
        i64 i = col / L, l = col % L;
        PD_G real *mc = m + i * sm0 + l * sm2;
        PD_G const real *bc = b + i * sb0 + l * sb2;
        i64 off = i * sq0 + l * sq2;
        PD_G real *qm = rm + off, *qx = rxm + off, *qy = rym + off, *qz = rzm + off;
        real beta_max = PD_R(0.0);
        for (int s = 0; s < nloop[l]; s++) {
            pd_faces prev, nxt;
            real bb = pd_yc_face(mc, sm1, qm, qx, qy, qz, sq1, sq3, nt, bc[0], 0, &prev);
            if (bb > beta_max) beta_max = bb;
            for (i64 j = 1; j < n - 1; j++) {
                real b_lo = bc[(j - 1) * sb1], b_hi = bc[j * sb1];
                bb = pd_yc_face(mc, sm1, qm, qx, qy, qz, sq1, sq3, nt, b_hi, j, &nxt);
                if (bb > beta_max) beta_max = bb;
                real m_old = mc[j * sm1];
                real m_new = m_old + b_lo - b_hi;
                real inv_mnew = PD_R(1.0) / m_new;
                real flow_conv = b_lo - b_hi;
                real flow_mean = PD_R(3.0) * (b_lo + b_hi);
                real mass_x3 = PD_R(3.0) * m_old;
                for (i64 t = 0; t < nt; t++) {
                    i64 o = j * sq1 + t * sq3;
                    real rmn = qm[o] + (prev.f[t] - nxt.f[t]);
                    real f_sum = prev.f[t] + nxt.f[t];
                    real ry = qy[o] + ((prev.pf[t] - nxt.pf[t]) - flow_conv * qy[o] + flow_mean * rmn - mass_x3 * f_sum) *
                                            inv_mnew;
                    qm[o] = rmn;
                    qy[o] = limits ? pd_limit(ry, rmn) : ry;
                    qx[o] = qx[o] + (prev.ft[t] - nxt.ft[t]);
                    qz[o] = qz[o] + (prev.fw[t] - nxt.fw[t]);
                }
                mc[j * sm1] = m_new;
                pd_faces_copy(&prev, &nxt, nt);
            }
        }
        pd_max_nonneg(&bmax[l], beta_max);
    }
}

/* ------------------------------------------------------------------ z: column_trial with a scope */
/* One round of dynamw's region-wide search, a column's mass advanced only inside the scope
   [si0, si1) x [sj0, sj1) (zoom.advect.column_trial). */
PD_GLOBAL void pd_z_scope_search(i64 im, i64 jm, i64 lm, PD_G const real *m, i64 sm0, i64 sm1, i64 sm2, PD_G const real *c,
                                 i64 sc0, i64 sc1, i64 sc2, i64 si0, i64 si1, i64 sj0, i64 sj1, PD_G const int *cand, int nmax,
                                 PD_G int *nxt, PD_G int *bad)
{
    PD_LOOP(col, im * jm)
    {
        i64 i = col / jm, j = col % jm;
        PD_G const real *mc = m + i * sm0 + j * sm1, *cc = c + i * sc0 + j * sc1;
        if (cc[0] != PD_R(0.0) || cc[lm * sc2] != PD_R(0.0)) pd_or_i(bad, 1);
        int scoped = i >= si0 && i < si1 && j >= sj0 && j < sj1;
        int found = nmax + 1;
        for (int nl = cand[0]; nl <= nmax && found > nmax; nl++) {
            int ok = 1;
            for (i64 k = 1; k < lm && ok; k++) {
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
            if (ok) found = nl;
        }
        pd_max_i(nxt, found);
    }
}

/* ------------------------------------------------------------------ mix_edges */
/* The means of the parent blocks the spread reads (nb of them, bp flat ip * jmp + jp) over the
   nq level slices given by address qa and element strides qs0, qs1, into tp (nq, imp, jmp):
   acc = 0 + sum_b sum_a (ring ? q / xyref : 0.0), j outer / i inner */
PD_GLOBAL void pd_ring_means(i64 nq, i64 nb, PD_G const int *bp, i64 imp, i64 jmp, i64 xr, i64 yr,
                             PD_G const unsigned char *ring, i64 jm_c, PD_G const i64a *qa, PD_G const i64a *qs0,
                             PD_G const i64a *qs1, PD_G real *tp)
{
    PD_LOOP(t, nq * nb)
    {
        i64 k = t / nb, c = bp[t % nb], ip = c / jmp, jp = c % jmp;
        PD_G const real *q = PD_PTR(qa[k]);
        i64 s0 = qs0[k], s1 = qs1[k];
        real acc = PD_R(0.0), xyref = (real)(xr * yr);
        for (i64 b = 0; b < yr; b++)
            for (i64 a = 0; a < xr; a++) {
                i64 i = ip * xr + a, j = jp * yr + b;
                acc = acc + (ring[i * jm_c + j] ? q[i * s0 + j * s1] / xyref : PD_R(0.0));
            }
        tp[k * imp * jmp + c] = acc;
    }
}

/* the spread over the ring cells (nr of them, rc flat i * jmr + j): each rewritten with its
   parent cell's mean, later strips winning as the reference's statements do (x walls, then the
   south, then the north) */
PD_GLOBAL void pd_ring_spread(i64 nq, i64 nr, PD_G const int *rc, i64 imr, i64 jmr, i64 xr, i64 yr, i64 imp,
                              i64 jmp, int xedges, int ysp, int ynp, PD_G const i64a *qa, PD_G const i64a *qs0,
                              PD_G const i64a *qs1, PD_G const real *tp)
{
    PD_LOOP(t, nq * nr)
    {
        i64 k = t / nr, c = rc[t % nr], i = c / jmr, j = c % jmr;
        PD_G const real *p = tp + k * imp * jmp;
        real v = PD_R(0.0);
        if (xedges && i < xr) v = p[0 * jmp + j / yr];
        if (xedges && i >= imr - xr) v = p[(imp - 1) * jmp + j / yr];
        if (ysp && j < yr) v = p[(i / xr) * jmp + 0];
        if (ynp && j >= jmr - yr) v = p[(i / xr) * jmp + jmp - 1];
        PD_PTR(qa[k])[i * qs0[k] + j * qs1[k]] = v;
    }
}

/* ------------------------------------------------------------------ the limiter on a box */
/* s = kernels._limit(s, rm) over (n0, n1, n2, n3) views */
PD_GLOBAL void pd_limit_box(i64 n0, i64 n1, i64 n2, i64 n3, PD_G real *s, i64 s0, i64 s1, i64 s2, i64 s3, PD_G const real *rm,
                            i64 r0, i64 r1, i64 r2, i64 r3)
{
    PD_LOOP(t, n0 * n1 * n2 * n3)
    {
        i64 d = t % n3, r = t / n3, c = r % n2;
        r /= n2;
        i64 b = r % n1, a = r / n1;
        PD_G real *p = s + a * s0 + b * s1 + c * s2 + d * s3;
        *p = pd_limit(*p, rm[a * r0 + b * r1 + c * r2 + d * r3]);
    }
}

/* ------------------------------------------------------------------ update_parent */
/* block sums of the child's (im, jm, L) view into the parent's (imp, jmp, L) view, i outer / j
   inner, from zero */
PD_GLOBAL void pd_block_sums(i64 imp, i64 jmp, i64 L, i64 xr, i64 yr, PD_G const real *q, i64 q0, i64 q1, i64 q2,
                             PD_G real *p, i64 p0, i64 p1, i64 p2)
{
    PD_LOOP(t, imp * jmp * L)
    {
        i64 l = t % L, r = t / L, jp = r % jmp, ip = r / jmp;
        real acc = PD_R(0.0);
        for (i64 a = 0; a < xr; a++)
            for (i64 b = 0; b < yr; b++) acc = acc + q[(ip * xr + a) * q0 + (jp * yr + b) * q1 + l * q2];
        p[ip * p0 + jp * p1 + l * p2] = acc;
    }
}
#endif

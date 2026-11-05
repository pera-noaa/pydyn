/* dynam0 (kernels.fluxes_from_meteo), m2phlb1, do_add_2d and the chemistry's decay applied
   (zoom.physics), one thread per column or per cell. */
#ifndef PD_MASS
#define PD_MASS
#include "pd.h"

/* dst = dt * src over an (n0, n1, n2) box: am = dt pu, bm = dt pv */
PD_GLOBAL void pd_scale_copy(i64 n0, i64 n1, i64 n2, real dt, PD_G real *dst, i64 d0, i64 d1, i64 d2,
                             PD_G const real *src, i64 s0, i64 s1, i64 s2)
{
    PD_LOOP(t, n0 * n1 * n2)
    {
        i64 k = t % n2, r = t / n2, j = r % n1, i = r / n1;
        dst[i * d0 + j * d1 + k * d2] = dt * src[i * s0 + j * s1 + k * s2];
    }
}

/* dynam0's vertical flux of column (i, j): conv_adv, pit accumulated bottom-up, sd(l) for
   l = lm-1 .. 1, cm(l) = (-dt) sd(l) written for l = 1 .. lm-1 into cm's levels 1..lm-1. */
PD_GLOBAL void pd_dynam0_cm(i64 im, i64 jm, i64 lm, real dt, PD_G const real *pu, i64 u0, i64 u1, i64 u2,
                            PD_G const real *pv, i64 v0, i64 v1, i64 v2, PD_G const real *bt, PD_G real *cm, i64 c0, i64 c1,
                            i64 c2)
{
    PD_LOOP(col, im * jm)
    {
        i64 i = col / jm, j = col % jm;
#define CONV(l) (pu[i * u0 + j * u1 + (l) * u2] - pu[(i + 1) * u0 + j * u1 + (l) * u2] + pv[i * v0 + j * v1 + (l) * v2] - \
                 pv[i * v0 + (j + 1) * v1 + (l) * v2])
        real pit = CONV(0);
        for (i64 l = 1; l < lm; l++) pit = pit + CONV(l);
        real sd = CONV(lm - 1) - (bt[lm - 1] - bt[lm]) * pit;
        PD_G real *cc = cm + i * c0 + j * c1;
        if (lm > 1) cc[(lm - 1) * c2] = (-dt) * sd;
        for (i64 lev = lm - 2; lev > 0; lev--) {
            sd = sd + CONV(lev) - (bt[lev] - bt[lev + 1]) * pit;
            cc[lev * c2] = (-dt) * sd;
        }
#undef CONV
    }
}

/* m2phlb1 of column (i, j): p = (sum_l m) grav / dxyp(j); sp = p, phlb(l) = at(l) + bt(l) p.
   The caller zeroes sp's halo. */
PD_GLOBAL void pd_m2phlb1(i64 im, i64 jm, i64 lm, real grav, PD_G const real *m, i64 m0, i64 m1, i64 m2,
                          PD_G const real *dxyp, PD_G const real *at, PD_G const real *bt, PD_G real *sp, i64 p0, i64 p1,
                          PD_G real *phlb, i64 h0, i64 h1, i64 h2)
{
    PD_LOOP(col, im * jm)
    {
        i64 i = col / jm, j = col % jm;
        PD_G const real *mc = m + i * m0 + j * m1;
        real s = mc[0];
        for (i64 l = 1; l < lm; l++) s = s + mc[l * m2];
        real p = s * grav / dxyp[j];
        sp[i * p0 + j * p1] = p;
        for (i64 l = 0; l <= lm; l++) phlb[i * h0 + j * h1 + l * h2] = at[l] + bt[l] * p;
    }
}

/* zoom.massflow.m2phlb of a child column: m2phlb1's p, on the ring rescaled to the parent
   cell's area, p * dxy(j) * xref * yref / dxy_parent(j / yref) (NumPy's left-to-right order);
   sp = p, phlb(l) = at(l) + bt(l) p.  The caller zeroes sp's halo. */
PD_GLOBAL void pd_m2phlb_child(i64 im, i64 jm, i64 lm, real grav, PD_G const real *m, i64 m0, i64 m1, i64 m2,
                               PD_G const real *dxyp, PD_G const real *dxypar, i64 xr, i64 yr,
                               PD_G const unsigned char *ring, PD_G const real *at, PD_G const real *bt,
                               PD_G real *sp, i64 p0, i64 p1, PD_G real *phlb, i64 h0, i64 h1, i64 h2)
{
    PD_LOOP(col, im * jm)
    {
        i64 i = col / jm, j = col % jm;
        PD_G const real *mc = m + i * m0 + j * m1;
        real s = mc[0];
        for (i64 l = 1; l < lm; l++) s = s + mc[l * m2];
        real p = s * grav / dxyp[j];
        if (ring[i * jm + j]) p = p * dxyp[j] * (real)xr * (real)yr / dxypar[j / yr];
        sp[i * p0 + j * p1] = p;
        for (i64 l = 0; l <= lm; l++) phlb[i * h0 + j * h1 + l * h2] = at[l] + bt[l] * p;
    }
}

/* flag = 1 where any of the (n0, n1) view is not 0 (NaN included): the halo checks */
PD_GLOBAL void pd_any_nonzero(i64 n0, i64 n1, PD_G const real *q, i64 s0, i64 s1, PD_G int *flag)
{
    PD_LOOP(t, n0 * n1)
    {
        if (q[(t / n1) * s0 + (t % n1) * s1] != PD_R(0.0)) pd_or_i(flag, 1);
    }
}

/* do_add_2d on the owned cells of level/tracer views rm, rzm (im, jm): rm + x, rzm - x */
PD_GLOBAL void pd_add_2d(i64 im, i64 jm, PD_G const unsigned char *own, PD_G const real *x, PD_G real *rm, PD_G real *rzm,
                         i64 q0, i64 q1)
{
    PD_LOOP(c, im * jm)
    {
        i64 i = c / jm, j = c % jm;
        if (!own[c]) continue;
        real v = x[c];
        rm[i * q0 + j * q1] = rm[i * q0 + j * q1] + v;
        rzm[i * q0 + j * q1] = rzm[i * q0 + j * q1] - v;
    }
}

/* chemistry_region: q = q * factor on the (im, jm, lm) view of one tracer, factor = decay on the
   owned cells of the scope (decay is the scope's, offset si0, sj0), 1.0 elsewhere */
PD_GLOBAL void pd_scale_owned(i64 im, i64 jm, i64 lm, PD_G const unsigned char *own, i64 si0, i64 sj0, i64 sni, i64 snj,
                              PD_G const real *decay, i64 d0, i64 d1, i64 d2, PD_G real *q, i64 q0, i64 q1, i64 q2)
{
    PD_LOOP(t, im * jm * lm)
    {
        i64 l = t % lm, r = t / lm, j = r % jm, i = r / jm;
        real f = PD_R(1.0);
        i64 a = i - si0, b = j - sj0;
        if (own[i * jm + j] && a >= 0 && a < sni && b >= 0 && b < snj) f = decay[a * d0 + b * d1 + l * d2];
        PD_G real *p = q + i * q0 + j * q1 + l * q2;
        *p = *p * f;
    }
}

/* chemistry_region in one pass: q = q * factor for rm and the three slopes of each listed
   (0-based) tracer, factor as pd_scale_owned's -- each element multiplied once, as there */
PD_GLOBAL void pd_scale_owned_all(i64 im, i64 jm, i64 lm, PD_G const unsigned char *own, i64 si0, i64 sj0, i64 sni,
                                  i64 snj, PD_G const real *decay, i64 d0, i64 d1, i64 d2, PD_G real *rm,
                                  PD_G real *rxm, PD_G real *rym, PD_G real *rzm, i64 q0, i64 q1, i64 q2, i64 q3,
                                  PD_G const int *tracers, i64 ntr)
{
    PD_LOOP(t, im * jm * lm)
    {
        i64 l = t % lm, r = t / lm, j = r % jm, i = r / jm;
        real f = PD_R(1.0);
        i64 a = i - si0, b = j - sj0;
        if (own[i * jm + j] && a >= 0 && a < sni && b >= 0 && b < snj) f = decay[a * d0 + b * d1 + l * d2];
        i64 o = i * q0 + j * q1 + l * q2;
        for (i64 k = 0; k < ntr; k++) {
            i64 ot = o + tracers[k] * q3;
            rm[ot] = rm[ot] * f;
            rxm[ot] = rxm[ot] * f;
            rym[ot] = rym[ot] * f;
            rzm[ot] = rzm[ot] * f;
        }
    }
}

#endif

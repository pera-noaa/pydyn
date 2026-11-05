/* chem.loss_rates and chem.decay for one column (i, j) of the scope: the three zonal Bruehl
   columns and the tropospheric Cl rebinned onto the model's half levels (rebin_many: each
   output layer sums, from zero and in order, the input layers first..last-1, an input layer
   that does not overlap contributing +0.0), the rates through pmath. */
#ifndef PD_CHEM
#define PD_CHEM
#include "pd.h"
#include "pmath.cuh"

/* the descending-levels check of chem._descending: 1 when not strictly descending */
PD_FN int pd_not_descending(PD_G const real *e, i64 se, i64 n)
{
    for (i64 k = 0; k + 1 < n; k++)
        if (!(e[(k + 1) * se] - e[k * se] < PD_R(0.0))) return 1;
    return 0;
}

/* rebin_levels(ip_mix) of input layer data d (nip) on edges e (nip+1) over output layer
   [bot, top] */
PD_FN real pd_rebin_layer(PD_G const real *e, i64 se, PD_G const real *d, i64 sd, i64 nip, real bot, real top)
{
    i64 first = 0, last = 0;
    for (i64 k = 0; k < nip; k++) {
        first += e[(k + 1) * se] >= bot;
        last += e[k * se] > top;
    }
    real overlap = PD_R(0.0);
    for (i64 k = first; k < last; k++) {
        i64 kk = k < nip - 1 ? k : nip - 1;
        real thickness = np_minimum(bot, e[kk * se]) - np_maximum(top, e[(kk + 1) * se]);
        overlap = overlap + (thickness > PD_R(0.0) ? d[kk * sd] * thickness : PD_R(0.0));
    }
    real width = bot - top;
    return width > PD_R(0.0) ? overlap / width : overlap;
}

/* consts: OH_A, OH_EA, CL_A, CL_T0, CL_N, CL_EA, oh_scale, cl_scale, o1d_scale, soil_rate, dt_sec */
PD_GLOBAL void pd_loss_rates(i64 ni, i64 nj, i64 lm, PD_G const real *k, PD_G const real *temper, i64 t0, i64 t1, i64 t2,
                             PD_G const real *phlb, i64 p0, i64 p1, i64 p2, PD_G const real *trop, PD_G const real *sedges,
                             i64 nstrat, PD_G const real *s_oh, PD_G const real *s_cl, PD_G const real *s_od,
                             PD_G const real *oh3d, i64 o0, i64 o1, i64 o2, PD_G const real *pl_trop,
                             PD_G const real *cl_trop, i64 c0, i64 c1, i64 c2, i64 ntrop, PD_G const real *soil,
                             PD_G real *loss, PD_G real *decay, PD_G int *bad)
{
    PD_LOOP(col, ni * nj)
    {
        i64 i = col / nj, j = col % nj;
        PD_G const real *P = phlb + i * p0 + j * p1;
        PD_G const real *E = sedges + j * (nstrat + 1);
        if (pd_not_descending(E, 1, nstrat + 1) || pd_not_descending(P, p2, lm + 1)) pd_or_i(bad, 1);
        PD_G const real *pl = pl_trop ? pl_trop + i * c0 + j * c1 : 0;
        PD_G const real *cl = cl_trop ? cl_trop + i * c0 + j * c1 : 0;
        if (pl) {
            /* trop_edges = (pl_trop, 0): descending through the appended zero too */
            if (pd_not_descending(pl, c2, ntrop) || !(PD_R(0.0) - pl[(ntrop - 1) * c2] < PD_R(0.0))) pd_or_i(bad, 1);
        }
        for (i64 l = 0; l < lm; l++) {
            real bot = P[l * p2], top = P[(l + 1) * p2];
            int below = bot > E[0];
            PD_G const real *ds[3] = {s_oh + j * nstrat, s_cl + j * nstrat, s_od + j * nstrat};
            real st[3];
            for (int c = 0; c < 3; c++) st[c] = below ? ds[c][0] : pd_rebin_layer(E, 1, ds[c], 1, nstrat, bot, top);

            real T = temper[i * t0 + j * t1 + l * t2];
            real cl_col = PD_R(0.0);
            if (pl) {
                real v;
                if (bot > pl[0]) {
                    v = cl[0];
                } else { /* edges pl(0..ntrop-1), then 0.0 */
                    i64 first = 0, last = 0;
                    for (i64 q = 0; q < ntrop; q++) {
                        real up = q + 1 < ntrop ? pl[(q + 1) * c2] : PD_R(0.0);
                        first += up >= bot;
                        last += pl[q * c2] > top;
                    }
                    real overlap = PD_R(0.0);
                    for (i64 q = first; q < last; q++) {
                        i64 kk = q < ntrop - 1 ? q : ntrop - 1;
                        real eb = pl[kk * c2], et = kk + 1 < ntrop ? pl[(kk + 1) * c2] : PD_R(0.0);
                        real thickness = np_minimum(bot, eb) - np_maximum(top, et);
                        overlap = overlap + (thickness > PD_R(0.0) ? cl[kk * c2] * thickness : PD_R(0.0));
                    }
                    real width = bot - top;
                    v = width > PD_R(0.0) ? overlap / width : overlap;
                }
                real krate = (k[2] * pm_pow(T / k[3], k[4])) * pm_exp(-k[5] / T);
                cl_col = (v * krate) * k[7];
            }
            real krate_oh = k[0] * pm_exp(-k[1] / T);
            real oh_spiv = (krate_oh * oh3d[i * o0 + j * o1 + l * o2]) * k[6];
            int strat = bot < trop[j];
            real L0 = strat ? st[0] : oh_spiv;
            real L1 = strat ? st[1] : st[1] + cl_col;
            real L2 = st[2] * k[8];
            real L3 = (soil && l == 0) ? -(k[9] * soil[i * nj + j]) : PD_R(0.0);
            PD_G real *out = loss + ((i * nj + j) * lm + l) * 4;
            out[0] = L0;
            out[1] = L1;
            out[2] = L2;
            out[3] = L3;
            decay[(i * nj + j) * lm + l] = pm_exp(-((((L0 + L1) + L2) + L3) * k[10]));
        }
    }
}

#endif

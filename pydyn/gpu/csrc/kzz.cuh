/* diffusion.calc_kzz, one thread per column (i, j): ustar, geopotential, mass_t, bldiff and
   dkg in the reference's statement order, every `jqif` a 0/1 multiplier as there, NumPy's
   maximum/minimum, and exp/log/pow through pmath.cuh.  The per-level work arrays are thread
   local, PD_LMK levels. */
#ifndef PD_KZZ
#define PD_KZZ
#include "pd.h"
#include "pmath.cuh"

#ifndef PD_LMK
#define PD_LMK 64
#endif

PD_FN real pd_jqif(real x, real y) { return pd_signbit(x - y) ? PD_R(0.0) : PD_R(1.0); }
PD_FN real pd_copysign(real x, real y) { return pm_copysign(x, y); }
PD_FN real pd_cube(real x) { return (x * x) * x; }
/* q * a + (1 - q) * b, q 0 or 1: the Fortran's arithmetic in double; a select in float, where the
   unused branch can be inf (a difference rounded to zero) and 0 * inf is NaN */
PD_FN real pd_blend(real q, real a, real b) {
#ifdef PD_F32
    return q != PD_R(0.0) ? a : b;
#else
    return q * a + (PD_R(1.0) - q) * b;
#endif
}
/* a difference of values near ref; in float, one under ref's half ulp (zero, as a rounded
   temperature carries no more) is taken at that size: the double's 1/d is large, the float's inf */
PD_FN real pd_resolved(real d, real ref) {
#ifdef PD_F32
    real eps = pd_fabs(ref) * PD_R(5.9604644775390625e-08);
    return pd_fabs(d) < eps ? pm_copysign(eps, d) : d;
#else
    return d;
#endif
}

/* consts: GRAV VKARMAN RHO_AIR CP_AIR LVAP CKMIN SFFRAC ONET APZERO CCON BETAH BINM FAK FAKN ZKAPPA
   ZCRDQ TINY ZACB RICR SSFRAC KVHMIN dyy (diffusion.py's names, in this order) */
#define GRAV_ k[0]
#define VKARMAN_ k[1]
#define RHO_AIR_ k[2]
#define CP_AIR_ k[3]
#define LVAP_ k[4]
#define CKMIN_ k[5]
#define SFFRAC_ k[6]
#define ONET_ k[7]
#define APZERO_ k[8]
#define CCON_ k[9]
#define BETAH_ k[10]
#define BINM_ k[11]
#define FAK_ k[12]
#define FAKN_ k[13]
#define ZKAPPA_ k[14]
#define ZCRDQ_ k[15]
#define TINY_ k[16]
#define ZACB_ k[17]
#define RICR_ k[18]
#define SSFRAC_ k[19]
#define KVHMIN_ k[20]
#define DYY_ k[21]

/* diffusion._pbl_pass */
PD_FN real pd_pbl_pass(PD_G const real *k, int lm, real pblh, PD_L const real *whm, PD_L const real *uw, PD_L const real *vw,
                         PD_L const real *wthm, PD_L const real *q, real ustr, real wtseff, real reference, int first)
{
    real jwork = PD_R(1.0), wrino = PD_R(0.0);
    for (int l = first ? 0 : 1; l < lm; l++) {
        real du = uw[l] - uw[0], dv = vw[l] - vw[0];
        real vvk = np_maximum(du * du + dv * dv + ZACB_ * ustr * ustr, TINY_);
        real tkv = wthm[l] * (PD_R(1.0) + ZCRDQ_ * q[l]);
        real dz = whm[l] - whm[0];
        real zrino;
        if (first)
            zrino = GRAV_ * (tkv - wtseff) * dz / (wtseff * vvk);
        else
            zrino = GRAV_ * (tkv - reference) * dz / (vvk * wtseff);
        zrino = zrino + pd_copysign(PD_R(1.0e-10), zrino);
        zrino = jwork * zrino;
        real jq = pd_jqif(RICR_, zrino);
        real frac = (RICR_ - wrino) / (zrino - wrino);
        real at_ricr = l == 0 ? frac * whm[l] : whm[l - 1] + frac * (whm[l] - whm[l - 1]);
        pblh = pd_blend(jq, pblh, at_ricr);
        jwork = jq * jwork;
        wrino = zrino + (PD_R(1.0) - jwork) * PD_R(0.1);
    }
    if (first) return (PD_R(1.0) - jwork) * pblh + whm[lm - 1] * jwork;
    return (PD_R(1.0) - jwork) * pblh + jwork * whm[lm - 1];
}

/* surface: oro lsm sr srols u10m v10m sshf slhf ewss nsss, each (im, jm) contiguous */
PD_GLOBAL void pd_calc_kzz(i64 im, i64 jm, i64 lm, i64 lmax_conv, PD_G const real *k, PD_G const real *spm, PD_G const real *sp,
                           PD_G const real *pu, i64 u0, i64 u1, i64 u2, PD_G const real *pv, i64 v0, i64 v1, i64 v2,
                           PD_G const real *t, PD_G const real *q, i64 t0, i64 t1, i64 t2, PD_G const real *surface,
                           PD_G const real *at, PD_G const real *bt, PD_G const real *dxyp, PD_G const real *dxx, PD_G real *dkg,
                           PD_G real *pblh_out)
{
    PD_LOOP(col, im * jm)
    {
        i64 i = col / jm, j = col % jm;
        i64 n2 = im * jm;
        PD_G const real *S = surface + col;
        real oro = S[0], lsm = S[n2], sr = S[2 * n2], srols = S[3 * n2], u10m = S[4 * n2], v10m = S[5 * n2];
        real sshf = S[6 * n2], slhf = S[7 * n2], ewss = S[8 * n2], nsss = S[9 * n2];
        real T[PD_LMK], Q[PD_LMK], M[PD_LMK], gph[PD_LMK + 1], phlb[PD_LMK + 1];
        real pf[PD_LMK], wthm[PD_LMK], zthvk[PD_LMK], whm[PD_LMK], uw[PD_LMK], vw[PD_LMK], lpf[PD_LMK];
        real z[PD_LMK], zdup2[PD_LMK], zrinub[PD_LMK], wkvf[PD_LMK], zdz[PD_LMK], dthv[PD_LMK];
        for (int l = 0; l < lm; l++) {
            T[l] = t[i * t0 + j * t1 + l * t2];
            Q[l] = q[i * t0 + j * t1 + l * t2];
        }

        /* ustar (dd_calc_ustar) */
        real xland = lsm / PD_R(100.0);
        real tau = pd_sqrt(ewss * ewss + nsss * nsss);
        real ustar_sea = np_maximum(PD_R(0.001), pd_sqrt(tau / RHO_AIR_));
        real wind10m = pd_sqrt(u10m * u10m + v10m * v10m);
        real sr_land = np_maximum(PD_R(1e-2), srols);
        real sr_help = np_minimum(sr, PD_R(0.03));
        real land = VKARMAN_ * wind10m / pm_log(PD_R(10.0) / sr_help) * pm_log(PD_R(75.0) / sr_help);
        land = land / pm_log(PD_R(10.0) / sr_land);
        real ustar_land = xland > PD_R(0.0) ? land : PD_R(0.0);
        real ustar_loc = xland * ustar_land + (PD_R(1.0) - xland) * ustar_sea;

        /* geopotential (Meteo_Produce_Gph) */
        real spc = sp[col];
        gph[0] = oro / GRAV_;
        for (int l = 0; l < lm - 1; l++) {
            real tv = T[l] * (PD_R(1.0) + PD_R(0.608) * Q[l]);
            real pdown = at[l] + bt[l] * spc;
            real pup = at[l + 1] + bt[l + 1] * spc;
            gph[l + 1] = gph[l] + tv * PD_R(287.05) * pm_log(pdown / pup) / GRAV_;
        }
        gph[lm] = PD_R(200000.0);

        /* phlb, m_t */
        for (int l = 0; l <= lm; l++) phlb[l] = at[l] + bt[l] * spm[col];
        for (int l = 0; l < lm; l++) M[l] = (phlb[l] - phlb[l + 1]) * dxyp[j] / GRAV_;

        /* bldiff */
        real z0 = gph[0];
        for (int l = 0; l < lm; l++) {
            real lo = phlb[l], hi = phlb[l + 1];
            pf[l] = (lo + hi) * PD_R(0.5);
            wthm[l] = T[l] * pm_pow(APZERO_ / pf[l], ZKAPPA_);
            zthvk[l] = wthm[l] * (PD_R(1.0) + ZCRDQ_ * Q[l]);
            real intfac = (lo - pf[l]) / (lo - hi);
            whm[l] = (gph[l] - z0) * (PD_R(1.0) - intfac) + (gph[l + 1] - z0) * intfac;
        }
        real dz1 = gph[1] - gph[0], dp1 = phlb[0] - phlb[1];
        real wheat = -sshf * dz1 * GRAV_ / dp1 / CP_AIR_;
        real wqflx = -slhf * dz1 * GRAV_ / dp1 / LVAP_;
        real ustr = np_maximum(ustar_loc, PD_R(0.01));
        for (int l = 0; l < lm; l++) { /* _winds */
            uw[l] = dxx[j] * (pu[(i + 1) * u0 + j * u1 + l * u2] + pu[i * u0 + j * u1 + l * u2]) * PD_R(0.5) / M[l];
            vw[l] = DYY_ * (pv[i * v0 + (j + 1) * v1 + l * v2] + pv[i * v0 + j * v1 + l * v2]) * PD_R(0.5) / M[l];
            lpf[l] = pm_log(pf[l]);
        }
        for (int l = 0; l < lm - 1; l++) { /* free troposphere, interfaces l+1 */
            z[l] = gph[l + 1] - z0;
            real jq = pd_jqif(PD_R(1000.0), z[l]);
            real zlambdac = jq * PD_R(300.0) + (PD_R(1.0) - jq) * (PD_R(30.0) + PD_R(270.0) * pm_exp(PD_R(1.0) - z[l] / PD_R(1000.0)));
            real cml2 = PD_R(1.0) / (PD_R(1.0) / zlambdac + PD_R(1.0) / (VKARMAN_ * z[l]));
            cml2 = cml2 * cml2;
            real du = uw[l + 1] - uw[l], dv = vw[l + 1] - vw[l];
            real zd = np_maximum(du * du + dv * dv, PD_R(1.0e-10));
            zdz[l] = whm[l + 1] - whm[l];
            zd = zd / (zdz[l] * zdz[l]);
            zdup2[l] = zd;
            real arg = (pm_log(phlb[l + 1]) - lpf[l]) / (lpf[l + 1] - lpf[l]);
            dthv[l] = zthvk[l + 1] - zthvk[l];
            real zthva = zthvk[l] + arg * dthv[l];
            real zsstab = GRAV_ / zthva * dthv[l] / zdz[l];
            zrinub[l] = zsstab / zd;
            real zfunst = pd_sqrt(np_maximum(PD_R(1.0) - PD_R(18.0) * zrinub[l], PD_R(0.0)));
            real zfstabh = PD_R(1.0) / (PD_R(1.0) + PD_R(10.0) * zrinub[l] * (PD_R(1.0) + PD_R(8.0) * zrinub[l]));
            real zkvn = cml2 * pd_sqrt(zd);
            jq = pd_jqif(zrinub[l], PD_R(0.0));
            wkvf[l] = jq * np_maximum(CKMIN_, zkvn * zfstabh) + (PD_R(1.0) - jq) * np_maximum(CKMIN_, zkvn * zfunst);
        }
        real wtseff = wthm[0] * (PD_R(1.0) + ZCRDQ_ * Q[0]);
        real wheatv = wheat + ZCRDQ_ * wthm[0] * wqflx;
        real wobkl = -wtseff * pd_cube(ustr) / (GRAV_ * VKARMAN_ * (wheatv + pd_copysign(PD_R(1.0e-10), wheatv)));
        real pblh = PD_R(0.0);
        pblh = pd_pbl_pass(k, (int)lm, pblh, whm, uw, vw, wthm, Q, ustr, wtseff, PD_R(0.0), 1);
        real jqh = pd_jqif(wheatv, PD_R(0.0));
        real fmt = pm_pow(jqh * (PD_R(1.0) - BINM_ * pblh / wobkl) + (PD_R(1.0) - jqh), ONET_);
        real wsc = ustr * fmt;
        real zexcess = wheatv * FAK_ / wsc;
        real thvparc = wtseff + jqh * zexcess;
        pblh = pd_pbl_pass(k, (int)lm, pblh, whm, uw, vw, wthm, Q, ustr, wtseff, thvparc, 0);
        pblh = np_maximum(pblh, PD_R(100.0));
        if (pblh_out) pblh_out[col] = pblh;

        real P = pblh, W = wobkl, HV = wheatv, U = ustr, TS = wtseff;
        int n = (int)lmax_conv - 1;
        PD_G real *out = dkg + col * lmax_conv;
        for (int l = 0; l < lmax_conv; l++) out[l] = PD_R(0.0);
        for (int l = 0; l < lm - 1; l++) { /* boundary layer, interfaces l+1 */
            real zl_ = z[l];
            real jq = pd_jqif(P, zl_);
            real zh = jq * zl_ / P;
            real zl = jq * zl_ / W + (PD_R(1.0) - jq);
            real zzh = jq * ((PD_R(1.0) - zh) * (PD_R(1.0) - zh));
            real jq1 = pd_jqif(HV, PD_R(0.0));
            real jq2 = jq * (PD_R(1.0) - jq1);
            real jq3 = jq * jq1;
            real jq6 = jq1 * pd_jqif(P, whm[l]) * pd_jqif(whm[l + 1], P);
            jq1 = pd_jqif(SFFRAC_, zh);
            real jq4 = jq3 * jq1;
            real jq5 = jq3 * (PD_R(1.0) - jq1);
            real cml2 = PD_R(1.0) / (PD_R(1.0) / (VKARMAN_ * zl_) + PD_R(1.0) / PD_R(450.0));
            cml2 = cml2 * cml2;
            real jqq = pd_jqif(zrinub[l], PD_R(0.0));
            real zfstabh = jqq / (PD_R(1.0) + PD_R(10.0) * zrinub[l] * pd_sqrt(PD_R(1.0) + jqq * zrinub[l])) + (PD_R(1.0) - jqq);
            real zpblk = cml2 * pd_sqrt(zdup2[l]) * zfstabh;
            real zkvh = jq2 * np_maximum(zpblk, KVHMIN_) + (PD_R(1.0) - jq2) * wkvf[l];
            real zslask = PD_R(1.0) - BETAH_ * zl;
            zslask = jq4 * zslask + (PD_R(1.0) - jq4);
            zpblk = (PD_R(1.0) - jq4) * zkvh + jq4 * (U * VKARMAN_ * zl_ * zzh * pm_pow(zslask, ONET_));
            zslask = PD_R(1.0) - SSFRAC_ * BETAH_ * P / W;
            zslask = jq5 * zslask + (PD_R(1.0) - jq5);
            real fm = pm_pow(zslask, ONET_);
            real ws = U * fm;
            zpblk = (PD_R(1.0) - jq5) * zpblk + jq5 * ws * VKARMAN_ * zl_ * zzh;
            real zwstr = pm_pow(pd_fabs(HV) * GRAV_ * P / TS, ONET_);
            real obukov = jq3 * W - (PD_R(1.0) - jq3);
            real stab = PD_R(1.0) - SSFRAC_ * BETAH_ * P / obukov;
            real term1 = pm_pow(stab, ONET_);
            real term2 = pd_sqrt(stab);
            real term3 = CCON_ * FAKN_ * zwstr / (U * term1);
            real pr = jq3 * (term1 / term2 + term3) + (PD_R(1.0) - jq3);
            real kvh = jq3 * np_maximum(zpblk / pr, KVHMIN_) + (PD_R(1.0) - jq3) * zkvh;
            real thvgrad = pd_resolved(dthv[l], zthvk[l + 1]) / zdz[l];
            real kvhentr = PD_R(0.2) * HV / thvgrad;
            kvh = pd_blend(jq6, kvhentr, kvh);
            if (l < n) { /* dkg */
                real dz = gph[l + 2] - gph[l];
                out[l] = np_maximum(PD_R(0.0), kvh) * PD_R(2.0) * (M[l + 1] + M[l]) / (dz * dz);
            }
        }
    }
}
#undef GRAV_
#undef VKARMAN_
#undef RHO_AIR_
#undef CP_AIR_
#undef LVAP_
#undef CKMIN_
#undef SFFRAC_
#undef ONET_
#undef APZERO_
#undef CCON_
#undef BETAH_
#undef BINM_
#undef FAK_
#undef FAKN_
#undef ZKAPPA_
#undef ZCRDQ_
#undef TINY_
#undef ZACB_
#undef RICR_
#undef SSFRAC_
#undef KVHMIN_
#undef DYY_

#endif

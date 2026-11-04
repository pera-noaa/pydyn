/* exp, log and pow that round the same on the CPU and the GPU: Sun's fdlibm (e_exp.c, e_log.c,
   e_pow.c, s_scalbn.c, the algorithms Java's StrictMath specifies), plain IEEE double
   operations and word manipulation, nothing a compiler or a device library may evaluate
   differently once contraction is off.  Not the system libm: within an ulp of it, and the
   CPU reference reaches these same functions through the host build's exp/log/pow
   (ops.use_libm), which is what makes a device result comparable bitwise. */
#ifndef PD_PMATH
#define PD_PMATH
#include "pd.h"

#ifndef PD_F32

#ifdef PD_CUDA
PD_FN int pm_hi(double x) { return __double2hiint(x); }
PD_FN unsigned pm_lo(double x) { return (unsigned)__double2loint(x); }
PD_FN double pm_make(int hi, unsigned lo) { return __hiloint2double(hi, (int)lo); }
#else
PD_FN int pm_hi(double x)
{
    u64 b;
    memcpy(&b, &x, 8);
    return (int)(b >> 32);
}
PD_FN unsigned pm_lo(double x)
{
    u64 b;
    memcpy(&b, &x, 8);
    return (unsigned)b;
}
PD_FN double pm_make(int hi, unsigned lo)
{
    u64 b = ((u64)(unsigned)hi << 32) | lo;
    double x;
    memcpy(&x, &b, 8);
    return x;
}
#endif
PD_FN double pm_set_hi(double x, int hi) { return pm_make(hi, pm_lo(x)); }
PD_FN double pm_set_lo(double x, unsigned lo) { return pm_make(pm_hi(x), lo); }

PD_FN double pm_copysign(double x, double y) { return pm_make((pm_hi(x) & 0x7fffffff) | (pm_hi(y) & 0x80000000), pm_lo(x)); }

PD_FN_NOINLINE double pm_scalbn(double x, int n)
{
    const double two54 = 1.80143985094819840000e+16, twom54 = 5.55111512312578270212e-17;
    const double huge = 1.0e+300, tiny = 1.0e-300;
    int hx = pm_hi(x);
    int lx = (int)pm_lo(x);
    int k = (hx & 0x7ff00000) >> 20;
    if (k == 0) {
        if ((lx | (hx & 0x7fffffff)) == 0) return x;
        x *= two54;
        hx = pm_hi(x);
        k = ((hx & 0x7ff00000) >> 20) - 54;
        if (n < -50000) return tiny * x;
    }
    if (k == 0x7ff) return x + x;
    k = k + n;
    if (k > 0x7fe) return huge * pm_copysign(huge, x);
    if (k > 0) return pm_set_hi(x, (hx & 0x800fffff) | (k << 20));
    if (k <= -54) {
        if (n > 50000) return huge * pm_copysign(huge, x);
        return tiny * pm_copysign(tiny, x);
    }
    k += 54;
    x = pm_set_hi(x, (hx & 0x800fffff) | (k << 20));
    return x * twom54;
}

PD_FN_NOINLINE double pm_exp(double x)
{
    const double one = 1.0, huge = 1.0e+300, twom1000 = 9.33263618503218878990e-302;
    const double o_threshold = 7.09782712893383973096e+02, u_threshold = -7.45133219101941108420e+02;
    const double ln2HI[2] = {6.93147180369123816490e-01, -6.93147180369123816490e-01};
    const double ln2LO[2] = {1.90821492927058770002e-10, -1.90821492927058770002e-10};
    const double halF[2] = {0.5, -0.5};
    const double invln2 = 1.44269504088896338700e+00;
    const double P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03,
                 P3 = 6.61375632143793436117e-05, P4 = -1.65339022054652515390e-06,
                 P5 = 4.13813679705723846039e-08;
    double y, hi = 0.0, lo = 0.0, c, t;
    int k = 0, xsb;
    unsigned hx = (unsigned)pm_hi(x);
    xsb = (hx >> 31) & 1;
    hx &= 0x7fffffff;
    if (hx >= 0x40862E42) {
        if (hx >= 0x7ff00000) {
            if (((hx & 0xfffff) | pm_lo(x)) != 0) return x + x;
            return (xsb == 0) ? x : 0.0;
        }
        if (x > o_threshold) return huge * huge;
        if (x < u_threshold) return twom1000 * twom1000;
    }
    if (hx > 0x3fd62e42) {
        if (hx < 0x3FF0A2B2) {
            hi = x - ln2HI[xsb];
            lo = ln2LO[xsb];
            k = 1 - xsb - xsb;
        } else {
            k = (int)(invln2 * x + halF[xsb]);
            t = k;
            hi = x - t * ln2HI[0];
            lo = t * ln2LO[0];
        }
        x = hi - lo;
    } else if (hx < 0x3e300000) {
        if (huge + x > one) return one + x;
    } else {
        k = 0;
    }
    t = x * x;
    c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if (k == 0) return one - ((x * c) / (c - 2.0) - x);
    y = one - ((lo - (x * c) / (2.0 - c)) - hi);
    if (k >= -1021) return pm_set_hi(y, pm_hi(y) + (int)((unsigned)k << 20));
    y = pm_set_hi(y, pm_hi(y) + (int)((unsigned)(k + 1000) << 20));
    return y * twom1000;
}

PD_FN_NOINLINE double pm_log(double x)
{
    const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10;
    const double two54 = 1.80143985094819840000e+16;
    const double Lg1 = 6.666666666666735130e-01, Lg2 = 3.999999999940941908e-01, Lg3 = 2.857142874366239149e-01,
                 Lg4 = 2.222219843214978396e-01, Lg5 = 1.818357216161805012e-01, Lg6 = 1.531383769920937332e-01,
                 Lg7 = 1.479819860511658591e-01;
    const double zero = 0.0;
    double hfsq, f, s, z, R, w, t1, t2, dk;
    int k, hx, i, j;
    unsigned lx;
    hx = pm_hi(x);
    lx = pm_lo(x);
    k = 0;
    if (hx < 0x00100000) {
        if (((hx & 0x7fffffff) | lx) == 0) return -two54 / zero;
        if (hx < 0) return (x - x) / zero;
        k -= 54;
        x *= two54;
        hx = pm_hi(x);
    }
    if (hx >= 0x7ff00000) return x + x;
    k += (hx >> 20) - 1023;
    hx &= 0x000fffff;
    i = (hx + 0x95f64) & 0x100000;
    x = pm_set_hi(x, hx | (i ^ 0x3ff00000));
    k += (i >> 20);
    f = x - 1.0;
    if ((0x000fffff & (2 + hx)) < 3) {
        if (f == zero) {
            if (k == 0) return zero;
            dk = (double)k;
            return dk * ln2_hi + dk * ln2_lo;
        }
        R = f * f * (0.5 - 0.33333333333333333 * f);
        if (k == 0) return f - R;
        dk = (double)k;
        return dk * ln2_hi - ((R - dk * ln2_lo) - f);
    }
    s = f / (2.0 + f);
    dk = (double)k;
    z = s * s;
    i = hx - 0x6147a;
    w = z * z;
    j = 0x6b851 - hx;
    t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    i |= j;
    R = t2 + t1;
    if (i > 0) {
        hfsq = 0.5 * f * f;
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
    }
    if (k == 0) return f - s * (f - R);
    return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

PD_FN_NOINLINE double pm_pow(double x, double y)
{
    const double bp[2] = {1.0, 1.5}, dp_h[2] = {0.0, 5.84962487220764160156e-01},
                 dp_l[2] = {0.0, 1.35003920212974897128e-08};
    const double zero = 0.0, one = 1.0, two = 2.0, two53 = 9007199254740992.0, huge = 1.0e300, tiny = 1.0e-300;
    const double L1 = 5.99999999999994648725e-01, L2 = 4.28571428578550184252e-01, L3 = 3.33333329818377432918e-01,
                 L4 = 2.72728123808534006489e-01, L5 = 2.30660745775561754067e-01, L6 = 2.06975017800338417784e-01;
    const double P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03,
                 P3 = 6.61375632143793436117e-05, P4 = -1.65339022054652515390e-06,
                 P5 = 4.13813679705723846039e-08;
    const double lg2 = 6.93147180559945286227e-01, lg2_h = 6.93147182464599609375e-01,
                 lg2_l = -1.90465429995776804525e-09;
    const double ovt = 8.0085662595372944372e-17;
    const double cp = 9.61796693925975554329e-01, cp_h = 9.61796700954437255859e-01,
                 cp_l = -7.02846165095275826516e-09;
    const double ivln2 = 1.44269504088896338700e+00, ivln2_h = 1.44269502162933349609e+00,
                 ivln2_l = 1.92596299112661746887e-08;
    double z, ax, z_h, z_l, p_h, p_l;
    double y1, t1, t2, r, s, t, u, v, w;
    int i, j, k, yisint, n;
    int hx, hy, ix, iy;
    unsigned lx, ly;

    hx = pm_hi(x);
    lx = pm_lo(x);
    hy = pm_hi(y);
    ly = pm_lo(y);
    ix = hx & 0x7fffffff;
    iy = hy & 0x7fffffff;
    if ((iy | ly) == 0) return one;
    if (ix > 0x7ff00000 || ((ix == 0x7ff00000) && (lx != 0)) || iy > 0x7ff00000 || ((iy == 0x7ff00000) && (ly != 0)))
        return x + y;
    yisint = 0;
    if (hx < 0) {
        if (iy >= 0x43400000)
            yisint = 2;
        else if (iy >= 0x3ff00000) {
            k = (iy >> 20) - 0x3ff;
            if (k > 20) {
                j = (int)(ly >> (52 - k));
                if (((unsigned)j << (52 - k)) == ly) yisint = 2 - (j & 1);
            } else if (ly == 0) {
                j = iy >> (20 - k);
                if ((j << (20 - k)) == iy) yisint = 2 - (j & 1);
            }
        }
    }
    if (ly == 0) {
        if (iy == 0x7ff00000) {
            if (((ix - 0x3ff00000) | (int)lx) == 0) return y - y;
            if (ix >= 0x3ff00000) return (hy >= 0) ? y : zero;
            return (hy < 0) ? -y : zero;
        }
        if (iy == 0x3ff00000) {
            if (hy < 0) return one / x;
            return x;
        }
        if (hy == 0x40000000) return x * x;
        if (hy == 0x3fe00000) {
            if (hx >= 0) return sqrt(x);
        }
    }
    ax = pd_fabs(x);
    if (lx == 0) {
        if (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000) {
            z = ax;
            if (hy < 0) z = one / z;
            if (hx < 0) {
                if (((ix - 0x3ff00000) | yisint) == 0)
                    z = (z - z) / (z - z);
                else if (yisint == 1)
                    z = -z;
            }
            return z;
        }
    }
    n = (hx >> 31) + 1;
    if ((n | yisint) == 0) return (x - x) / (x - x);
    s = one;
    if ((n | (yisint - 1)) == 0) s = -one;
    if (iy > 0x41e00000) {
        if (iy > 0x43f00000) {
            if (ix <= 0x3fefffff) return (hy < 0) ? huge * huge : tiny * tiny;
            if (ix >= 0x3ff00000) return (hy > 0) ? huge * huge : tiny * tiny;
        }
        if (ix < 0x3fefffff) return (hy < 0) ? s * huge * huge : s * tiny * tiny;
        if (ix > 0x3ff00000) return (hy > 0) ? s * huge * huge : s * tiny * tiny;
        t = ax - one;
        w = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25));
        u = ivln2_h * t;
        v = t * ivln2_l - w * ivln2;
        t1 = u + v;
        t1 = pm_set_lo(t1, 0);
        t2 = v - (t1 - u);
    } else {
        double ss, s2, s_h, s_l, t_h, t_l;
        n = 0;
        if (ix < 0x00100000) {
            ax *= two53;
            n -= 53;
            ix = pm_hi(ax);
        }
        n += ((ix) >> 20) - 0x3ff;
        j = ix & 0x000fffff;
        ix = j | 0x3ff00000;
        if (j <= 0x3988E)
            k = 0;
        else if (j < 0xBB67A)
            k = 1;
        else {
            k = 0;
            n += 1;
            ix -= 0x00100000;
        }
        ax = pm_set_hi(ax, ix);
        u = ax - bp[k];
        v = one / (ax + bp[k]);
        ss = u * v;
        s_h = ss;
        s_h = pm_set_lo(s_h, 0);
        t_h = zero;
        t_h = pm_set_hi(t_h, ((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18));
        t_l = ax - (t_h - bp[k]);
        s_l = v * ((u - s_h * t_h) - s_h * t_l);
        s2 = ss * ss;
        r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        s2 = s_h * s_h;
        t_h = 3.0 + s2 + r;
        t_h = pm_set_lo(t_h, 0);
        t_l = r - ((t_h - 3.0) - s2);
        u = s_h * t_h;
        v = s_l * t_h + t_l * ss;
        p_h = u + v;
        p_h = pm_set_lo(p_h, 0);
        p_l = v - (p_h - u);
        z_h = cp_h * p_h;
        z_l = cp_l * p_h + p_l * cp + dp_l[k];
        t = (double)n;
        t1 = (((z_h + z_l) + dp_h[k]) + t);
        t1 = pm_set_lo(t1, 0);
        t2 = z_l - (((t1 - t) - dp_h[k]) - z_h);
    }
    y1 = y;
    y1 = pm_set_lo(y1, 0);
    p_l = (y - y1) * t1 + y * t2;
    p_h = y1 * t1;
    z = p_l + p_h;
    j = pm_hi(z);
    i = (int)pm_lo(z);
    if (j >= 0x40900000) {
        if (((j - 0x40900000) | i) != 0) return s * huge * huge;
        if (p_l + ovt > z - p_h) return s * huge * huge;
    } else if ((j & 0x7fffffff) >= 0x4090cc00) {
        if (((j - (int)0xc090cc00) | i) != 0) return s * tiny * tiny;
        if (p_l <= z - p_h) return s * tiny * tiny;
    }
    i = j & 0x7fffffff;
    k = (i >> 20) - 0x3ff;
    n = 0;
    if (i > 0x3fe00000) {
        n = j + (0x00100000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff;
        t = zero;
        t = pm_set_hi(t, (n & ~(0x000fffff >> k)));
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if (j < 0) n = -n;
        p_h -= t;
    }
    t = p_l + p_h;
    t = pm_set_lo(t, 0);
    u = t * lg2_h;
    v = (p_l - (t - p_h)) * lg2 + t * lg2_l;
    z = u + v;
    w = v - (z - u);
    t = z * z;
    t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    r = (z * t1) / (t1 - two) - (w + z * w);
    z = one - (r - z);
    j = pm_hi(z);
    j += (int)((unsigned)n << 20);
    if ((j >> 20) <= 0)
        z = pm_scalbn(z, n);
    else
        z = pm_set_hi(z, pm_hi(z) + (int)((unsigned)n << 20));
    return s * z;
}


#else /* PD_F32 */
/* float exp / log from FreeBSD msun's e_expf.c / e_logf.c (the same reductions in float, its
   minimax coefficients), pow as exp(y log x): a few ulp where |y log x| is large, which the
   PD_F32 build accepts; all of it IEEE float operations and word manipulation. */
PD_FN real pm_copysign(real x, real y) { return pd_u2f((pd_f2u(x) & 0x7fffffffu) | (pd_f2u(y) & 0x80000000u)); }

PD_FN float pm_pow2i(int e) { return pd_u2f((u32)(e + 127) << 23); } /* 2^e, e in [-126, 127] */

PD_FN_NOINLINE float pm_exp(float x)
{
    const float ln2HI[2] = {6.9314575195e-01f, -6.9314575195e-01f}; /* 0x3f317200 */
    const float ln2LO[2] = {1.4286067653e-06f, -1.4286067653e-06f}; /* 0x35bfbe8e */
    const float halF[2] = {0.5f, -0.5f};
    const float invln2 = 1.4426950216e+00f; /* 0x3fb8aa3b */
    const float P1 = 1.6666625440e-1f, P2 = -2.7667332906e-3f;
    u32 hx = pd_f2u(x);
    int xsb = (int)(hx >> 31);
    hx &= 0x7fffffffu;
    if (hx >= 0x42b17218u) {              /* |x| >= 88.721, or NaN */
        if (hx > 0x7f800000u) return x + x; /* NaN */
        if (hx == 0x7f800000u) return xsb == 0 ? x : 0.0f;
        if (xsb == 0) return 1.0e+30f * 1.0e+30f; /* overflow */
        if (hx > 0x42cff1b5u) return 0.0f;        /* x < -103.97: underflow */
    }
    float hi = 0.0f, lo = 0.0f;
    int k = 0;
    if (hx > 0x3eb17218u) {      /* |x| > 0.5 ln2 */
        if (hx < 0x3F851592u) { /* and |x| < 1.5 ln2 */
            hi = x - ln2HI[xsb];
            lo = ln2LO[xsb];
            k = 1 - xsb - xsb;
        } else {
            k = (int)(invln2 * x + halF[xsb]);
            float t = (float)k;
            hi = x - t * ln2HI[0];
            lo = t * ln2LO[0];
        }
        x = hi - lo;
    } else if (hx < 0x39000000u) { /* |x| < 2^-14 */
        return 1.0f + x;
    }
    float t = x * x;
    float c = x - t * (P1 + t * P2);
    if (k == 0) return 1.0f - ((x * c) / (c - 2.0f) - x);
    float y = 1.0f - ((lo - (x * c) / (2.0f - c)) - hi);
    if (k >= -125) return k == 128 ? y * 2.0f * pm_pow2i(127) : y * pm_pow2i(k);
    return (y * pm_pow2i(k + 100)) * pm_pow2i(-100); /* flushed below 2^-126 as the GPU does */
}

PD_FN_NOINLINE float pm_log(float x)
{
    const float ln2_hi = 6.9313812256e-01f; /* 0x3f317180 */
    const float ln2_lo = 9.0580006145e-06f; /* 0x3717f7d1 */
    const float Lg1 = 0.66666662693f, Lg2 = 0.40000972152f, Lg3 = 0.28498786688f, Lg4 = 0.24279078841f;
    int ix = (int)pd_f2u(x);
    int k = 0;
    if (ix < 0x00800000) {                           /* x < 2^-126 */
        if ((ix & 0x7fffffff) < 0x00800000) return -1.0f / 0.0f; /* +-0, and a subnormal the GPU reads as 0 */
        return (x - x) / 0.0f;                       /* log(-#) = NaN */
    }
    if (ix >= 0x7f800000) return x + x;
    k += (ix >> 23) - 127;
    ix &= 0x007fffff;
    int i = (ix + (0x95f64 << 3)) & 0x800000;
    x = pd_u2f((u32)(ix | (i ^ 0x3f800000))); /* normalize x or x/2 */
    k += (i >> 23);
    float f = x - 1.0f;
    float dk;
    if ((0x007fffff & (0x8000 + ix)) < 0xc000) { /* -2^-9 <= f < 2^-9 */
        if (f == 0.0f) {
            if (k == 0) return 0.0f;
            dk = (float)k;
            return dk * ln2_hi + dk * ln2_lo;
        }
        float R = f * f * (0.5f - 0.33333333333333333f * f);
        if (k == 0) return f - R;
        dk = (float)k;
        return dk * ln2_hi - ((R - dk * ln2_lo) - f);
    }
    float s = f / (2.0f + f);
    dk = (float)k;
    float z = s * s;
    i = ix - (0x6147a << 3);
    float w = z * z;
    int j = (0x6b851 << 3) - ix;
    float t1 = w * (Lg2 + w * Lg4);
    float t2 = z * (Lg1 + w * Lg3);
    i |= j;
    float R = t2 + t1;
    if (i > 0) {
        float hfsq = 0.5f * f * f;
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
    }
    if (k == 0) return f - s * (f - R);
    return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

/* log(x) as hi + lo for a positive normal x: msun's reduction, the sum k ln2 + f - f^2/2 + T
   carried in two floats (two_sum, fma) so that pow keeps ~2^-34 of it */
PD_FN void pm_two_sum(float a, float b, PD_L float *s, PD_L float *e)
{
    float t = a + b, bb = t - a;
    *s = t;
    *e = (a - (t - bb)) + (b - bb);
}

PD_FN void pm_log_ff(float x, PD_L float *hi, PD_L float *lo)
{
    const float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f;
    const float Lg1 = 0.66666662693f, Lg2 = 0.40000972152f, Lg3 = 0.28498786688f, Lg4 = 0.24279078841f;
    int ix = (int)pd_f2u(x);
    int k = (ix >> 23) - 127;
    ix &= 0x007fffff;
    int i = (ix + (0x95f64 << 3)) & 0x800000;
    float m = pd_u2f((u32)(ix | (i ^ 0x3f800000)));
    k += (i >> 23);
    float f = m - 1.0f; /* exact */
    float s = f / (2.0f + f), z = s * s, w = z * z;
    float R = z * (Lg1 + w * Lg3) + w * (Lg2 + w * Lg4);
    float hf = 0.5f * f;
    float hfsq = hf * f, hfsq_lo = pd_fma(hf, f, -hfsq);
    float T = s * (hfsq + R);
    float dk = (float)k;
    float s1, e1, s2, e2;
    pm_two_sum(dk * ln2_hi, f, &s1, &e1); /* dk * ln2_hi exact */
    pm_two_sum(s1, -hfsq, &s2, &e2);
    float rest = ((e1 + e2) - hfsq_lo) + (T + dk * ln2_lo);
    float h = s2 + rest;
    *hi = h;
    *lo = rest - (h - s2);
}

PD_FN_NOINLINE float pm_pow(float x, float y)
{
    if (y == 0.0f || x == 1.0f) return 1.0f;
    if (pd_isnan(x) || pd_isnan(y)) return x + y;
    float ax = pd_fabs(x);
    float sign = 1.0f;
    if (x < 0.0f) {
        float hy = y * 0.5f;
        if (pd_fabs(y) < 16777216.0f && (float)(int)y != y) return (x - x) / (x - x); /* non-integer y */
        if (pd_fabs(y) < 16777216.0f && (float)(int)hy != hy) sign = -1.0f;
    }
    if (ax == 0.0f || pd_f2u(ax) < 0x00800000u) return y > 0.0f ? sign * 0.0f : sign / 0.0f;
    if (pd_f2u(ax) >= 0x7f800000u) return (y > 0.0f) == (ax > 1.0f) ? sign * ax : sign * 0.0f;
    float lh, ll;
    pm_log_ff(ax, &lh, &ll);
    float p = y * lh;
    float pe = pd_fma(y, lh, -p) + y * ll;
    float e = pm_exp(p);
    return sign * (e + e * pe);
}
#endif /* PD_F32 */

#endif

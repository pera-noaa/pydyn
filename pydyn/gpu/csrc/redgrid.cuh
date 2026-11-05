/* redgrid.uni2red's cluster sums: out(i, a, b) = seqsum_k q(i c + k, a, b), k = 0..c-1, in order */
#ifndef PD_REDGRID
#define PD_REDGRID
#include "pd.h"

PD_GLOBAL void pd_cluster_sum(i64 n, i64 c, i64 r1, i64 r2, PD_G const real *q, i64 s0, i64 s1, i64 s2, PD_G real *out,
                              i64 o0, i64 o1, i64 o2)
{
    PD_LOOP(t, n * r1 * r2)
    {
        i64 b = t % r2, r = t / r2, a = r % r1, i = r / r1;
        PD_G const real *p = q + (i * c) * s0 + a * s1 + b * s2;
        real acc = p[0];
        for (i64 k = 1; k < c; k++) acc = acc + p[k * s0];
        out[i * o0 + a * o1 + b * o2] = acc;
    }
}

#endif

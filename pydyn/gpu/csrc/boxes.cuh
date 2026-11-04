/* The driver's small statements -- copies, scalings, divisions and sequential sums of strided
   boxes of the state (the zoom exchange, halos, the reduced grid) -- batched (gpu/boxes.py):
   entry b of a batch is one statement dst = op(src) over a box of up to 4 axes, one thread
   per destination element.  No entry of a batch reads what another writes; the caller stages
   them.  Arrays are given by address (PD_PTR), so one batch spans any arrays of the state.

   An entry's header, PD_BOX_W i64s: op, (unused), start (its first thread), dst, extents
   e0..e3, dst strides, then three sources each as address, strides, divisors (src index = dst
   index / div, the repeat of _repeat_rows), the summed axes (n1, stride1, n2, stride2) and the
   element count; sc[b] its scalar.  Entries start on PD_BOX_CHUNK-thread boundaries, so
   blk[t / PD_BOX_CHUNK] is the entry of thread t (padding threads past an entry's count
   return).

     COPY    s0             MUL    s0 * sc         DIV  s0 / sc          SET  sc
     SEQSUM  ops.seqsum of s0 along summed axis 1: the first term, then the rest added in order
     ZSUM    0 + the terms in order, axis 1 outer, axis 2 inner (the exchange's block sums)
     ZSUMMUL 0 + each term * sc, in that order (coarsen_region's faces * tref)
     ADV     s0 + s1 - s2 (ops.advance)   DIVMUL  (s0 / s1) * s2 (red2uni) */
#ifndef PD_BOXES
#define PD_BOXES
#include "pd.h"

#define PD_BOX_W 44
#define PD_BOX_CHUNK 256
enum {
    PD_BOX_COPY = 0,
    PD_BOX_MUL,
    PD_BOX_DIV,
    PD_BOX_SEQSUM,
    PD_BOX_SET,
    PD_BOX_ADV,
    PD_BOX_DIVMUL,
    PD_BOX_ZSUM,
    PD_BOX_ZSUMMUL
};

/* 32-bit index arithmetic (a box has < 2^31 elements; 64-bit division is slow on a GPU), and
   no division where the divisor is 1 (a broadcast axis has stride 0 and divisor 1) */
PD_FN int pd_box_div(int i, int d) { return d == 1 ? i : i / d; }

PD_FN PD_G const real *pd_box_src(PD_G const i64a *s, int i0, int i1, int i2, int i3)
{
    return PD_PTR(s[0]) + pd_box_div(i0, (int)s[5]) * (i64)s[1] + pd_box_div(i1, (int)s[6]) * (i64)s[2] +
           pd_box_div(i2, (int)s[7]) * (i64)s[3] + pd_box_div(i3, (int)s[8]) * (i64)s[4];
}

PD_GLOBAL void pd_boxes(i64 total, PD_G const int *blk, PD_G const i64a *hdr, PD_G const real *sc)
{
    PD_LOOP(t, total)
    {
        int lo = blk[t / PD_BOX_CHUNK];
        PD_G const i64a *h = hdr + (i64a)lo * PD_BOX_W;
        if (t - h[2] >= h[43]) continue; /* padding */
        int r = (int)(t - h[2]), e1 = (int)h[5], e2 = (int)h[6], e3 = (int)h[7];
        int i3 = r % e3;
        r /= e3;
        int i2 = r % e2;
        r /= e2;
        int i1 = r % e1, i0 = r / e1;
        PD_G real *d = PD_PTR(h[3]) + i0 * (i64)h[8] + i1 * (i64)h[9] + i2 * (i64)h[10] + i3 * (i64)h[11];
        int op = (int)h[0];
        if (op == PD_BOX_SET) {
            *d = sc[lo];
            continue;
        }
        PD_G const real *s0 = pd_box_src(h + 12, i0, i1, i2, i3);
        real v;
        if (op == PD_BOX_COPY) {
            v = *s0;
        } else if (op == PD_BOX_MUL) {
            v = *s0 * sc[lo];
        } else if (op == PD_BOX_DIV) {
            v = *s0 / sc[lo];
        } else if (op == PD_BOX_SEQSUM) { /* the first term, then the rest added in order */
            v = s0[0];
            for (int k = 1; k < (int)h[39]; k++) v = v + s0[k * (i64)h[40]];
        } else if (op == PD_BOX_ZSUM || op == PD_BOX_ZSUMMUL) {
            v = PD_R(0.0);
            for (int a = 0; a < (int)h[39]; a++)
                for (int b = 0; b < (int)h[41]; b++) {
                    real x = s0[a * (i64)h[40] + b * (i64)h[42]];
                    v = v + (op == PD_BOX_ZSUMMUL ? x * sc[lo] : x);
                }
        } else {
            real a1 = *pd_box_src(h + 21, i0, i1, i2, i3), a2 = *pd_box_src(h + 30, i0, i1, i2, i3);
            v = op == PD_BOX_ADV ? *s0 + a1 - a2 : (*s0 / a1) * a2;
        }
        *d = v;
    }
}

#endif

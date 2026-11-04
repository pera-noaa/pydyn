/* pydyn.gpu: what lets one source be the CUDA module (NVRTC, --fmad=false), the Metal library
   (xcrun metal -ffp-contract=off) and their host build (cc -ffp-contract=off), so that each
   device and its host build run the same IEEE operations in the same order.

   Kernels are grid-stride loops over independent work items with no block cooperation, so
   the host build runs them as plain loops.  The only cross-thread operations are order-free:
   max/or of integers, and max of non-negative reals through their bit patterns.

   `real` is double, or float under PD_F32 (Metal has no double).  Apple's GPU flushes float
   subnormals to zero on input and output; a PD_F32 host build sets the CPU's FZ bit for the
   kernel's duration, which reproduces it (bitwise on add, mul, div, sqrt, fma). */
#ifndef PD_H
#define PD_H

#ifdef PD_F32
typedef float real;
#define PD_R(x) x##f
#define PD_REAL_TINY 1.17549435e-38f /* SLAMCH('S') */
#else
typedef double real;
#define PD_R(x) x
#define PD_REAL_TINY 2.2250738585072014e-308 /* DLAMCH('S') */
#endif

#if defined(__METAL_VERSION__)
/* PD_GLOBAL kernels are rewritten by backend.metal_source into a body taking (pd_tid, pd_nth)
   and a `kernel` wrapper reading its arguments from one struct */
#include <metal_stdlib>
using namespace metal;
#define PD_METAL 1
#define PD_G device
#define PD_L thread
#define PD_GLOBAL static
#define PD_FN static inline
#define PD_FN_NOINLINE static
/* indices and strides in 32 bits: every array is under 2^31 elements (the arena is < 8 GB) and
   64-bit division is slow on the GPU; i64a for arrays that hold 64-bit data (addresses) */
typedef int i64;
typedef long i64a;
typedef ulong u64;
typedef uint u32;
#define PD_LOOP(t, n) for (i64 t = (i64)pd_tid; t < (i64)(n); t += (i64)pd_nth)
#define PD_PTR(a) reinterpret_cast<PD_G real *>((ulong)(a)) /* an address from the host */
PD_FN void pd_max_i(PD_G int *p, int v) { atomic_fetch_max_explicit((PD_G atomic_int *)p, v, memory_order_relaxed); }
PD_FN void pd_or_i(PD_G int *p, int v) { atomic_fetch_or_explicit((PD_G atomic_int *)p, v, memory_order_relaxed); }
PD_FN void pd_max_nonneg(PD_G real *p, real v)
{
    atomic_fetch_max_explicit((PD_G atomic_uint *)p, as_type<uint>(v), memory_order_relaxed);
}
PD_FN u32 pd_f2u(float x) { return as_type<uint>(x); }
PD_FN float pd_u2f(u32 x) { return as_type<float>(x); }
PD_FN real pd_fabs(real x) { return fabs(x); }
PD_FN int pd_isnan(real x) { return isnan(x); }
PD_FN int pd_signbit(real x) { return signbit(x); }
PD_FN real pd_sqrt(real x) { return sqrt(x); }
PD_FN real pd_fma(real a, real b, real c) { return fma(a, b, c); }
#elif defined(__CUDACC_RTC__) || defined(__CUDACC__) || defined(__HIPCC_RTC__) || defined(__HIPCC__)
/* CUDA (NVRTC --fmad=false) or HIP (hiprtc -ffp-contract=off): one CUDA C++ */
#define PD_CUDA 1
#define PD_G
#define PD_L
#define PD_GLOBAL extern "C" __global__
#define PD_FN static __device__ __forceinline__
#define PD_FN_NOINLINE static __device__
typedef long long i64;
typedef long long i64a;
typedef unsigned long long u64;
#define PD_PTR(a) ((real *)(u64)(a))
#define PD_LOOP(t, n) \
    for (i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x; t < (i64)(n); t += (i64)blockDim.x * gridDim.x)
PD_FN void pd_max_i(int *p, int v) { atomicMax(p, v); }
PD_FN void pd_or_i(int *p, int v) { atomicOr(p, v); }
PD_FN void pd_max_nonneg(double *p, double v) { atomicMax((u64 *)p, (u64)__double_as_longlong(v)); }
PD_FN real pd_fabs(real x) { return fabs(x); }
PD_FN int pd_isnan(real x) { return isnan(x); }
PD_FN int pd_signbit(real x) { return signbit(x); }
PD_FN real pd_sqrt(real x) { return sqrt(x); }
PD_FN real pd_fma(real a, real b, real c) { return fma(a, b, c); }
/* a 32-lane group's operations (coop.py's lines, convec_gen's columns): a CUDA warp, or half
   of an AMD 64-lane wavefront -- width-32 shuffles stay in the half, a ballot is masked to it */
#if defined(__HIPCC_RTC__) || defined(__HIPCC__)
#define PD_HIP 1
#define pd_shfl(v, l) __shfl((v), (int)(l), 32)
#define pd_shfl_up(v, d) __shfl_up((v), (unsigned)(d), 32)
#define pd_shfl_xor(v, d) __shfl_xor((v), (int)(d), 32)
PD_FN unsigned pd_ballot(int b) { return (unsigned)(__ballot(b) >> (__lane_id() & 32)); }
PD_FN int pd_any(int b) { return pd_ballot(b != 0) != 0u; }
PD_FN int pd_all(int b) { return pd_ballot(b != 0) == 0xffffffffu; }
PD_FN void pd_group_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#else
#define pd_shfl(v, l) __shfl_sync(0xffffffffu, (v), (int)(l))
#define pd_shfl_up(v, d) __shfl_up_sync(0xffffffffu, (v), (unsigned)(d))
#define pd_shfl_xor(v, d) __shfl_xor_sync(0xffffffffu, (v), (int)(d))
#define pd_any(b) __any_sync(0xffffffffu, (int)(b))
#define pd_all(b) __all_sync(0xffffffffu, (int)(b))
#define pd_group_sync() __syncwarp()
#endif
#else
#include <math.h>
#include <string.h>
#define PD_G
#define PD_L
#define PD_GLOBAL extern "C"
#define PD_FN static inline
#define PD_FN_NOINLINE static
typedef long long i64;
typedef long long i64a;
typedef unsigned long long u64;
typedef unsigned int u32;
#define PD_PTR(a) ((real *)(u64)(a))
#if defined(PD_F32) && defined(__aarch64__)
/* the GPU's float subnormal flushing (FPCR.FZ), for the kernel's duration */
struct pd_fz_guard {
    u64 old;
    pd_fz_guard()
    {
        __asm__ volatile("mrs %0, fpcr" : "=r"(old));
        u64 fz = old | (1ull << 24);
        __asm__ volatile("msr fpcr, %0" ::"r"(fz));
    }
    ~pd_fz_guard() { __asm__ volatile("msr fpcr, %0" ::"r"(old)); }
};
#define PD_LOOP(t, n) \
    pd_fz_guard pd_fz_; \
    for (i64 t = 0; t < (i64)(n); t++)
#elif defined(PD_F32)
#error "PD_F32 host build: only arm64's FZ is known to reproduce the Metal GPU"
#else
#define PD_LOOP(t, n) for (i64 t = 0; t < (i64)(n); t++)
#endif
PD_FN void pd_max_i(int *p, int v) { if (v > *p) *p = v; }
PD_FN void pd_or_i(int *p, int v) { *p |= v; }
PD_FN void pd_max_nonneg(real *p, real v)
{
    real cur = *p;
    if (sizeof(real) == 8) {
        u64 a, b;
        memcpy(&a, &cur, 8);
        memcpy(&b, &v, 8);
        if (b > a) *p = v;
    } else {
        u32 a, b;
        memcpy(&a, &cur, 4);
        memcpy(&b, &v, 4);
        if (b > a) *p = v;
    }
}
PD_FN u32 pd_f2u(float x)
{
    u32 b;
    memcpy(&b, &x, 4);
    return b;
}
PD_FN float pd_u2f(u32 b)
{
    float x;
    memcpy(&x, &b, 4);
    return x;
}
PD_FN real pd_fabs(real x) { return fabs(x); }
PD_FN int pd_isnan(real x) { return isnan(x); }
PD_FN int pd_signbit(real x) { return signbit(x); }
PD_FN real pd_sqrt(real x) { return sqrt(x); }
PD_FN real pd_fma(real a, real b, real c) { return fma(a, b, c); } /* explicit: one rounding everywhere */
#endif

/* NumPy's float semantics, which the reference's arrays are computed in. */
/* np.maximum / np.minimum: NaN propagates, -0 < +0 (arm64 and x86 SIMD paths agree on values;
   only the sign of a zero result could differ, which == does not see) */
PD_FN real np_maximum(real a, real b)
{
    if (pd_isnan(a)) return a;
    if (pd_isnan(b)) return b;
    if (a == b) return pd_signbit(a) ? b : a;
    return a > b ? a : b;
}
PD_FN real np_minimum(real a, real b)
{
    if (pd_isnan(a)) return a;
    if (pd_isnan(b)) return b;
    if (a == b) return pd_signbit(a) ? a : b;
    return a < b ? a : b;
}
/* np.clip's ufunc: _NPY_MIN(_NPY_MAX(x, lo), hi), _NPY_MAX(a, b) = isnan(a) ? a : (a > b ? a : b) */
PD_FN real np_clip(real x, real lo, real hi)
{
    real t = pd_isnan(x) ? x : (x > lo ? x : lo);
    return pd_isnan(t) ? t : (t < hi ? t : hi);
}
/* kernels._limit: max(min(s, abs(rm)), -abs(rm)) as np.clip(s, -abs(rm), abs(rm)) */
PD_FN real pd_limit(real s, real rm)
{
    real a = pd_fabs(rm);
    return np_clip(s, -a, a);
}

/* kernels._slopes_fluxes for one face and tracer: f, pf, and the two advected cross slopes. */
PD_FN void pd_face(real flux, real c, real rm_u, real s_u, real t_u, real w_u, PD_L real *f,
                   PD_L real *pf, PD_L real *ft, PD_L real *fw)
{
    real fl = flux >= PD_R(0.0) ? c * (rm_u + (PD_R(1.0) - c) * s_u) : c * (rm_u - (PD_R(1.0) + c) * s_u);
    *f = fl;
    *pf = flux * (c * c * s_u - PD_R(3.0) * fl);
    *ft = c * t_u;
    *fw = c * w_u;
}

/* kernels._along_slope_update for one cell: lo/hi the cell's lower/upper face. */
PD_FN real pd_slope_update(real s, real pf_lo, real pf_hi, real a_lo, real a_hi, real rm_new,
                             real f_lo, real f_hi, real m_old, real m_new)
{
    return s + (pf_lo - pf_hi - (a_lo - a_hi) * s + PD_R(3.0) * ((a_lo + a_hi) * rm_new - (f_lo + f_hi) * m_old)) / m_new;
}

#ifndef PD_NT_MAX
#define PD_NT_MAX 16 /* tracers a thread holds face values for; checked by the caller */
#endif

#endif

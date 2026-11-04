/* Host build only: the open twins of the libraries the CPU reference calls, under the names it
   looks them up by, so that `replay --lapack <this .so> --libm <this .so>` computes with the
   functions the device runs.
     dgetrf_ / dgetri_   LAPACK ABI, convec.cuh's reference-LAPACK transcription
     exp / log / pow     pmath.cuh (fdlibm)
     cos / sin           the system libm's: the geometry the device uses is computed on the host */
#if !defined(PD_CUDA) && !defined(PD_F32)
#include <dlfcn.h>

extern "C" void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info)
{
    *info = pd_dgetrf2(*m, *n, a, *lda, ipiv);
}

extern "C" void dgetri_(const int *n, double *a, const int *lda, const int *ipiv, double *work, const int *lwork,
                        int *info)
{
    (void)lwork;
    *info = pd_dgetri(*n, a, *lda, ipiv, work);
}

extern "C" double exp(double x) { return pm_exp(x); }
extern "C" double log(double x) { return pm_log(x); }
extern "C" double pow(double x, double y) { return pm_pow(x, y); }

typedef double (*pd_fn1)(double);
static pd_fn1 pd_system(const char *name)
{
#ifdef __APPLE__
    void *h = dlopen("/usr/lib/libSystem.B.dylib", RTLD_LAZY | RTLD_LOCAL);
#else
    void *h = dlopen("libm.so.6", RTLD_LAZY | RTLD_LOCAL);
#endif
    return (pd_fn1)dlsym(h, name);
}
extern "C" double cos(double x)
{
    static pd_fn1 f = 0;
    if (!f) f = pd_system("cos");
    return f(x);
}
extern "C" double sin(double x)
{
    static pd_fn1 f = 0;
    if (!f) f = pd_system("sin");
    return f(x);
}
#endif

#if !defined(PD_CUDA) && defined(PD_F32)
/* the float32 build's exp / log / pow, for tests/test_metal.py's accuracy check */
extern "C" void pd_test_exp(const float *x, float *out, long long n)
{
    PD_LOOP(i, n) out[i] = pm_exp(x[i]);
}
extern "C" void pd_test_log(const float *x, float *out, long long n)
{
    PD_LOOP(i, n) out[i] = pm_log(x[i]);
}
extern "C" void pd_test_pow(const float *x, const float *y, float *out, long long n)
{
    PD_LOOP(i, n) out[i] = pm_pow(x[i], y[i]);
}
#endif

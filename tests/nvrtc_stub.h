/* What NVRTC provides without includes, declared for clang's CUDA front end: tests/test_gpu.py
   compiles csrc/all.cu and convec_gen's CUDA passes for sm_80 with it (no GPU, no CUDA SDK). */
#define __CUDACC__ 1
#define __CUDACC_RTC__ 1
#include <__clang_cuda_builtin_vars.h>
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ __attribute__((always_inline)) inline
__device__ double fabs(double);
__device__ double sqrt(double);
__device__ double fma(double, double, double);
__device__ int isnan(double);
__device__ int signbit(double);
__device__ double floor(double);
__device__ double ceil(double);
__device__ int atomicOr(int *, int);
__device__ int atomicMax(int *, int);
__device__ unsigned long long atomicMax(unsigned long long *, unsigned long long);
__device__ double __shfl_sync(unsigned, double, int, int = 32);
__device__ int __shfl_sync(unsigned, int, int, int = 32);
__device__ double __shfl_up_sync(unsigned, double, unsigned, int = 32);
__device__ int __any_sync(unsigned, int);
__device__ long long __double_as_longlong(double);
__device__ double __longlong_as_double(long long);
__device__ int __double2hiint(double);
__device__ int __double2loint(double);
__device__ double __hiloint2double(int, int);
__device__ void __syncwarp(unsigned = 0xffffffffu);
__device__ int __all_sync(unsigned, int);
__device__ double __shfl_xor_sync(unsigned, double, int, int = 32);
__device__ double fmax(double, double);
